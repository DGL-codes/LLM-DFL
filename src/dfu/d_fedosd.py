"""D-FedOSD: 基于正交子空间下降的去中心化联邦遗忘方法。

流程：
1. 遗忘阶段：
   - 目标客户端已退出，不参与 DFU 更新
   - 每个被选中的保留客户端从自己的 DFL 最终模型出发
   - 在本地状态上用目标客户端 forget set 计算 UCE 遗忘梯度
   - 用本节点及其环上邻居的 retain 梯度计算正交投影方向
   - 更新各自本地模型，不复制 target agent 模型
   - 重复 unlearn_rounds 轮
2. 恢复阶段：
   - 移除目标客户端，构建新环状拓扑
   - 幸存客户端训练并投影梯度
   - 环状聚合
   - 重复 recovery_rounds 轮
"""
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from copy import deepcopy

from .snapshot_loader import SnapshotLoader
from .verification import UnlearningVerifier, UnlearningMetrics
from .agent_selection import select_agents, AgentSelectionResult
from .lora_param_selection import (
    compute_module_sensitivities,
    compute_module_sensitivities_relative,
    select_lora_modules,
    select_lora_modules_top_ratio,
    get_lora_module_keys,
    LoRASelectionResult
)
from .losses import GradAscentLoss, UCELoss
from .trainer import DFURingTopology, DFUAgent
from ..models.lora_model import LoRAModelWrapper
from ..models.trainer import LLMTrainer
from ..models.multi_agent_eval import build_final_stats, evaluate_lora_states
from ..data.base import Sample
from ..data.collator import LLMCollator
from ..data.partitioner import PartitionInfo
from ..dfl.agent import RingTopology
from ..utils.determinism import derive_seed


class DFedOSDAgent(DFUAgent):
    """D-FedOSD Agent，支持梯度计算和正交投影。"""

    def __init__(
        self,
        agent_id: int,
        local_samples: List[Sample],
        model: LoRAModelWrapper,
        collator: LLMCollator,
        lr: float = 1e-4,
        device: str = "cuda",
        selected_param_keys: Optional[set] = None,
        optimizer_name: str = "adamw",
        is_target: bool = False,
        forget_samples: Optional[List[Sample]] = None,
        forget_loss: str = "uce",
    ):
        super().__init__(
            agent_id=agent_id,
            local_samples=local_samples,
            model=model,
            collator=collator,
            lr=lr,
            device=device,
            selected_param_keys=selected_param_keys,
            optimizer_name=optimizer_name
        )
        self.is_target = is_target
        self.forget_samples = forget_samples if forget_samples else []
        self.forget_loss_name = str(forget_loss or "uce").lower().strip()
        if self.forget_loss_name == "grad_ascent":
            self.forget_loss = GradAscentLoss()
        elif self.forget_loss_name == "uce":
            self.forget_loss = UCELoss()
        else:
            raise ValueError(
                f"Unsupported FedOSD forget_loss={forget_loss!r}; expected 'uce' or 'grad_ascent'."
            )

    def compute_gradient(
        self,
        state: Dict[str, torch.Tensor],
        samples: List[Sample],
        loss_fn,
        batch_size: int = 4,
        show_progress: bool = False,
        progress_desc: Optional[str] = None,
        progress_position: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """计算给定样本上的梯度。
        
        Args:
            state: 模型状态
            samples: 样本列表
            loss_fn: 损失函数
            batch_size: 批次大小
            
        Returns:
            梯度字典
        """
        self.model.set_lora_state_dict(state)
        self.model.model.zero_grad()
        
        # 冻结未选中的参数
        self._freeze_unselected_params()
        
        from torch.utils.data import DataLoader
        dataloader = DataLoader(
            samples, batch_size=batch_size, shuffle=False,
            collate_fn=self.collator
        )
        
        total_loss = 0.0
        num_batches = 0
        
        batch_iter = dataloader
        pbar = None
        if show_progress:
            pbar_kwargs = {
                "desc": progress_desc or "Gradient",
                "dynamic_ncols": True,
                "leave": True,
                "disable": False
            }
            if progress_position is not None:
                pbar_kwargs["position"] = progress_position
            pbar = tqdm(dataloader, **pbar_kwargs)
            batch_iter = pbar

        for batch in batch_iter:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            outputs = self.model.forward(input_ids, attention_mask, labels)
            logits = outputs.logits
            
            loss = loss_fn(logits, labels)
            loss.backward()
            
            total_loss += loss.item()
            num_batches += 1

        if pbar is not None:
            pbar.close()
        
        # 收集梯度
        gradients = {}
        for name, param in self.model.model.named_parameters():
            if param.grad is not None and "lora" in name.lower():
                grad = param.grad.clone().cpu()
                if num_batches > 0:
                    grad = grad / float(num_batches)
                gradients[name] = grad
        
        # 清理梯度
        self.model.model.zero_grad()
        self._unfreeze_all_params()
        
        return gradients

    @staticmethod
    def _gradient_norm(gradient: Dict[str, torch.Tensor]) -> float:
        return float(sum((v.float() ** 2).sum().item() for v in gradient.values()) ** 0.5)

    @staticmethod
    def _clip_gradient(
        gradient: Dict[str, torch.Tensor],
        max_norm: Optional[float],
    ) -> Tuple[Dict[str, torch.Tensor], float, float]:
        """Clip a CPU gradient dict, mirroring normal LoRA training stability."""
        raw_norm = DFedOSDAgent._gradient_norm(gradient)
        if max_norm is None or max_norm <= 0 or raw_norm <= max_norm:
            return gradient, raw_norm, raw_norm
        scale = float(max_norm) / (raw_norm + 1e-12)
        clipped = {k: v * scale for k, v in gradient.items()}
        return clipped, raw_norm, float(max_norm)

    @staticmethod
    def average_gradients(gradients: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Average gradient dictionaries over their common LoRA keys."""
        if not gradients:
            return {}
        keys = [
            k for k in gradients[0].keys()
            if all(k in grad for grad in gradients)
        ]
        if not keys:
            return {}
        return {
            k: torch.stack([grad[k].float().cpu() for grad in gradients], dim=0).mean(dim=0)
            for k in keys
        }

    def compute_unlearn_gradient(
        self,
        state: Dict[str, torch.Tensor],
        batch_size: int = 4,
        show_progress: bool = False,
        progress_desc: Optional[str] = None,
        progress_position: Optional[int] = None,
        max_samples: Optional[int] = None,
        forget_samples: Optional[List[Sample]] = None
    ) -> Dict[str, torch.Tensor]:
        """计算遗忘梯度（使用 UCE Loss）。
        
        Args:
            max_samples: 最大采样数量，None 表示使用全部样本
            forget_samples: 外部传入的目标遗忘样本；None 时使用 agent 自身缓存样本
        """
        import random
        
        samples = forget_samples if forget_samples is not None else self.forget_samples
        if max_samples is not None and len(samples) > max_samples:
            samples = random.sample(samples, max_samples)
        
        return self.compute_gradient(
            state,
            samples,
            self.forget_loss,
            batch_size,
            show_progress=show_progress,
            progress_desc=progress_desc,
            progress_position=progress_position
        )

    def compute_retain_gradient(
        self,
        state: Dict[str, torch.Tensor],
        batch_size: int = 4,
        show_progress: bool = False,
        progress_desc: Optional[str] = None,
        progress_position: Optional[int] = None,
        max_samples: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """计算保留梯度（使用标准 CE Loss）。
        
        Args:
            max_samples: 最大采样数量，None 表示使用全部样本
        """
        import torch.nn.functional as F
        import random
        
        def ce_loss(logits, labels):
            return F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
        
        # 采样样本
        samples = self.local_samples
        if max_samples is not None and len(samples) > max_samples:
            samples = random.sample(samples, max_samples)
        
        return self.compute_gradient(
            state,
            samples,
            ce_loss,
            batch_size,
            show_progress=show_progress,
            progress_desc=progress_desc,
            progress_position=progress_position
        )

    def retain_train(
        self,
        init_state: Dict[str, torch.Tensor],
        epochs: int = None,
        local_steps: int = None,
        batch_size: int = 4,
        grad_accum_steps: int = 2,
        show_progress: bool = True,
        progress_desc: Optional[str] = None,
        progress_position: Optional[int] = None,
        seed: Optional[int] = None
    ) -> Tuple[Dict[str, torch.Tensor], float]:
        """执行正常训练（用于恢复阶段）。
        
        Args:
            epochs: 训练的epoch数（遍历所有数据）
            local_steps: 训练的步数（与DFL一致，优先于epochs）
            
        注意：如果同时指定epochs和local_steps，优先使用local_steps
        """
        self.model.set_lora_state_dict(init_state)
        self._freeze_unselected_params()

        avg_loss = self.trainer.train_local(
            self.local_samples,
            self.collator,
            local_steps=local_steps,
            epochs=epochs,
            batch_size=batch_size,
            grad_accum_steps=grad_accum_steps,
            show_progress=show_progress,
            progress_desc=progress_desc,
            progress_position=progress_position,
            progress_leave=True,
            seed=seed,
            reset_optimizer=True,
        )

        self._unfreeze_all_params()
        trained_state = self.model.get_lora_state_dict()
        # 彻底清理优化器状态和梯度，避免显存累积
        if self.trainer.optimizer is not None:
            self.trainer.optimizer.zero_grad(set_to_none=True)
            for group in self.trainer.optimizer.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        p.grad = None
                    state = self.trainer.optimizer.state.get(p, {})
                    state.clear()
            self.trainer.optimizer.state.clear()
        self.trainer.optimizer = None
        self.trainer.scheduler = None
        
        return trained_state, avg_loss

    def retain_train_projected(
        self,
        init_state: Dict[str, torch.Tensor],
        forget_samples: List[Sample],
        local_steps: int = None,
        epochs: int = None,
        batch_size: int = 4,
        grad_accum_steps: int = 2,
        retain_grad_samples: Optional[int] = 50,
        forget_grad_samples: Optional[int] = None,
        lr: Optional[float] = None,
        projection_strength: float = 1.0,
        show_progress: bool = True,
        progress_desc: Optional[str] = None,
        progress_position: Optional[int] = None,
        seed: Optional[int] = None
    ) -> Tuple[Dict[str, torch.Tensor], float]:
        """Projected retained training for FedOSD recovery.

        FedOSD should not run ordinary retained-data training after the
        unlearning step, because unconstrained recovery can move the model back
        along the forgotten-data direction.  This local routine computes the
        retained CE gradient, removes its component along the forget UCE
        gradient, and then applies the projected retained update.
        """
        import random
        import torch.nn.functional as F

        def ce_loss(logits, labels):
            return F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )

        def choose_subset(pool: List[Sample], max_n: Optional[int]) -> List[Sample]:
            if max_n is None or max_n <= 0 or len(pool) <= max_n:
                return list(pool)
            return rng.sample(pool, max_n)

        def apply_update(
            state: Dict[str, torch.Tensor],
            gradient: Dict[str, torch.Tensor],
            step_lr: float,
        ) -> Dict[str, torch.Tensor]:
            new_state = {}
            for key, value in state.items():
                if key in gradient:
                    new_state[key] = value.cpu() - step_lr * gradient[key].cpu()
                else:
                    new_state[key] = value.cpu()
            return new_state

        rng = random.Random(seed)
        retain_pool = list(self.local_samples)
        forget_pool = list(forget_samples or self.forget_samples)
        if not retain_pool or not forget_pool:
            return {k: v.clone().cpu() for k, v in init_state.items()}, 0.0

        step_lr = float(lr if lr is not None else self.lr)
        steps = local_steps if (local_steps is not None and int(local_steps) > 0) else None
        if steps is None:
            steps = max(1, int(epochs or 1))
        steps = max(1, int(steps))

        effective_retain_n = retain_grad_samples
        if effective_retain_n is None or effective_retain_n <= 0:
            effective_retain_n = max(1, int(batch_size) * max(1, int(grad_accum_steps)))

        state = {k: v.clone().cpu() for k, v in init_state.items()}
        norms: List[float] = []

        step_iter = range(steps)
        pbar = None
        if show_progress:
            pbar_kwargs = {
                "desc": progress_desc or f"Projected retain A{self.agent_id}",
                "dynamic_ncols": True,
                "leave": True,
                "disable": False,
            }
            if progress_position is not None:
                pbar_kwargs["position"] = progress_position
            pbar = tqdm(step_iter, **pbar_kwargs)
            step_iter = pbar

        for _ in step_iter:
            retain_subset = choose_subset(retain_pool, effective_retain_n)
            forget_subset = choose_subset(forget_pool, forget_grad_samples)

            retain_grad = self.compute_gradient(
                state,
                retain_subset,
                ce_loss,
                batch_size=batch_size,
                show_progress=False,
            )
            forget_grad = self.compute_unlearn_gradient(
                state,
                batch_size=batch_size,
                show_progress=False,
                max_samples=None,
                forget_samples=forget_subset,
            )

            common_keys = [k for k in retain_grad.keys() if k in forget_grad]
            if common_keys:
                retain_grad = {k: retain_grad[k] for k in common_keys}
                forget_grad = {k: forget_grad[k] for k in common_keys}
                projected_retain = compute_orthogonal_direction(
                    retain_grad,
                    [forget_grad],
                    projection_strength=projection_strength,
                )
            else:
                projected_retain = retain_grad

            projected_retain, raw_norm, clipped_norm = self._clip_gradient(
                projected_retain,
                getattr(self.trainer, "max_grad_norm", 1.0),
            )
            norms.append(float(clipped_norm))
            state = apply_update(state, projected_retain, step_lr)

            del retain_grad, forget_grad, projected_retain

        if pbar is not None:
            pbar.close()

        self.model.set_lora_state_dict(state)
        self.model.model.zero_grad()
        self._unfreeze_all_params()
        return state, float(np.mean(norms)) if norms else 0.0


def compute_orthogonal_direction(
    unlearn_grad: Dict[str, torch.Tensor],
    retain_grads: List[Dict[str, torch.Tensor]],
    projection_strength: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """计算正交投影方向。
    
    d = g_u - A^T(AA^T)^{-1}Ag_u
    其中 A 是保留梯度矩阵
    
    Args:
        unlearn_grad: 遗忘梯度 g_u
        retain_grads: 保留梯度列表 [g_r1, g_r2, ...]
        
    Returns:
        正交投影方向 d
    """
    if not retain_grads:
        return unlearn_grad
    
    # 将梯度展平为向量。只使用所有梯度都同时拥有的 LoRA key，避免
    # 个别模块在某个 batch 上没有梯度时破坏投影计算。
    keys = [
        k for k in unlearn_grad.keys()
        if all(k in retain_grad for retain_grad in retain_grads)
    ]
    if not keys:
        return unlearn_grad
    
    def flatten(grad_dict):
        return torch.cat([grad_dict[k].flatten() for k in keys])
    
    def unflatten(vec):
        result = {}
        offset = 0
        for k in keys:
            shape = unlearn_grad[k].shape
            numel = unlearn_grad[k].numel()
            result[k] = vec[offset:offset + numel].reshape(shape)
            offset += numel
        return result
    
    # g_u: (d,)
    g_u = flatten(unlearn_grad)
    
    # A: (n, d) - 每行是一个保留梯度
    A = torch.stack([flatten(g) for g in retain_grads])
    
    n, d = A.shape
    
    # 计算 AA^T: (n, n)
    AAT = A @ A.T
    
    # 添加正则化以避免奇异矩阵
    AAT = AAT + 1e-6 * torch.eye(n, device=AAT.device)
    
    # 计算伪逆 (AA^T)^{-1}
    try:
        AAT_inv = torch.linalg.inv(AAT)
    except:
        # 如果求逆失败，使用伪逆
        AAT_inv = torch.linalg.pinv(AAT)
    
    # 计算投影: d = g_u - gamma * A^T(AA^T)^{-1}Ag_u
    #
    # gamma=1 是标准 FedOSD 的完整正交投影。去中心化 DSU 只更新部分节点/模块时，
    # 完整投影可能把关键遗忘方向一并削弱；因此保留一个算法级诊断/超参，用于
    # 公平地调节 FedOSD 自身的保留约束强度，而不是改变 DSU 选择机制。
    gamma = min(1.0, max(0.0, float(projection_strength)))
    Ag_u = A @ g_u  # (n,)
    proj = A.T @ (AAT_inv @ Ag_u)  # (d,)
    d_vec = g_u - gamma * proj
    
    return unflatten(d_vec)


class DFedOSDTrainer:
    """D-FedOSD: 基于正交子空间下降的去中心化联邦遗忘方法。"""

    def __init__(
        self,
        snapshot_loader: SnapshotLoader,
        model: LoRAModelWrapper,
        collator: LLMCollator,
        all_samples: List[Sample],
        val_samples: List[Sample],
        test_samples: List[Sample],
        partition: PartitionInfo,
        target_agent: int = 0,
        # D-FedOSD 特有参数
        unlearn_rounds: int = 5,
        unlearn_lr: float = 1e-4,
        recovery_rounds: int = 5,
        recovery_epochs: int = 1,
        recovery_lr: float = 1e-5,
        # 通用参数
        lr: float = 1e-4,
        label_names: Optional[List[str]] = None,
        task_type: str = "classification",
        device: str = "cuda",
        max_eval_samples: Optional[int] = None,
        mia_nonmember_source: str = "test",
        tofu_local_dir: Optional[str] = None,
        # 节点选择参数
        selection_strategy: str = "full",
        selection_ratio: float = None,
        selection_count: int = None,
        selection_seed: int = 42,
        selection_epsilon: float = 0.1,
        tdb_sketch_dim: int = 64,
        tdb_max_intervals: int = 3,
        tdb_round_stride: int = 1,
        tdb_alpha_u: float = 1.0,
        tdb_alpha_p: float = 1.0,
        tdb_alpha_q: float = 0.1,
        tdb_epsilon_u: Optional[float] = None,
        tdb_epsilon_p: Optional[float] = None,
        tdb_tau_q: float = 0.0,
        tdb_exposure_rho: float = 0.8,
        tdb_use_target_similarity: bool = False,
        tdb_time_limit: float = 30.0,
        tdb_aggregation_scope: str = "local",
        # LoRA参数选择参数
        enable_param_selection: bool = False,
        param_selection_ratio: float = 0.5,
        param_epsilon_W: float = 0.1,
        param_random_selection: bool = False,
        param_relative_sensitivity: bool = False,
        param_sensitivity_alpha: float = 1.0,
        param_selection_mode: str = "epsilon",
        param_sensitivity_cache: Optional[str] = None,
        # Retrain TR 分布
        retrain_tr_path: Optional[str] = None,
        optimizer_name: str = "adamw",
        save_lora_states: bool = True,
        projected_recovery: bool = False,
        retain_subspace_mode: str = "separate",
        orthogonal_update_norm: Optional[float] = None,
        projection_strength: float = 1.0,
        forget_loss: str = "uce",
    ):
        """初始化 D-FedOSD Trainer。"""
        self.snapshot_loader = snapshot_loader
        self.snapshot = snapshot_loader.snapshot
        self.model = model
        self.collator = collator
        self.all_samples = all_samples
        self.val_samples = val_samples
        self.test_samples = test_samples
        self.partition = partition
        self.target_agent = target_agent
        
        # D-FedOSD 特有参数
        self.unlearn_rounds = unlearn_rounds
        self.unlearn_lr = unlearn_lr
        self.recovery_rounds = recovery_rounds
        self.recovery_epochs = recovery_epochs
        self.recovery_lr = recovery_lr
        
        # 通用参数
        self.lr = lr
        self.label_names = label_names
        self.task_type = task_type
        self.device = device
        self.max_eval_samples = max_eval_samples
        self.mia_nonmember_source = str(mia_nonmember_source or "test").lower().strip()
        self.tofu_local_dir = tofu_local_dir
        self.selection_strategy = selection_strategy
        self.selection_ratio = selection_ratio
        self.selection_count = selection_count
        self.selection_seed = selection_seed
        self.selection_epsilon = selection_epsilon
        self.tdb_aggregation_scope = str(tdb_aggregation_scope or "local").lower().strip()
        self.optimizer_name = optimizer_name
        self.save_lora_states = bool(save_lora_states)
        self.projected_recovery = bool(projected_recovery)
        self.retain_subspace_mode = str(retain_subspace_mode or "separate").lower().strip()
        if self.retain_subspace_mode not in {"separate", "mean"}:
            raise ValueError(
                f"Unsupported FedOSD retain_subspace_mode={retain_subspace_mode!r}; "
                "expected 'separate' or 'mean'."
            )
        self.orthogonal_update_norm = (
            float(orthogonal_update_norm)
            if orthogonal_update_norm is not None and float(orthogonal_update_norm) > 0
            else None
        )
        self.projection_strength = min(1.0, max(0.0, float(projection_strength)))
        self.forget_loss = str(forget_loss or "uce").lower().strip()
        if self.forget_loss not in {"uce", "grad_ascent"}:
            raise ValueError(
                f"Unsupported FedOSD forget_loss={forget_loss!r}; expected 'uce' or 'grad_ascent'."
            )
        
        # LoRA参数选择
        self.enable_param_selection = enable_param_selection
        self.param_selection_ratio = param_selection_ratio
        self.param_epsilon_W = param_epsilon_W
        self.param_random_selection = param_random_selection
        self.param_relative_sensitivity = param_relative_sensitivity
        self.param_sensitivity_alpha = param_sensitivity_alpha
        self.param_selection_mode = param_selection_mode
        self.param_sensitivity_cache = param_sensitivity_cache
        self.param_selection_result = None
        self.selected_param_keys = None
        
        # 加载 retrain TR 分布
        self.retrain_tr = None
        if retrain_tr_path is not None:
            retrain_tr_path = Path(retrain_tr_path)
            if retrain_tr_path.exists():
                self.retrain_tr = np.load(retrain_tr_path)
                print(f"Loaded retrain TR distribution from {retrain_tr_path}")
        
        # 执行LoRA参数选择
        if enable_param_selection:
            self._perform_param_selection()
        
        # 准备 forget/retain 样本
        forget_indices = sorted(partition.agent_indices[target_agent])
        self.forget_samples = [all_samples[i] for i in forget_indices]
        
        # 获取所有幸存agent
        all_surviving = [i for i in range(self.snapshot.num_agents) if i != target_agent]
        
        # 执行节点选择
        num_classes = len(label_names) if label_names else 20
        all_labels = [s.label for s in all_samples] if hasattr(all_samples[0], 'label') else None
        
        self.selection_result = select_agents(
            strategy=selection_strategy,
            surviving_agents=all_surviving,
            ratio=selection_ratio,
            count=selection_count,
            seed=selection_seed,
            agent_indices=partition.agent_indices,
            all_labels=all_labels,
            num_classes=num_classes,
            epsilon=selection_epsilon,
            snapshot_loader=snapshot_loader,
            target_agent=target_agent,
            tdb_sketch_dim=tdb_sketch_dim,
            tdb_max_intervals=tdb_max_intervals,
            tdb_round_stride=tdb_round_stride,
            tdb_alpha_u=tdb_alpha_u,
            tdb_alpha_p=tdb_alpha_p,
            tdb_alpha_q=tdb_alpha_q,
            tdb_epsilon_u=tdb_epsilon_u,
            tdb_epsilon_p=tdb_epsilon_p,
            tdb_tau_q=tdb_tau_q,
            tdb_exposure_rho=tdb_exposure_rho,
            tdb_use_target_similarity=tdb_use_target_similarity,
            tdb_time_limit=tdb_time_limit,
        )
        
        selected_agents = self.selection_result.selected_agents
        
        # 初始化遗忘阶段拓扑（包含目标客户端）
        self.unlearn_topology = DFURingTopology(
            self.snapshot.num_agents,
            removed_agent=-1,
            selected_agents=list(range(self.snapshot.num_agents))
        )
        
        # 初始化恢复阶段拓扑（移除目标客户端）
        self.recovery_topology = DFURingTopology(
            self.snapshot.num_agents,
            removed_agent=target_agent,
            selected_agents=selected_agents,
            aggregation_weights=self.selection_result.weights,
            aggregation_scope=self.tdb_aggregation_scope,
        )
        
        # 初始化所有 agents
        self.agents: Dict[int, DFedOSDAgent] = {}
        for agent_id in range(self.snapshot.num_agents):
            indices = partition.agent_indices[agent_id]
            local_samples = [all_samples[idx] for idx in indices]
            
            is_target = (agent_id == target_agent)
            
            self.agents[agent_id] = DFedOSDAgent(
                agent_id=agent_id,
                local_samples=local_samples,
                model=model,
                collator=collator,
                lr=self.unlearn_lr if is_target else self.lr,
                device=device,
                selected_param_keys=self.selected_param_keys,
                optimizer_name=optimizer_name,
                is_target=is_target,
                forget_samples=self.forget_samples if is_target else None,
                forget_loss=self.forget_loss,
            )
        
        # 当前模型状态
        self.current_models: Dict[int, Dict[str, torch.Tensor]] = {}
        
        # retain 样本
        retain_indices: List[int] = []
        for aid in all_surviving:
            agent_indices = partition.agent_indices[aid]
            retain_indices.extend(agent_indices)
        retain_indices = sorted(retain_indices)
        self.retain_samples = [all_samples[i] for i in retain_indices]

        # Non-member pool for membership inference (MIA).
        if self.mia_nonmember_source == "val":
            self.nonmember_samples = list(self.val_samples)
        else:
            self.nonmember_samples = list(self.test_samples)
        
        # 创建固定的评估子集（用于加速评估；使用确定性前N条，避免 random.sample 带来的波动）
        eval_limit = self.max_eval_samples if (self.max_eval_samples is not None and self.max_eval_samples > 0) else None
        if eval_limit is None:
            self.retain_samples_eval = self.retain_samples
            self.forget_samples_eval = self.forget_samples
            self.nonmember_samples_eval = self.nonmember_samples
        else:
            self.retain_samples_eval = self.retain_samples[: min(eval_limit, len(self.retain_samples))]
            self.forget_samples_eval = self.forget_samples[: min(eval_limit, len(self.forget_samples))]
            n_forget = len(self.forget_samples_eval)
            self.nonmember_samples_eval = self.nonmember_samples[: min(n_forget, len(self.nonmember_samples))]
        
        # 训练历史
        self.history = {
            "unlearn_phase": {
                "rounds": [],
                "orthogonal_norms": [],
                "unlearn_losses": [],
            },
            "recovery_phase": {
                "rounds": [],
                "agent_losses": {i: [] for i in selected_agents},
                "avg_metrics": [],
                "avg_metrics_std": [],
            },
            "unlearning_metrics": [],
            "tofu_metrics": [],
        }

        # Optional: official TOFU eval samples (retain_perturbed / forgetXX_perturbed).
        # If provided, evaluate_tofu() uses these instead of client training samples.
        self.tofu_external_retain_samples: Optional[List[Sample]] = None
        self.tofu_external_forget_samples: Optional[List[Sample]] = None
        
        # TOFU 评估器
        if self.task_type == "tofu":
            self.init_tofu_evaluator()
        
        self.eval_agent_ids: Optional[List[int]] = None

        # Cache per-agent test-set metrics for final_stats writing.
        self.last_test_per_agent_results: Optional[Dict[int, Dict[str, float]]] = None

    def _perform_param_selection(self):
        """执行 LoRA 参数选择。"""
        print("\n--- LoRA Parameter Selection ---")
        
        if self.param_random_selection:
            sensitivities = compute_module_sensitivities(
                self.snapshot_loader, target_agent=self.target_agent, verbose=False
            )
            all_modules = list(sensitivities.keys())
            num_select = max(1, int(len(all_modules) * self.param_selection_ratio))
            import random
            rng = random.Random(self.selection_seed)
            selected_modules = rng.sample(all_modules, num_select)
            
            self.param_selection_result = LoRASelectionResult(
                selected_modules=selected_modules,
                all_modules=all_modules,
                sensitivities=sensitivities,
                total_sensitivity=sum(sensitivities.values()),
                covered_sensitivity=sum(sensitivities[m] for m in selected_modules),
                selection_ratio=len(selected_modules) / len(all_modules),
                epsilon_W=self.param_epsilon_W
            )
        else:
            sensitivities = None
            cache_path = Path(self.param_sensitivity_cache) if self.param_sensitivity_cache else None
            if cache_path is not None and cache_path.exists():
                print(f"Loading cached sensitivities: {cache_path}")
                with open(cache_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                sensitivities = {str(k): float(v) for k, v in loaded.items()}
            else:
                if self.param_relative_sensitivity:
                    print(
                        f"Computing relative sensitivities (alpha={self.param_sensitivity_alpha}) "
                        f"for target agent {self.target_agent}..."
                    )
                    sensitivities = compute_module_sensitivities_relative(
                        self.snapshot_loader,
                        target_agent=self.target_agent,
                        alpha=self.param_sensitivity_alpha,
                        verbose=True
                    )
                else:
                    sensitivities = compute_module_sensitivities(
                        self.snapshot_loader, target_agent=self.target_agent, verbose=True
                    )
                if cache_path is not None:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(cache_path, "w", encoding="utf-8") as f:
                        json.dump(sensitivities, f, ensure_ascii=False, indent=2)
                    print(f"Saved sensitivities cache: {cache_path}")

            mode = (self.param_selection_mode or "epsilon").lower()
            if mode == "top_ratio":
                self.param_selection_result = select_lora_modules_top_ratio(
                    sensitivities=sensitivities,
                    ratio=self.param_selection_ratio,
                    verbose=True,
                )
            else:
                self.param_selection_result = select_lora_modules(
                    sensitivities=sensitivities,
                    epsilon_W=self.param_epsilon_W,
                    verbose=True,
                    max_ratio=self.param_selection_ratio
                )
        
        sample_state = self.snapshot_loader.load_agent_state(
            self.snapshot.available_rounds[0], self.target_agent
        )
        self.selected_param_keys = get_lora_module_keys(
            sample_state, self.param_selection_result.selected_modules
        )
        print(f"Selected {len(self.param_selection_result.selected_modules)} modules")

    def initialize_models(self):
        """从 DFL 快照初始化所有客户端模型。"""
        print(f"\n[初始化] 从DFL快照加载初始模型")
        final_round = self.snapshot.available_rounds[-1]
        
        for agent_id in range(self.snapshot.num_agents):
            agent_state = self.snapshot_loader.load_agent_state(final_round, agent_id)
            self.current_models[agent_id] = {k: v.clone() for k, v in agent_state.items()}
            print(f"  Agent {agent_id}: 加载 round_{final_round}")
        
        print(f"[初始化完成]\n")

    def _apply_gradient_update(
        self,
        state: Dict[str, torch.Tensor],
        gradient: Dict[str, torch.Tensor],
        lr: float
    ) -> Dict[str, torch.Tensor]:
        """应用梯度更新：new_state = state - lr * gradient。"""
        new_state = {}
        for key in state.keys():
            if key in gradient:
                new_state[key] = state[key].cpu() - lr * gradient[key].cpu()
            else:
                new_state[key] = state[key].cpu()
        return new_state

    @staticmethod
    def _scale_gradient_to_norm(
        gradient: Dict[str, torch.Tensor],
        target_norm: Optional[float],
    ) -> Tuple[Dict[str, torch.Tensor], float, float]:
        raw_norm = float(sum((v.float() ** 2).sum().item() for v in gradient.values()) ** 0.5)
        if target_norm is None or target_norm <= 0 or raw_norm <= 0:
            return gradient, raw_norm, raw_norm
        scale = float(target_norm) / (raw_norm + 1e-12)
        return {k: v * scale for k, v in gradient.items()}, raw_norm, float(target_norm)

    def _ring_aggregate(
        self,
        agent_models: Dict[int, Dict[str, torch.Tensor]],
        topology: DFURingTopology
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """环状聚合。"""
        new_models = {}
        for agent_id in topology.surviving_agents:
            neighbors = topology.get_neighbors(agent_id)
            all_ids = [agent_id] + neighbors
            
            aggregated = {}
            weights = topology.get_aggregation_weights(agent_id, all_ids)
            for key in agent_models[agent_id].keys():
                # 确保所有张量都在 CPU 上进行聚合
                tensors = [agent_models[i][key].cpu() for i in all_ids]
                stacked = torch.stack(tensors)
                coeffs = torch.tensor(weights, dtype=stacked.dtype, device=stacked.device)
                aggregated[key] = (stacked * coeffs.view(-1, *([1] * (stacked.dim() - 1)))).sum(dim=0)
            
            new_models[agent_id] = aggregated
        return new_models

    def _get_retain_grads_cache_path(
        self,
        retain_grad_samples: Optional[int] = None,
        agent_ids: Optional[List[int]] = None,
        mode: str = "decentralized_local",
    ) -> Path:
        """获取保留梯度缓存文件路径。
        
        缓存路径基于DFL快照目录。路径必须包含参与节点、LoRA key、采样数和算法模式；
        不能再只按 target agent 复用，否则不同 DSU 配置会读到旧梯度。
        """
        import hashlib

        cache_dir = Path(self.snapshot.snapshot_dir) / "retain_grads_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        ids = agent_ids if agent_ids is not None else self.recovery_topology.surviving_agents
        key_names = sorted(self.selected_param_keys or [])
        digest_src = json.dumps(
            {
                "target_agent": self.target_agent,
                "agent_ids": sorted(int(i) for i in ids),
                "retain_grad_samples": retain_grad_samples,
                "mode": mode,
                "aggregation_scope": getattr(self.recovery_topology, "aggregation_scope", None),
                "selected_keys": key_names,
            },
            sort_keys=True,
        )
        digest = hashlib.sha1(digest_src.encode("utf-8")).hexdigest()[:12]
        return cache_dir / f"retain_grads_{mode}_target{self.target_agent}_{digest}.pt"

    def _load_retain_grads_cache(
        self,
        retain_grad_samples: Optional[int] = None,
        agent_ids: Optional[List[int]] = None,
        mode: str = "decentralized_local",
    ) -> Optional[List[Dict[str, torch.Tensor]]]:
        """从磁盘加载保留梯度缓存。"""
        cache_path = self._get_retain_grads_cache_path(
            retain_grad_samples=retain_grad_samples,
            agent_ids=agent_ids,
            mode=mode,
        )
        if cache_path.exists():
            try:
                cache_data = torch.load(cache_path, map_location='cpu')
                print(f"  [磁盘缓存] 从 {cache_path} 加载保留梯度")
                retain_grads = cache_data.get("retain_grads")
                if not isinstance(retain_grads, list) or not retain_grads:
                    print(f"  [磁盘缓存] 缓存格式不正确，将重新计算")
                    return None
                if not all(isinstance(g, dict) for g in retain_grads):
                    print(f"  [磁盘缓存] 缓存格式不正确（非dict），将重新计算")
                    return None
                return retain_grads
            except Exception as e:
                print(f"  [磁盘缓存] 加载失败: {e}，将重新计算")
                return None
        return None

    def _save_retain_grads_cache(
        self,
        retain_grads: List[Dict[str, torch.Tensor]],
        retain_grad_samples: Optional[int] = None,
        agent_ids: Optional[List[int]] = None,
        mode: str = "decentralized_local",
    ):
        """将保留梯度缓存保存到磁盘。"""
        cache_path = self._get_retain_grads_cache_path(
            retain_grad_samples=retain_grad_samples,
            agent_ids=agent_ids,
            mode=mode,
        )
        try:
            cache_data = {
                "retain_grads": retain_grads,
                "target_agent": self.target_agent,
                "num_agents": self.snapshot.num_agents,
                "agent_ids": sorted(agent_ids or self.recovery_topology.surviving_agents),
                "retain_grad_samples": retain_grad_samples,
                "mode": mode,
                "aggregation_scope": getattr(self.recovery_topology, "aggregation_scope", None),
                "grad_keys": sorted(list(retain_grads[0].keys())) if retain_grads else [],
            }
            # Atomic save to avoid partial reads when multiple processes share the same snapshot cache.
            tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
            torch.save(cache_data, tmp_path)
            tmp_path.replace(cache_path)
            print(f"  [磁盘缓存] 保留梯度已保存到 {cache_path}")
        except Exception as e:
            print(f"  [磁盘缓存] 保存失败: {e}")

    def unlearn_phase(
        self,
        batch_size: int = 4,
        verbose: bool = True,
        show_progress: bool = True,
        retain_grad_samples: int = 50,
        forget_grad_samples: Optional[int] = None,
        cache_retain_grads: bool = True,
        save_dir: Optional[str] = None,
        seed: int = 42
    ):
        """遗忘阶段：对每个参与保留客户端做去中心化本地正交遗忘。
        
        每轮：
        1. 每个参与客户端从自己的 DFL 最终模型出发，计算目标 forget set 的 UCE 梯度
        2. 同一客户端及其环上邻居计算 retain 梯度
        3. 对该客户端的遗忘梯度做正交投影并更新本客户端模型
        4. 不复制 target agent 的模型，不做全局聚合
        
        Args:
            retain_grad_samples: 每个 agent 计算保留梯度时的采样数量，默认 50
            forget_grad_samples: 计算目标 agent 遗忘梯度时的采样数量；None 表示使用完整 forget set。
                                这不能和 retain_grad_samples 绑定，否则后门审计中可能只覆盖极少投毒样本。
            cache_retain_grads: 旧版 target-copy 流程可缓存；当前去中心化本地流程中模型每轮变化，
                                为避免旧缓存污染，会强制禁用磁盘缓存
            save_dir: 输出目录（用于持久化缓存）
            seed: 随机种子，用于固定采样顺序，确保不同实验可复现
        """
        # 固定随机种子：为避免“是否命中磁盘缓存”影响采样顺序，这里为每次梯度计算独立设种子。
        import random

        def _seed_everything(seed_value: int) -> None:
            torch.manual_seed(seed_value)
            np.random.seed(seed_value)
            random.seed(seed_value)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed_value)

        def _derive_seed(kind: str, round_idx: int, agent_id: int) -> int:
            kind_id = 0 if kind == "unlearn" else 1
            mixed = seed * 1000003 + kind_id * 10007 + (round_idx + 1) * 1009 + agent_id * 37
            return int(mixed % (2**32 - 1))

        _seed_everything(seed)
        
        print(f"\n{'='*60}")
        print("D-FedOSD: 遗忘阶段")
        print(f"{'='*60}")
        print(f"  遗忘轮数: {self.unlearn_rounds}")
        print(f"  目标客户端: {self.target_agent}")
        print(f"  去中心化参与客户端: {self.recovery_topology.surviving_agents}")
        print(f"  保留梯度采样数: {retain_grad_samples}")
        print(f"  遗忘梯度采样数: {'all' if forget_grad_samples is None else forget_grad_samples}")
        print(f"  遗忘损失: {self.forget_loss}")
        print(f"  保留梯度子空间模式: {self.retain_subspace_mode}")
        print(f"  正交投影强度: {self.projection_strength}")
        print(f"  正交遗忘方向目标范数: {self.orthogonal_update_norm if self.orthogonal_update_norm is not None else 'raw'}")
        if cache_retain_grads:
            print("  缓存保留梯度: 已请求，但当前 decentralized-local 模式会禁用磁盘缓存")
        else:
            print("  缓存保留梯度: 关闭")
        print(f"  随机种子: {seed}")
        
        participant_agents = list(self.recovery_topology.surviving_agents)
        if not participant_agents:
            raise ValueError("D-FedOSD decentralized-local unlearn phase has no participant agents")
        
        round_pbar = None
        if show_progress and self.unlearn_rounds > 0:
            round_pbar = tqdm(
                range(self.unlearn_rounds),
                desc="Unlearn rounds",
                dynamic_ncols=True,
                leave=True
            )
            round_iter = round_pbar
        else:
            round_iter = range(self.unlearn_rounds)

        for round_idx in round_iter:
            if verbose:
                print(f"\n--- 遗忘轮次 {round_idx + 1}/{self.unlearn_rounds} ---")
            
            updated_models: Dict[int, Dict[str, torch.Tensor]] = {}
            round_orth_norms: Dict[int, float] = {}
            participant_pbar = tqdm(
                participant_agents,
                desc=f"Decentralized unlearn r{round_idx + 1}",
                dynamic_ncols=True,
                leave=True,
                disable=not show_progress,
            )
            for agent_id in participant_pbar:
                agent = self.agents[agent_id]

                _seed_everything(_derive_seed("unlearn", round_idx, agent_id))
                unlearn_grad = agent.compute_unlearn_gradient(
                    self.current_models[agent_id],
                    batch_size=batch_size,
                    show_progress=False,
                    progress_desc=f"Unlearn grad A{agent_id}",
                    progress_position=1 if show_progress else None,
                    max_samples=forget_grad_samples,
                    forget_samples=self.forget_samples,
                )

                local_ids = [agent_id] + self.recovery_topology.get_neighbors(agent_id)
                # 去重并确保只使用当前参与节点，避免把未参与 DSU 的节点纳入本地投影。
                seen = set()
                local_ids = [
                    int(aid) for aid in local_ids
                    if int(aid) in participant_agents and not (int(aid) in seen or seen.add(int(aid)))
                ]

                retain_grads = []
                for retain_agent_id in local_ids:
                    retain_agent = self.agents[retain_agent_id]
                    _seed_everything(_derive_seed("retain", round_idx, retain_agent_id))
                    # The projected update is applied to `agent_id`, so all retain
                    # gradients used in the projection should be evaluated at the
                    # same parameter point as the forget gradient. Neighbor agents
                    # contribute retain data/gradients, not their own already
                    # diverged model states.
                    retain_grad = retain_agent.compute_retain_gradient(
                        self.current_models[agent_id],
                        batch_size=batch_size,
                        show_progress=False,
                        progress_desc=f"Retain grad A{retain_agent_id}",
                        progress_position=1 if show_progress else None,
                        max_samples=retain_grad_samples,
                    )
                    retain_grads.append(retain_grad)

                if self.retain_subspace_mode == "mean" and retain_grads:
                    mean_retain_grad = DFedOSDAgent.average_gradients(retain_grads)
                    retain_grads = [mean_retain_grad] if mean_retain_grad else retain_grads

                orthogonal_dir = compute_orthogonal_direction(
                    unlearn_grad,
                    retain_grads,
                    projection_strength=self.projection_strength,
                )
                orthogonal_dir, raw_orth_norm, applied_orth_norm = self._scale_gradient_to_norm(
                    orthogonal_dir,
                    self.orthogonal_update_norm,
                )
                round_orth_norms[int(agent_id)] = applied_orth_norm
                updated_models[int(agent_id)] = self._apply_gradient_update(
                    self.current_models[agent_id],
                    orthogonal_dir,
                    self.unlearn_lr,
                )

                del unlearn_grad, retain_grads, orthogonal_dir

            for agent_id, state in updated_models.items():
                self.current_models[agent_id] = state

            avg_orth_norm = float(np.mean(list(round_orth_norms.values()))) if round_orth_norms else 0.0
            if verbose:
                print(f"  平均正交方向范数: {avg_orth_norm:.6f}")
            if round_pbar is not None:
                round_pbar.set_postfix({"avg_orth": f"{avg_orth_norm:.4f}"})
            
            # 记录历史
            self.history["unlearn_phase"]["rounds"].append(round_idx + 1)
            self.history["unlearn_phase"]["orthogonal_norms"].append(avg_orth_norm)
            self.history["unlearn_phase"].setdefault("orthogonal_norms_by_agent", []).append(
                {str(k): float(v) for k, v in round_orth_norms.items()}
            )
        
        print(f"\n遗忘阶段完成：未复制 target agent 模型，保留节点只更新自己的本地模型")
        if round_pbar is not None:
            round_pbar.close()

    def recovery_phase(
        self,
        batch_size: int = 4,
        grad_accum_steps: int = 2,
        verbose: bool = True,
        show_progress: bool = True,
        local_steps: int = None,
        seed: int = 42
    ):
        """恢复阶段：带正交投影的联邦训练。
        
        每轮：
        1. 幸存客户端正常训练
        2. 环状聚合
        
        Args:
            local_steps: 每轮每个agent的训练步数（与DFL一致）
                        如果为None，则使用recovery_epochs遍历所有数据
            seed: 随机种子，用于固定训练batch顺序，确保不同实验可复现
        """
        import gc
        import random
        
        # 固定随机种子，确保不同实验使用相同的batch顺序
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        print(f"\n{'='*60}")
        print("D-FedOSD: 恢复阶段")
        print(f"{'='*60}")
        print(f"  恢复轮数: {self.recovery_rounds}")
        print(f"  参与客户端: {self.recovery_topology.surviving_agents}")
        print(f"  正交投影恢复: {'开启' if self.projected_recovery else '关闭'}")
        print(f"  随机种子: {seed}")
        effective_local_steps = local_steps if (local_steps is not None and int(local_steps) > 0) else None
        if effective_local_steps is not None:
            print(f"  每轮本地训练步数: {effective_local_steps}")
        else:
            print(f"  每轮本地训练epochs: {self.recovery_epochs}")
        
        round_pbar = None
        if show_progress and self.recovery_rounds > 0:
            round_pbar = tqdm(
                range(self.recovery_rounds),
                desc="Recovery rounds",
                dynamic_ncols=True,
                leave=True
            )
            round_iter = round_pbar
        else:
            round_iter = range(self.recovery_rounds)

        for round_idx in round_iter:
            if verbose:
                print(f"\n--- 恢复轮次 {round_idx + 1}/{self.recovery_rounds} ---")
            
            trained_models: Dict[int, Dict[str, torch.Tensor]] = {}
            round_losses = []
            
            # 幸存客户端正常训练
            for agent_id in self.recovery_topology.surviving_agents:
                agent = self.agents[agent_id]
                
                train_seed = derive_seed(
                    seed,
                    salt="fedosd_recovery",
                    round_idx=round_idx,
                    agent_id=agent_id,
                )
                if self.projected_recovery:
                    recovery_forget_grad_samples = getattr(self, "_last_forget_grad_samples", None)
                    if recovery_forget_grad_samples is None:
                        # The unlearn phase can use the full forget set.  During
                        # recovery, the forget gradient is only a local
                        # constraint that prevents retained training from moving
                        # back along the forgotten direction, so we use the same
                        # bounded sketch size as retained gradients for
                        # practical decentralized execution.
                        recovery_forget_grad_samples = getattr(self, "_last_retain_grad_samples", 50)
                    trained_state, loss = agent.retain_train_projected(
                        init_state=self.current_models[agent_id],
                        forget_samples=self.forget_samples,
                        epochs=self.recovery_epochs if effective_local_steps is None else None,
                        local_steps=effective_local_steps,
                        batch_size=batch_size,
                        grad_accum_steps=grad_accum_steps,
                        retain_grad_samples=getattr(self, "_last_retain_grad_samples", 50),
                        forget_grad_samples=recovery_forget_grad_samples,
                        lr=self.recovery_lr,
                        projection_strength=self.projection_strength,
                        show_progress=show_progress,
                        progress_desc=f"Projected retain A{agent_id}",
                        progress_position=1 if show_progress else None,
                        seed=train_seed
                    )
                else:
                    trained_state, loss = agent.retain_train(
                        init_state=self.current_models[agent_id],
                        epochs=self.recovery_epochs if effective_local_steps is None else None,
                        local_steps=effective_local_steps,
                        batch_size=batch_size,
                        grad_accum_steps=grad_accum_steps,
                        show_progress=show_progress,
                        progress_desc=f"Retain train A{agent_id}",
                        progress_position=1 if show_progress else None,
                        seed=train_seed
                    )
                trained_models[agent_id] = trained_state
                round_losses.append(loss)
                self.history["recovery_phase"]["agent_losses"][agent_id].append(loss)
            
            avg_loss = np.mean(round_losses) if round_losses else 0.0
            if verbose:
                metric_name = "平均 projected retain norm" if self.projected_recovery else "平均 loss"
                print(f"  {metric_name} = {avg_loss:.4f}")
            if round_pbar is not None:
                round_pbar.set_postfix({"proj": f"{avg_loss:.4f}"} if self.projected_recovery else {"loss": f"{avg_loss:.4f}"})
            
            # 环状聚合
            aggregated = self._ring_aggregate(trained_models, self.recovery_topology)
            for agent_id in self.recovery_topology.surviving_agents:
                self.current_models[agent_id] = aggregated[agent_id]
            
            self.history["recovery_phase"]["rounds"].append(round_idx + 1)
            
            # 清理本轮的临时数据
            del trained_models, aggregated
        
        print(f"\n恢复阶段完成")
        if round_pbar is not None:
            round_pbar.close()

    def _get_eval_agent_ids(self) -> List[int]:
        """获取用于评估的 agent 列表。"""
        if self.eval_agent_ids is not None:
            return [aid for aid in self.eval_agent_ids 
                    if aid in self.recovery_topology.surviving_agents]
        # Default: evaluate on ALL actually selected agents (selected-k).
        return sorted(self.recovery_topology.surviving_agents)

    def set_eval_agent_ids(self, agent_ids: List[int]) -> None:
        self.eval_agent_ids = list(agent_ids)

    def verify_unlearning(self, agent_id: int = None, verbose: bool = True) -> UnlearningMetrics:
        """验证遗忘效果。"""
        if agent_id is None:
            agent_id = self.recovery_topology.surviving_agents[0]
        
        self.model.set_lora_state_dict(self.current_models[agent_id])
        verifier = UnlearningVerifier(self.model, self.collator, self.device)
        
        if verbose:
            print(f"\n--- Unlearning Verification (Agent {agent_id}) ---")
        
        # 使用固定的评估子集（已在初始化时创建）
        return verifier.verify_unlearning(
            self.forget_samples_eval,
            self.retain_samples_eval,
            nonmember_samples=self.nonmember_samples_eval,
            batch_size=8,
            verbose=verbose,
            show_progress=verbose,
        )

    def _evaluate_unlearning_all(self, verbose: bool = True) -> Tuple[Dict[str, float], Dict[str, float]]:
        """对所有评估 agent 运行遗忘验证。"""
        per_agent_metrics: Dict[int, UnlearningMetrics] = {}
        eval_agent_ids = self._get_eval_agent_ids()
        
        for agent_id in eval_agent_ids:
            per_agent_metrics[agent_id] = self.verify_unlearning(
                agent_id=agent_id,
                verbose=(verbose and len(eval_agent_ids) == 1)
            )
        
        mean_metrics, std_metrics = self._aggregate_unlearning_metrics(per_agent_metrics)
        
        if verbose and len(eval_agent_ids) > 1:
            print("\n--- Unlearning Verification (Selected Agents) ---")
            print(f"  MIA AUC: {mean_metrics.get('mia_auc', 0):.4f}±{std_metrics.get('mia_auc', 0):.4f}")
        
        return mean_metrics, std_metrics

    def _aggregate_unlearning_metrics(
        self, per_agent_metrics: Dict[int, UnlearningMetrics]
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """聚合遗忘指标。"""
        if not per_agent_metrics:
            return {}, {}
        
        metric_keys = [
            "mia_auc", "mia_tpr_at_1fpr", "mia_tpr_at_5fpr",
            "forget_loss", "retain_loss", "loss_gap",
            "forget_accuracy", "retain_accuracy", "accuracy_gap"
        ]
        
        mean_metrics, std_metrics = {}, {}
        for key in metric_keys:
            values = [getattr(m, key) for m in per_agent_metrics.values()
                     if getattr(m, key) is not None and not np.isnan(getattr(m, key))]
            mean_metrics[key] = float(np.mean(values)) if values else 0.0
            std_metrics[key] = float(np.std(values)) if values else 0.0
        
        return mean_metrics, std_metrics

    def evaluate_test_set(self, verbose: bool = True) -> Tuple[Dict[str, float], Dict[str, float]]:
        """在 test set 上评估分类性能。
        
        Returns:
            Tuple of (mean_metrics, std_metrics)
        """
        if self.task_type == "tofu" or not self.test_samples or not self.label_names:
            return {}, {}

        eval_agent_ids = self._get_eval_agent_ids()
        lora_states = {int(aid): self.current_models[int(aid)] for aid in eval_agent_ids}

        per_agent_results, summary = evaluate_lora_states(
            model=self.model,
            train_collator=self.collator,
            task_type="classification",
            label_names=self.label_names,
            test_samples=self.test_samples,
            lora_states=lora_states,
            eval_agent_ids=eval_agent_ids,
            max_eval_samples=self.max_eval_samples,
            batch_size=16,
            max_new_tokens_classification=16,
            max_new_tokens_generation=64,
        )

        # Cache per-agent results (used by _save_history for final_stats).
        self.last_test_per_agent_results = per_agent_results

        metric_keys = ["accuracy", "precision", "recall", "macro_f1", "valid_ratio"]
        mean_metrics = {k: float(summary.get(f"{k}_mean", 0.0)) for k in metric_keys}
        std_metrics = {k: float(summary.get(f"{k}_std", 0.0)) for k in metric_keys}

        if verbose:
            print(f"\n--- Test Set Evaluation ---")
            print(f"  Test samples: {len(self.test_samples)}")
            print(
                f"  Mean Accuracy: {mean_metrics.get('accuracy', 0):.4f}±{std_metrics.get('accuracy', 0):.4f}"
            )
            print(
                f"  Mean Macro F1: {mean_metrics.get('macro_f1', 0):.4f}±{std_metrics.get('macro_f1', 0):.4f}"
            )
            print(
                f"  Best Agent: {summary.get('best_agent_id')} "
                f"(Macro F1={summary.get('macro_f1_best', 0):.4f})"
            )

        return mean_metrics, std_metrics

    def init_tofu_evaluator(self):
        """初始化 TOFU 评估器。"""
        if self.task_type != "tofu":
            return
        from .tofu_evaluator import TOFUEvaluator
        self.tofu_evaluator = TOFUEvaluator(
            model=self.model, tokenizer=self.model.tokenizer,
            collator=self.collator,
            device=self.device,
            tofu_local_dir=self.tofu_local_dir,
        )

    def evaluate_tofu(self, agent_id: int = None, verbose: bool = True) -> Dict:
        """使用 TOFU 指标评估。"""
        if not hasattr(self, 'tofu_evaluator'):
            return {}
        
        agent_ids = [agent_id] if agent_id else self._get_eval_agent_ids()
        per_agent_results = {}

        retain_samples = self.tofu_external_retain_samples or self.retain_samples
        forget_samples = self.tofu_external_forget_samples or self.forget_samples
        
        for aid in agent_ids:
            self.model.set_lora_state_dict(self.current_models[aid])
            results = self.tofu_evaluator.evaluate_all(
                retain_samples=retain_samples,
                forget_samples=forget_samples,
                retain_truth_ratios=self.retrain_tr,
                compute_generation=True,
                verbose=(verbose and len(agent_ids) == 1),
                print_summary=(verbose and len(agent_ids) == 1),
                show_progress=verbose,
                max_samples=self.max_eval_samples
            )
            per_agent_results[aid] = results.to_dict()
        
        return self._aggregate_tofu_metrics(per_agent_results)

    def set_tofu_external_eval_samples(self, *, retain_samples: List[Sample], forget_samples: List[Sample]) -> None:
        """Set official TOFU eval samples for retain/forget."""
        self.tofu_external_retain_samples = list(retain_samples)
        self.tofu_external_forget_samples = list(forget_samples)

    def _aggregate_tofu_metrics(self, per_agent_results: Dict[int, Dict]) -> Dict:
        """聚合 TOFU 指标。"""
        if not per_agent_results:
            return {}
        
        keys = list(next(iter(per_agent_results.values())).keys())
        aggregated = {}
        
        for key in keys:
            values = [m.get(key, np.nan) for m in per_agent_results.values()]
            values = [v for v in values if isinstance(v, (int, float, np.floating)) and not np.isnan(v)]
            aggregated[key] = float(np.mean(values)) if values else 0.0
            aggregated[f"{key}_std"] = float(np.std(values)) if values else 0.0
        
        return aggregated

    def run(
        self,
        batch_size: int = 4,
        grad_accum_steps: int = 2,
        save_dir: Optional[str] = None,
        eval_every: int = 1,
        show_progress: bool = True,
        retain_grad_samples: int = 50,
        forget_grad_samples: Optional[int] = None,
        cache_retain_grads: bool = True,
        recovery_local_steps: int = None,
        skip_final_eval: bool = False
    ):
        """运行完整的 D-FedOSD 遗忘流程。
        
        Args:
            retain_grad_samples: 遗忘阶段每个 agent 计算梯度时的采样数量
            forget_grad_samples: 目标 agent 遗忘梯度采样量；None 表示完整 forget set
            cache_retain_grads: 是否缓存保留梯度（幸存客户端模型在遗忘阶段不变，可复用）
            recovery_local_steps: 恢复阶段每轮每个agent的训练步数（与DFL一致）
                                 如果为None，则使用recovery_epochs遍历所有数据
        """
        print(f"\n{'='*60}")
        print("Starting D-FedOSD (Decentralized Federated Unlearning)")
        print(f"{'='*60}")
        print(f"Target agent to forget: {self.target_agent}")
        print(f"Unlearn rounds: {self.unlearn_rounds}")
        print(f"Recovery rounds: {self.recovery_rounds}")
        print(f"Retain subspace mode: {self.retain_subspace_mode}")
        print(f"Projection strength: {self.projection_strength}")
        if recovery_local_steps is not None:
            print(f"Recovery local steps: {recovery_local_steps}")
        else:
            print(f"Recovery epochs: {self.recovery_epochs}")
        print(f"Retain grad samples: {retain_grad_samples}")
        print(f"Forget grad samples: {'all' if forget_grad_samples is None else forget_grad_samples}")
        print(f"Cache retain grads: {cache_retain_grads}")
        
        print(f"\n--- Agent Selection ---")
        print(f"Selection strategy: {self.selection_strategy}")
        print(f"Selected agents for recovery: {self.recovery_topology.surviving_agents}")
        
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            curves_dir = save_path / "curves"
            curves_dir.mkdir(exist_ok=True)

        self._last_retain_grad_samples = retain_grad_samples
        self._last_forget_grad_samples = forget_grad_samples
        
        # 初始化模型
        self.initialize_models()
        
        # 1. 遗忘阶段
        self.unlearn_phase(
            batch_size=batch_size, 
            show_progress=show_progress,
            retain_grad_samples=retain_grad_samples,
            forget_grad_samples=forget_grad_samples,
            cache_retain_grads=cache_retain_grads,
            seed=self.selection_seed,
        )
        
        print("\n[去中心化本地遗忘] 跳过 target 模型复制；参与保留节点保留各自更新后的本地模型\n")
        
        # 2. 恢复阶段
        self.recovery_phase(
            batch_size=batch_size,
            grad_accum_steps=grad_accum_steps,
            show_progress=show_progress,
            local_steps=recovery_local_steps,
            seed=self.selection_seed,
        )
        
        if skip_final_eval:
            print("\n=== 最终评估已跳过 ===")
            print("诊断模式：仅保存 DFU 后模型状态，外部审计脚本负责计算后门 ASR。")
        else:
            # 最终评估
            print("\n=== 最终评估 ===")

            # 对于非 TOFU 数据集，评估 MIA/Unlearning 指标和 test set 性能
            if self.task_type != "tofu":
                final_unlearn_mean, final_unlearn_std = self._evaluate_unlearning_all(verbose=True)

                # 评估 test set 分类性能
                test_mean, test_std = self.evaluate_test_set(verbose=True)

                best_agent_id = None
                best_accuracy = 0.0
                best_macro_f1 = 0.0
                if self.last_test_per_agent_results:
                    fs = build_final_stats(
                        self.last_test_per_agent_results,
                        primary_metric="macro_f1",
                        eval_agent_ids=sorted(self._get_eval_agent_ids()),
                    )
                    best_agent_id = fs.get("best_agent_id")
                    best_accuracy = float(fs.get("accuracy_best", 0.0))
                    best_macro_f1 = float(fs.get("macro_f1_best", 0.0))

                self.history["unlearning_metrics"].append({
                    "phase": "final", "step": self.unlearn_rounds + self.recovery_rounds,
                    **final_unlearn_mean,
                    **{f"{k}_std": v for k, v in final_unlearn_std.items()},
                    # 添加 test set 指标
                    "test_accuracy": test_mean.get("accuracy", 0),
                    "test_macro_f1": test_mean.get("macro_f1", 0),
                    "test_precision": test_mean.get("precision", 0),
                    "test_recall": test_mean.get("recall", 0),
                    "test_accuracy_std": test_std.get("accuracy", 0),
                    "test_macro_f1_std": test_std.get("macro_f1", 0),
                    "test_best_agent_id": best_agent_id,
                    "test_accuracy_best": best_accuracy,
                    "test_macro_f1_best": best_macro_f1,
                })
            
            # 对于 TOFU 数据集，只评估 TOFU 特定指标
            if self.task_type == "tofu" and hasattr(self, 'tofu_evaluator'):
                final_tofu = self.evaluate_tofu(verbose=True)
                final_tofu["phase"] = "final"
                final_tofu["step"] = self.unlearn_rounds + self.recovery_rounds
                self.history["tofu_metrics"].append(final_tofu)
        
        # 保存结果
        if save_dir:
            self._save_history(save_path)
            if self.save_lora_states:
                try:
                    self._save_final(save_path)
                except Exception as e:  # noqa: BLE001
                    print(f"[WARN] Failed to save final LoRA states: {e}")
            self._plot_curves(curves_dir)
        
        print(f"\n{'='*60}")
        print("D-FedOSD completed!")
        print(f"{'='*60}")

    def _save_final(self, save_path: Path):
        """保存最终模型。"""
        final_dir = save_path / "final"
        final_dir.mkdir(exist_ok=True)
        
        for agent_id in self.recovery_topology.surviving_agents:
            agent_dir = final_dir / f"agent_{agent_id}"
            agent_dir.mkdir(exist_ok=True)
            torch.save(self.current_models[agent_id], agent_dir / "lora_state.pt")

    def _save_history(self, save_path: Path):
        """保存训练历史。"""
        def safe_float(v):
            if isinstance(v, (int, float)):
                return None if np.isnan(v) else float(v)
            return v
        
        history_json = {
            "config": {
                "target_agent": self.target_agent,
                "unlearn_rounds": self.unlearn_rounds,
                "unlearn_lr": self.unlearn_lr,
                "recovery_rounds": self.recovery_rounds,
                "recovery_epochs": self.recovery_epochs,
                "recovery_lr": self.recovery_lr,
                "retain_grad_samples": getattr(self, "_last_retain_grad_samples", None),
                "forget_grad_samples": getattr(self, "_last_forget_grad_samples", None),
                "selection_strategy": self.selection_strategy,
                "selected_agents": self.recovery_topology.surviving_agents,
                "fedosd_unlearn_mode": "decentralized_local",
                "target_model_broadcast": False,
                "retain_gradient_reference_state": "updated_agent_model",
                "projected_recovery": self.projected_recovery,
                "retain_subspace_mode": self.retain_subspace_mode,
                "orthogonal_update_norm": self.orthogonal_update_norm,
                "recovery_gradient_mode": (
                    "retain_gradient_orthogonal_to_forget_gradient"
                    if self.projected_recovery
                    else "ordinary_retain_training"
                ),
                "recovery_forget_gradient_samples": (
                    getattr(self, "_last_forget_grad_samples", None)
                    if getattr(self, "_last_forget_grad_samples", None) is not None
                    else getattr(self, "_last_retain_grad_samples", None)
                ),
                "retain_grad_cache_mode": "disabled_for_decentralized_local",
                "dfl_snapshot": str(self.snapshot.snapshot_dir),
            },
            "unlearn_phase": {
                "rounds": self.history["unlearn_phase"]["rounds"],
                "orthogonal_norms": [safe_float(v) for v in self.history["unlearn_phase"]["orthogonal_norms"]],
                "orthogonal_norms_by_agent": self.history["unlearn_phase"].get("orthogonal_norms_by_agent", []),
            },
            "recovery_phase": {
                "rounds": self.history["recovery_phase"]["rounds"],
                "agent_losses": {
                    str(k): [safe_float(v) for v in vals]
                    for k, vals in self.history["recovery_phase"]["agent_losses"].items()
                },
            },
            "unlearning_metrics": [
                {k: safe_float(v) for k, v in m.items()}
                for m in self.history.get("unlearning_metrics", [])
            ],
            "tofu_metrics": [
                {k: safe_float(v) for k, v in m.items()}
                for m in self.history.get("tofu_metrics", [])
            ],
        }

        # Final stats (test-set metrics over eval_agent_ids, typically selected-k agents).
        if self.last_test_per_agent_results:
            per_agent = self.last_test_per_agent_results
            history_json["final_stats"] = build_final_stats(
                per_agent,
                primary_metric="macro_f1",
                eval_agent_ids=sorted(per_agent.keys()),
            )

            # Attach unlearning (MIA) summary if available.
            if self.history.get("unlearning_metrics"):
                last_unlearn = self.history["unlearning_metrics"][-1]
                for k in [
                    "mia_auc",
                    "mia_tpr_at_1fpr",
                    "mia_tpr_at_5fpr",
                    "forget_loss",
                    "retain_loss",
                    "loss_gap",
                    "forget_accuracy",
                    "retain_accuracy",
                    "accuracy_gap",
                ]:
                    if k in last_unlearn:
                        history_json["final_stats"][k] = safe_float(last_unlearn.get(k))
                    std_k = f"{k}_std"
                    if std_k in last_unlearn:
                        history_json["final_stats"][std_k] = safe_float(last_unlearn.get(std_k))
        
        with open(save_path / "history.json", "w") as f:
            json.dump(history_json, f, indent=2)

    def _plot_curves(self, curves_dir: Path):
        """绘制训练曲线。"""
        curves_dir = Path(curves_dir)
        curves_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. 正交方向范数曲线
        if self.history["unlearn_phase"]["rounds"]:
            fig, ax = plt.subplots(figsize=(10, 6))
            rounds = self.history["unlearn_phase"]["rounds"]
            ax.plot(rounds, self.history["unlearn_phase"]["orthogonal_norms"], 'b-o')
            ax.set_xlabel("Unlearn Round")
            ax.set_ylabel("Orthogonal Direction Norm")
            ax.set_title("D-FedOSD: Orthogonal Direction Norm")
            plt.tight_layout()
            plt.savefig(curves_dir / "orthogonal_norms.png", dpi=150)
            plt.close()
        
        # 2. 恢复阶段损失曲线
        if self.history["recovery_phase"]["rounds"]:
            fig, ax = plt.subplots(figsize=(10, 6))
            rounds = self.history["recovery_phase"]["rounds"]
            
            for agent_id, losses in self.history["recovery_phase"]["agent_losses"].items():
                if losses:
                    ax.plot(rounds[:len(losses)], losses, alpha=0.5, label=f"Agent {agent_id}")
            
            all_losses = list(self.history["recovery_phase"]["agent_losses"].values())
            if all_losses and all_losses[0]:
                avg_losses = [np.mean([l[i] for l in all_losses if i < len(l)]) 
                             for i in range(len(all_losses[0]))]
                ax.plot(rounds[:len(avg_losses)], avg_losses, 'k-', linewidth=2, label="Average")
            
            ax.set_xlabel("Recovery Round")
            ax.set_ylabel("Loss")
            ax.set_title("D-FedOSD: Recovery Phase Loss")
            ax.legend(fontsize='small')
            plt.tight_layout()
            plt.savefig(curves_dir / "recovery_phase_loss.png", dpi=150)
            plt.close()
        
        # 3. MIA 曲线
        if self.history.get("unlearning_metrics"):
            fig, ax = plt.subplots(figsize=(10, 6))
            metrics = self.history["unlearning_metrics"]
            phases = [m.get("phase", f"step_{m.get('step', i)}") for i, m in enumerate(metrics)]
            mia_auc = [m.get("mia_auc", 0) for m in metrics]
            
            ax.bar(range(len(phases)), mia_auc, color=['blue', 'orange', 'green'][:len(phases)])
            ax.set_xticks(range(len(phases)))
            ax.set_xticklabels(phases)
            ax.set_ylabel("MIA AUC")
            ax.set_title("D-FedOSD: MIA AUC by Phase")
            ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
            ax.set_ylim(0, 1)
            plt.tight_layout()
            plt.savefig(curves_dir / "mia_curves.png", dpi=150)
            plt.close()
