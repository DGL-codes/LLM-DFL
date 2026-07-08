"""D-Oblivionis: 基于梯度上升的去中心化联邦遗忘方法。

流程：
1. 遗忘阶段：
   - 目标客户端已退出，不参与 DFU 更新
   - 每个被选中的保留客户端从自己的 DFL 最终模型出发
   - 使用目标客户端的 forget set 执行本地 GradAscent
   - 不复制 target agent 模型，也不做全局聚合
2. 传播阶段：移除目标客户端，幸存/选中客户端在新环状拓扑上继续联邦训练
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
from .losses import GradAscentLoss
from .trainer import DFURingTopology, DFUAgent
from ..models.lora_model import LoRAModelWrapper
from ..models.trainer import LLMTrainer
from ..models.multi_agent_eval import build_final_stats, evaluate_lora_states
from ..data.base import Sample
from ..data.collator import LLMCollator
from ..data.partitioner import PartitionInfo
from ..dfl.agent import RingTopology
from ..utils.determinism import derive_seed


class DOblivionisAgent(DFUAgent):
    """D-Oblivionis Agent，支持 GradAscent 训练和正常训练两种模式。"""

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
        forget_samples: Optional[List[Sample]] = None
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
        self.grad_ascent_loss = GradAscentLoss()

    def grad_ascent_train(
        self,
        init_state: Dict[str, torch.Tensor],
        epochs: Optional[int] = None,
        local_steps: Optional[int] = None,
        batch_size: int = 4,
        grad_accum_steps: int = 2,
        show_progress: bool = True,
        progress_desc: Optional[str] = None,
        progress_position: Optional[int] = None,
        seed: Optional[int] = None
    ) -> Tuple[Dict[str, torch.Tensor], float]:
        """执行 GradAscent 训练（用于本地去除目标客户端的影响）。
        
        Args:
            init_state: 初始 LoRA 状态
            epochs: 训练轮数（遍历所有数据，兼容旧逻辑）
            local_steps: 训练步数（与DFL一致，优先于epochs）
            batch_size: 批次大小
            grad_accum_steps: 梯度累积步数
            
        Returns:
            Tuple of (trained_state, avg_loss)
        """
        # 加载初始状态
        self.model.set_lora_state_dict(init_state)
        
        # 冻结未选中的参数
        self._freeze_unselected_params()

        avg_loss = self.trainer.train_local(
            self.forget_samples,
            self.collator,
            local_steps=local_steps,
            epochs=epochs,
            loss_fn=self.grad_ascent_loss,
            batch_size=batch_size,
            grad_accum_steps=grad_accum_steps,
            show_progress=show_progress,
            progress_desc=progress_desc,
            progress_position=progress_position,
            progress_leave=True,
            seed=seed,
            reset_optimizer=True,
        )
        
        # 恢复所有参数可训练
        self._unfreeze_all_params()
        
        # 获取训练后的状态
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
        """执行正常训练（用于非目标客户端保留）。
        
        Args:
            init_state: 初始 LoRA 状态
            epochs: 训练轮数（遍历所有数据）
            local_steps: 训练步数（与DFL一致，优先于epochs）
            batch_size: 批次大小
            grad_accum_steps: 梯度累积步数
            
        Returns:
            Tuple of (trained_state, avg_loss)
            
        注意：如果同时指定epochs和local_steps，优先使用local_steps
        """
        # 加载初始状态
        self.model.set_lora_state_dict(init_state)
        
        # 冻结未选中的参数
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
        
        # 恢复所有参数可训练
        self._unfreeze_all_params()
        
        # 获取训练后的状态
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


class DOblivionisTrainer:
    """D-Oblivionis: 基于梯度上升的去中心化联邦遗忘方法。
    
    流程：
    1. 遗忘阶段：
       - 目标客户端已退出
       - 每个参与保留客户端从自己的 DFL 最终模型出发
       - 使用目标客户端的 forget set 执行本地 GradAscent
       - 不复制 target agent 模型
    2. 传播阶段：移除目标客户端，幸存/选中客户端在新环状拓扑上继续联邦训练
    """

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
        # D-Oblivionis 特有参数
        unlearn_rounds: int = 3,
        unlearn_epochs: int = 1,
        unlearn_lr: float = 1e-4,
        propagation_rounds: int = 5,
        propagation_epochs: int = 1,
        propagation_lr: float = 1e-4,
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
    ):
        """初始化 D-Oblivionis Trainer。"""
        self.snapshot_loader = snapshot_loader
        self.snapshot = snapshot_loader.snapshot
        self.model = model
        self.collator = collator
        self.all_samples = all_samples
        self.val_samples = val_samples
        self.test_samples = test_samples
        self.partition = partition
        self.target_agent = target_agent
        
        # D-Oblivionis 特有参数
        self.unlearn_rounds = unlearn_rounds
        self.unlearn_epochs = unlearn_epochs
        self.unlearn_lr = unlearn_lr
        self.propagation_rounds = propagation_rounds
        self.propagation_epochs = propagation_epochs
        self.propagation_lr = propagation_lr
        
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
        
        # 执行LoRA参数选择（如果启用）
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
            removed_agent=-1,  # 不移除任何agent
            selected_agents=list(range(self.snapshot.num_agents))  # 所有agent参与
        )
        
        # 初始化传播阶段拓扑（移除目标客户端）
        self.propagation_topology = DFURingTopology(
            self.snapshot.num_agents,
            removed_agent=target_agent,
            selected_agents=selected_agents,
            aggregation_weights=self.selection_result.weights,
            aggregation_scope=self.tdb_aggregation_scope,
        )
        
        # 初始化所有 agents（包括目标客户端）
        self.agents: Dict[int, DOblivionisAgent] = {}
        for agent_id in range(self.snapshot.num_agents):
            indices = partition.agent_indices[agent_id]
            local_samples = [all_samples[idx] for idx in indices]
            
            is_target = (agent_id == target_agent)
            
            self.agents[agent_id] = DOblivionisAgent(
                agent_id=agent_id,
                local_samples=local_samples,
                model=model,
                collator=collator,
                lr=self.unlearn_lr if is_target else self.lr,
                device=device,
                selected_param_keys=self.selected_param_keys,
                optimizer_name=optimizer_name,
                is_target=is_target,
                forget_samples=self.forget_samples
            )
        
        # 当前模型状态
        self.current_models: Dict[int, Dict[str, torch.Tensor]] = {}
        
        # retain 样本（用于验证）
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
                "target_losses": [],
                "agent_losses": {i: [] for i in selected_agents},
            },
            "propagation_phase": {
                "rounds": [],
                "agent_losses": {i: [] for i in selected_agents},
                "avg_metrics": [],
                "avg_metrics_std": [],
            },
            "unlearning_metrics": [],
            "tofu_metrics": [],
        }

        # Cache per-agent test-set metrics for final_stats writing.
        self.last_test_per_agent_results: Optional[Dict[int, Dict[str, float]]] = None

        # Optional: official TOFU eval samples (retain_perturbed / forgetXX_perturbed).
        # If provided, evaluate_tofu() uses these instead of client training samples.
        self.tofu_external_retain_samples: Optional[List[Sample]] = None
        self.tofu_external_forget_samples: Optional[List[Sample]] = None
        
        # TOFU 评估器
        if self.task_type == "tofu":
            self.init_tofu_evaluator()
        
        # 评估用的 agent IDs
        self.eval_agent_ids: Optional[List[int]] = None

    def _perform_param_selection(self):
        """执行 LoRA 参数选择。"""
        print("\n--- LoRA Parameter Selection ---")
        
        if self.param_random_selection:
            print("Using RANDOM LoRA module selection...")
            sensitivities = compute_module_sensitivities(
                self.snapshot_loader,
                target_agent=self.target_agent,
                verbose=False
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
                    print(f"Computing module sensitivities for target agent {self.target_agent}...")
                    sensitivities = compute_module_sensitivities(
                        self.snapshot_loader,
                        target_agent=self.target_agent,
                        verbose=True
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
            sample_state,
            self.param_selection_result.selected_modules
        )
        
        print(f"Selected {len(self.param_selection_result.selected_modules)} / "
              f"{len(self.param_selection_result.all_modules)} modules")

    def initialize_models(self):
        """从 DFL 快照初始化所有客户端模型。"""
        print(f"\n[初始化] 从DFL快照加载初始模型")
        print(f"  快照路径: {self.snapshot.snapshot_dir}")
        print(f"  目标遗忘agent: {self.target_agent}")
        
        # 使用最终轮次的模型作为起点
        final_round = self.snapshot.available_rounds[-1]
        
        for agent_id in range(self.snapshot.num_agents):
            agent_state = self.snapshot_loader.load_agent_state(final_round, agent_id)
            self.current_models[agent_id] = {
                k: v.clone() for k, v in agent_state.items()
            }
            print(f"  Agent {agent_id}: 加载 round_{final_round}/agent_{agent_id}/lora_state.pt")
        
        print(f"[初始化完成] 共{len(self.current_models)}个agent\n")

    def _ring_aggregate(
        self,
        agent_models: Dict[int, Dict[str, torch.Tensor]],
        topology: DFURingTopology
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """环状聚合：每个客户端与邻居平均。
        
        Args:
            agent_models: 各客户端的模型状态
            topology: 环状拓扑
            
        Returns:
            聚合后的模型状态
        """
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

    def unlearn_phase(
        self,
        batch_size: int = 4,
        grad_accum_steps: int = 2,
        verbose: bool = True,
        show_progress: bool = True,
        local_steps: Optional[int] = None
    ):
        """遗忘阶段：每个参与保留客户端在自己的本地模型上执行 GradAscent。
        
        流程：
        1. 参与客户端从各自 DFL 最终模型出发
        2. 使用目标 agent 的 forget set 做 GradAscent
        3. 只更新各自本地模型，不复制 target agent 模型
        4. 后续传播阶段仍使用原有环形邻居聚合
        """
        print(f"\n{'='*60}")
        print("D-Oblivionis: 遗忘阶段")
        print(f"{'='*60}")
        print(f"  遗忘轮数: {self.unlearn_rounds}")
        if local_steps is not None and int(local_steps) > 0:
            print(f"  每轮本地训练步数: {int(local_steps)}")
        else:
            print(f"  每轮训练轮数: {self.unlearn_epochs}")
        print(f"  目标客户端: {self.target_agent}")
        print(f"  去中心化参与客户端: {self.propagation_topology.surviving_agents}")
        
        participant_agents = list(self.propagation_topology.surviving_agents)
        if not participant_agents:
            raise ValueError("D-Oblivionis decentralized-local unlearn phase has no participant agents")

        # 1. 参与保留客户端执行多轮 GradAscent 训练
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
            round_losses: Dict[int, float] = {}
            participant_pbar = tqdm(
                participant_agents,
                desc=f"Local GradAscent r{round_idx + 1}",
                dynamic_ncols=True,
                leave=True,
                disable=not show_progress,
            )
            for agent_id in participant_pbar:
                agent = self.agents[agent_id]
                agent.lr = self.unlearn_lr
                agent.trainer.lr = self.unlearn_lr
                train_seed = derive_seed(
                    self.selection_seed,
                    salt="oblivionis_unlearn_local",
                    round_idx=round_idx,
                    agent_id=agent_id,
                )
                trained_state, loss = agent.grad_ascent_train(
                    init_state=self.current_models[agent_id],
                    epochs=self.unlearn_epochs if (local_steps is None or int(local_steps) <= 0) else None,
                    local_steps=int(local_steps) if (local_steps is not None and int(local_steps) > 0) else None,
                    batch_size=batch_size,
                    grad_accum_steps=grad_accum_steps,
                    show_progress=False,
                    progress_desc=f"Unlearn train A{agent_id}",
                    progress_position=1 if show_progress else None,
                    seed=train_seed
                )
                updated_models[int(agent_id)] = trained_state
                round_losses[int(agent_id)] = float(loss)

            for agent_id, trained_state in updated_models.items():
                self.current_models[agent_id] = trained_state

            avg_loss = float(np.mean(list(round_losses.values()))) if round_losses else 0.0
            if verbose:
                print(f"  参与客户端平均 GradAscent loss = {avg_loss:.4f}")
            if round_pbar is not None:
                round_pbar.set_postfix({"avg_loss": f"{avg_loss:.4f}"})
            
            # 记录历史
            self.history["unlearn_phase"]["rounds"].append(round_idx + 1)
            self.history["unlearn_phase"]["target_losses"].append(avg_loss)
            agent_loss_history = self.history["unlearn_phase"].setdefault("agent_losses", {})
            for agent_id, loss in round_losses.items():
                agent_loss_history.setdefault(agent_id, []).append(loss)
        
        print(f"\n遗忘阶段完成：未复制 target agent 模型，保留节点只更新自己的本地模型")
        if round_pbar is not None:
            round_pbar.close()

    def propagation_phase(
        self,
        batch_size: int = 4,
        grad_accum_steps: int = 2,
        verbose: bool = True,
        show_progress: bool = True,
        local_steps: int = None
    ):
        """传播阶段：移除目标客户端，幸存客户端继续联邦训练。
        
        每轮：
        1. 幸存客户端用 retain 数据正常训练
        2. 环状聚合（不包含目标客户端）
        
        重复 propagation_rounds 轮。
        
        Args:
            local_steps: 每轮每个agent的训练步数（与DFL一致）
                        如果为None，则使用propagation_epochs遍历所有数据
        """
        import gc
        
        print(f"\n{'='*60}")
        print("D-Oblivionis: 传播阶段")
        print(f"{'='*60}")
        print(f"  传播轮数: {self.propagation_rounds}")
        effective_local_steps = local_steps if (local_steps is not None and int(local_steps) > 0) else None
        if effective_local_steps is not None:
            print(f"  每轮本地训练步数: {effective_local_steps}")
        else:
            print(f"  每轮训练轮数: {self.propagation_epochs}")
        print(f"  参与客户端: {self.propagation_topology.surviving_agents}")
        
        # 更新 agents 的学习率为传播阶段学习率
        for agent_id in self.propagation_topology.surviving_agents:
            self.agents[agent_id].lr = self.propagation_lr
            self.agents[agent_id].trainer.lr = self.propagation_lr
        
        round_pbar = None
        if show_progress and self.propagation_rounds > 0:
            round_pbar = tqdm(
                range(self.propagation_rounds),
                desc="Propagation rounds",
                dynamic_ncols=True,
                leave=True
            )
            round_iter = round_pbar
        else:
            round_iter = range(self.propagation_rounds)

        for round_idx in round_iter:
            if verbose:
                print(f"\n--- 传播轮次 {round_idx + 1}/{self.propagation_rounds} ---")
            
            trained_models: Dict[int, Dict[str, torch.Tensor]] = {}
            round_losses = []
            
            # 幸存客户端正常训练
            for agent_id in self.propagation_topology.surviving_agents:
                agent = self.agents[agent_id]
                train_seed = derive_seed(
                    self.selection_seed,
                    salt="oblivionis_propagation",
                    round_idx=round_idx,
                    agent_id=agent_id,
                )
                trained_state, loss = agent.retain_train(
                    init_state=self.current_models[agent_id],
                    epochs=self.propagation_epochs if effective_local_steps is None else None,
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
                self.history["propagation_phase"]["agent_losses"][agent_id].append(loss)
            
            avg_loss = np.mean(round_losses) if round_losses else 0.0
            if verbose:
                print(f"  平均 loss = {avg_loss:.4f}")
            if round_pbar is not None:
                round_pbar.set_postfix({"loss": f"{avg_loss:.4f}"})
            
            # 环状聚合（仅幸存客户端）
            aggregated = self._ring_aggregate(trained_models, self.propagation_topology)
            for agent_id in self.propagation_topology.surviving_agents:
                self.current_models[agent_id] = aggregated[agent_id]
            
            self.history["propagation_phase"]["rounds"].append(round_idx + 1)
            
            # 清理本轮的临时数据
            del trained_models, aggregated
        
        print(f"\n传播阶段完成")
        if round_pbar is not None:
            round_pbar.close()

    def _get_eval_agent_ids(self) -> List[int]:
        """获取用于评估的 agent 列表。"""
        if self.eval_agent_ids is not None:
            return [aid for aid in self.eval_agent_ids 
                    if aid in self.propagation_topology.surviving_agents]
        # Default: evaluate on ALL actually selected agents (selected-k).
        return sorted(self.propagation_topology.surviving_agents)

    def set_eval_agent_ids(self, agent_ids: List[int]) -> None:
        """设置评估时使用的 agent 模型 ID 列表。"""
        self.eval_agent_ids = list(agent_ids)

    def verify_unlearning(self, agent_id: int = None, verbose: bool = True) -> UnlearningMetrics:
        """验证遗忘效果。"""
        if agent_id is None:
            agent_id = self.propagation_topology.surviving_agents[0]
        
        self.model.set_lora_state_dict(self.current_models[agent_id])
        verifier = UnlearningVerifier(self.model, self.collator, self.device)
        
        if verbose:
            print(f"\n--- Unlearning Verification (Agent {agent_id}) ---")
        
        # 使用固定的评估子集（已在初始化时创建）
        metrics = verifier.verify_unlearning(
            self.forget_samples_eval,
            self.retain_samples_eval,
            nonmember_samples=self.nonmember_samples_eval,
            batch_size=8,
            verbose=verbose,
            show_progress=verbose
        )
        return metrics

    def _evaluate_unlearning_all(self, verbose: bool = True) -> Tuple[Dict[str, float], Dict[str, float]]:
        """对所有评估 agent 运行遗忘验证并聚合结果。"""
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
            print(f"  Forget Loss: {mean_metrics.get('forget_loss', 0):.4f}±{std_metrics.get('forget_loss', 0):.4f}")
            print(f"  Retain Loss: {mean_metrics.get('retain_loss', 0):.4f}±{std_metrics.get('retain_loss', 0):.4f}")
        
        return mean_metrics, std_metrics

    def _aggregate_unlearning_metrics(
        self,
        per_agent_metrics: Dict[int, UnlearningMetrics]
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """聚合遗忘指标。"""
        if not per_agent_metrics:
            return {}, {}
        
        metric_keys = [
            "mia_auc", "mia_tpr_at_1fpr", "mia_tpr_at_5fpr",
            "forget_loss", "retain_loss", "loss_gap",
            "forget_accuracy", "retain_accuracy", "accuracy_gap"
        ]
        
        mean_metrics: Dict[str, float] = {}
        std_metrics: Dict[str, float] = {}
        
        for key in metric_keys:
            values = [
                getattr(m, key) for m in per_agent_metrics.values()
                if getattr(m, key) is not None and not np.isnan(getattr(m, key))
            ]
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
            model=self.model,
            tokenizer=self.model.tokenizer,
            collator=self.collator,
            device=self.device,
            tofu_local_dir=self.tofu_local_dir,
        )

    def evaluate_tofu(self, agent_id: int = None, verbose: bool = True) -> Dict:
        """使用 TOFU 指标评估。"""
        if not hasattr(self, 'tofu_evaluator'):
            return {}
        
        if agent_id is None:
            agent_ids = self._get_eval_agent_ids()
        else:
            agent_ids = [agent_id]
        
        per_agent_results: Dict[int, Dict] = {}

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
        aggregated: Dict[str, float] = {}
        
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
        propagation_local_steps: int = None,
        unlearn_local_steps: int = None,
        skip_final_eval: bool = False
    ):
        """运行完整的 D-Oblivionis 遗忘流程。
        
        Args:
            batch_size: 批次大小
            grad_accum_steps: 梯度累积步数
            save_dir: 保存目录
            eval_every: 每隔多少轮评估一次
            propagation_local_steps: 传播阶段每轮每个agent的训练步数（与DFL一致）
                                    如果为None，则使用propagation_epochs遍历所有数据
        """
        print(f"\n{'='*60}")
        print("Starting D-Oblivionis (Decentralized Federated Unlearning)")
        print(f"{'='*60}")
        print(f"Target agent to forget: {self.target_agent}")
        print(f"Unlearn rounds: {self.unlearn_rounds}")
        print(f"Propagation rounds: {self.propagation_rounds}")
        if unlearn_local_steps is not None and int(unlearn_local_steps) > 0:
            print(f"Unlearn local steps: {int(unlearn_local_steps)}")
        else:
            print(f"Unlearn epochs: {self.unlearn_epochs}")
        if propagation_local_steps is not None and int(propagation_local_steps) > 0:
            print(f"Propagation local steps: {int(propagation_local_steps)}")
        else:
            print(f"Propagation epochs: {self.propagation_epochs}")
        
        # 显示节点选择信息
        print(f"\n--- Agent Selection ---")
        print(f"Selection strategy: {self.selection_strategy}")
        print(f"Selected agents for propagation: {self.propagation_topology.surviving_agents}")
        
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            curves_dir = save_path / "curves"
            curves_dir.mkdir(exist_ok=True)

        # 记录本次run使用的关键超参（用于 history.json 复现）
        self._run_unlearn_local_steps = unlearn_local_steps
        self._run_propagation_local_steps = propagation_local_steps
        
        # 初始化模型
        self.initialize_models()
        
        # 1. 遗忘阶段
        self.unlearn_phase(
            batch_size=batch_size,
            grad_accum_steps=grad_accum_steps,
            show_progress=show_progress,
            local_steps=unlearn_local_steps
        )
        
        # 2. 传播阶段
        self.propagation_phase(
            batch_size=batch_size,
            grad_accum_steps=grad_accum_steps,
            show_progress=show_progress,
            local_steps=propagation_local_steps
        )
        
        # 最终评估
        if skip_final_eval:
            print("\n跳过最终评估（skip_final_eval=True）；后门审计会单独评估保存后的模型。")
        else:
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
                    "phase": "final",
                    "step": self.unlearn_rounds + self.propagation_rounds,
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
                final_tofu["step"] = self.unlearn_rounds + self.propagation_rounds
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
        print("D-Oblivionis completed!")
        print(f"{'='*60}")

    def _save_final(self, save_path: Path):
        """保存最终模型。"""
        final_dir = save_path / "final"
        final_dir.mkdir(exist_ok=True)
        
        for agent_id in self.propagation_topology.surviving_agents:
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
                "seed": self.selection_seed,
                "unlearn_rounds": self.unlearn_rounds,
                "unlearn_epochs": self.unlearn_epochs,
                "unlearn_local_steps": getattr(self, "_run_unlearn_local_steps", None),
                "unlearn_lr": self.unlearn_lr,
                "propagation_rounds": self.propagation_rounds,
                "propagation_epochs": self.propagation_epochs,
                "propagation_local_steps": getattr(self, "_run_propagation_local_steps", None),
                "propagation_lr": self.propagation_lr,
                "selection_strategy": self.selection_strategy,
                "selected_agents": self.propagation_topology.surviving_agents,
                "oblivionis_unlearn_mode": "decentralized_local",
                "target_model_broadcast": False,
                "dfl_snapshot": str(self.snapshot.snapshot_dir),
            },
            "unlearn_phase": {
                "rounds": self.history["unlearn_phase"]["rounds"],
                "target_losses": [safe_float(v) for v in self.history["unlearn_phase"]["target_losses"]],
                "agent_losses": {
                    str(k): [safe_float(v) for v in vals]
                    for k, vals in self.history["unlearn_phase"].get("agent_losses", {}).items()
                },
            },
            "propagation_phase": {
                "rounds": self.history["propagation_phase"]["rounds"],
                "agent_losses": {
                    str(k): [safe_float(v) for v in vals]
                    for k, vals in self.history["propagation_phase"]["agent_losses"].items()
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
        
        # 1. 遗忘阶段损失曲线
        if self.history["unlearn_phase"]["rounds"]:
            fig, ax = plt.subplots(figsize=(10, 6))
            rounds = self.history["unlearn_phase"]["rounds"]
            ax.plot(rounds, self.history["unlearn_phase"]["target_losses"],
                   'r-o', label="Participant avg (GradAscent)")
            ax.set_xlabel("Unlearn Round")
            ax.set_ylabel("Loss")
            ax.set_title("D-Oblivionis: Unlearn Phase Loss")
            ax.legend()
            plt.tight_layout()
            plt.savefig(curves_dir / "unlearn_phase_loss.png", dpi=150)
            plt.close()
        
        # 2. 传播阶段损失曲线
        if self.history["propagation_phase"]["rounds"]:
            fig, ax = plt.subplots(figsize=(10, 6))
            rounds = self.history["propagation_phase"]["rounds"]
            
            for agent_id, losses in self.history["propagation_phase"]["agent_losses"].items():
                if losses:
                    ax.plot(rounds[:len(losses)], losses, alpha=0.5, label=f"Agent {agent_id}")
            
            # 平均损失
            all_losses = list(self.history["propagation_phase"]["agent_losses"].values())
            if all_losses and all_losses[0]:
                avg_losses = [np.mean([l[i] for l in all_losses if i < len(l)]) 
                             for i in range(len(all_losses[0]))]
                ax.plot(rounds[:len(avg_losses)], avg_losses, 'k-', linewidth=2, label="Average")
            
            ax.set_xlabel("Propagation Round")
            ax.set_ylabel("Loss")
            ax.set_title("D-Oblivionis: Propagation Phase Loss")
            ax.legend(fontsize='small')
            plt.tight_layout()
            plt.savefig(curves_dir / "propagation_phase_loss.png", dpi=150)
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
            ax.set_title("D-Oblivionis: MIA AUC by Phase")
            ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label="Random (0.5)")
            ax.set_ylim(0, 1)
            plt.tight_layout()
            plt.savefig(curves_dir / "mia_curves.png", dpi=150)
            plt.close()
