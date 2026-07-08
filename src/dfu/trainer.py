"""DFU Trainer implementing decentralized federated unlearning.

索引对应关系说明（更新后）：
=========================
DFL代码中的保存逻辑（src/dfl/trainer.py）已更新：
- 训练开始前保存 round_0（初始未训练LoRA）
- 第i轮训练后保存 round_i（i从1开始）
- 对于G个全局轮次，共保存 round_0 到 round_G，共G+1个快照

时间步映射：
- round_0 = m_k_0_old = 初始未训练LoRA
- round_i (i≥1) = m_k_i_old = 第i轮训练后的模型
- 例如: G=10 时，round_0 到 round_10，共11个快照

DFU校准调度（以G=10, interval=2为例）：
- Step 0: round_0 → round_2 (m_k_0 → m_k_2)
- Step 1: round_2 → round_4 (m_k_2 → m_k_4)
- Step 2: round_4 → round_6 (m_k_4 → m_k_6)
- Step 3: round_6 → round_8 (m_k_6 → m_k_8)
- Step 4: round_8 → round_10 (m_k_8 → m_k_10)
- 共5个校准步骤

不再需要特殊处理 round_from=-1 的情况。
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
from ..models.lora_model import LoRAModelWrapper
from ..models.trainer import LLMTrainer
from ..models.multi_agent_eval import build_final_stats, evaluate_lora_states, summarize_per_agent_metrics
from ..data.base import Sample
from ..data.collator import LLMCollator
from ..data.partitioner import PartitionInfo
from ..dfl.agent import RingTopology


class DFURingTopology(RingTopology):
    """Modified ring topology for DFU with selected agents.

    支持两种使用方式:
    1. 移除target agent后使用所有幸存agent (full策略)
    2. 指定selected_agents列表，只使用这些agent构建环

    例如: selected_agents=[1, 2, 4, 7, 9]
    构建的环: 9-1-2-4-7-9-1 (按照列表顺序)
    """

    def __init__(
        self,
        num_agents: int,
        removed_agent: int = 0,
        removed_agents: Optional[List[int]] = None,
        selected_agents: Optional[List[int]] = None,
        aggregation_weights: Optional[Dict[int, float]] = None,
        aggregation_scope: str = "local",
    ):
        """Initialize ring topology for DFU.

        Args:
            num_agents: Original number of agents (before removal)
            removed_agent: Agent ID to remove (target agent)
            removed_agents: Optional list of already/currently removed agents.
                           If omitted, uses [removed_agent].
            selected_agents: Optional list of agent IDs to use.
                           If None, use all surviving agents.
                           If provided, only these agents form the ring.
            aggregation_weights: Optional global aggregation weights for selected
                           agents (used by TDB-AS). Neighbor-level weights are
                           renormalized over each local ring neighborhood.
            aggregation_scope: "local" uses ring neighborhoods; "global" uses
                           the temporary selected overlay for weighted averaging.
        """
        self.original_num_agents = num_agents
        self.removed_agent = removed_agent
        removed = sorted({int(a) for a in (removed_agents if removed_agents is not None else [removed_agent])})
        if removed_agent not in removed:
            removed.append(int(removed_agent))
            removed = sorted(set(removed))
        self.removed_agents = removed
        self.aggregation_scope = str(aggregation_scope or "local").lower().strip()
        if self.aggregation_scope not in {"local", "global"}:
            raise ValueError(f"Unsupported aggregation_scope: {aggregation_scope}")

        # 确定参与DFU的agent
        all_surviving = [i for i in range(num_agents) if i not in set(self.removed_agents)]

        if selected_agents is not None:
            # 验证selected_agents都是幸存agent
            for agent_id in selected_agents:
                if agent_id not in all_surviving:
                    raise ValueError(
                        f"Agent {agent_id} is not a surviving agent. "
                        f"Surviving agents: {all_surviving}"
                    )
            self.surviving_agents = sorted(selected_agents)
        else:
            self.surviving_agents = all_surviving

        self.num_agents = len(self.surviving_agents)
        self.all_surviving_agents = all_surviving  # 保存所有幸存agent供参考
        self.aggregation_weights = self._normalize_aggregation_weights(aggregation_weights)

        # Build new adjacency: agent -> [left_neighbor, right_neighbor]
        self._build_new_topology()

    def _normalize_aggregation_weights(
        self,
        aggregation_weights: Optional[Dict[int, float]]
    ) -> Optional[Dict[int, float]]:
        if not aggregation_weights:
            return None

        weights: Dict[int, float] = {}
        for agent_id in self.surviving_agents:
            value = float(aggregation_weights.get(agent_id, aggregation_weights.get(str(agent_id), 0.0)))
            weights[agent_id] = max(0.0, value)

        total = sum(weights.values())
        if total <= 0:
            return None
        return {agent_id: value / total for agent_id, value in weights.items()}

    def _build_new_topology(self):
        """Build ring topology for selected agents.

        按照 surviving_agents 列表的顺序构建环。
        例如: [1, 2, 4, 7, 9] -> 9-1-2-4-7-9
        """
        self.neighbors = {}
        n = len(self.surviving_agents)

        for i, agent_id in enumerate(self.surviving_agents):
            left_idx = (i - 1) % n
            right_idx = (i + 1) % n
            self.neighbors[agent_id] = [
                self.surviving_agents[left_idx],
                self.surviving_agents[right_idx]
            ]

    def get_neighbors(self, agent_id: int) -> List[int]:
        """Get neighbors for a surviving agent."""
        if agent_id not in self.neighbors:
            raise ValueError(f"Agent {agent_id} not in surviving agents")
        if self.aggregation_scope == "global":
            return [aid for aid in self.surviving_agents if aid != agent_id]
        # For small rings, left/right neighbors may be identical:
        # - n=1: left=right=self
        # - n=2: left=right=the other agent
        # For aggregation, we want each *unique* neighbor counted once.
        seen = set()
        uniq: List[int] = []
        for nid in self.neighbors[agent_id]:
            if nid == agent_id:
                continue
            if nid in seen:
                continue
            seen.add(nid)
            uniq.append(nid)
        return uniq

    def get_aggregation_weights(self, agent_id: int, all_ids: List[int]) -> List[float]:
        """Return normalized aggregation weights over a local neighborhood."""
        if self.aggregation_weights is None:
            return [1.0 / len(all_ids)] * len(all_ids)

        weights = [float(self.aggregation_weights.get(i, 0.0)) for i in all_ids]
        total = sum(weights)
        if total <= 0:
            return [1.0 / len(all_ids)] * len(all_ids)
        return [w / total for w in weights]

    def aggregate(
        self,
        agent_id: int,
        agent_deltas: Dict[int, Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """Aggregate delta_hat vectors with neighbors.

        Args:
            agent_id: Current agent ID
            agent_deltas: Dict mapping agent_id to delta_hat vectors

        Returns:
            Aggregated delta_hat
        """
        neighbors = self.get_neighbors(agent_id)
        all_ids = [agent_id] + neighbors  # Self + unique neighbors

        result = {}
        weights = self.get_aggregation_weights(agent_id, all_ids)
        for key in agent_deltas[agent_id].keys():
            stacked = torch.stack([agent_deltas[i][key] for i in all_ids])
            coeffs = torch.tensor(weights, dtype=stacked.dtype, device=stacked.device)
            result[key] = (stacked * coeffs.view(-1, *([1] * (stacked.dim() - 1)))).sum(dim=0)

        return result


class DFUAgent:
    """A DFU agent that performs calibration training."""

    def __init__(
        self,
        agent_id: int,
        local_samples: List[Sample],
        model: LoRAModelWrapper,
        collator: LLMCollator,
        lr: float = 1e-4,
        device: str = "cuda",
        selected_param_keys: Optional[set] = None,
        optimizer_name: str = "adamw"
    ):
        self.agent_id = agent_id
        self.local_samples = local_samples
        self.model = model
        self.collator = collator
        self.lr = lr
        self.device = device
        self.selected_param_keys = selected_param_keys  # LoRA参数选择: 选中的参数键集合

        # Create trainer for calibration
        self.trainer = LLMTrainer(
            model=model,
            lr=lr,
            device=device,
            optimizer_name=optimizer_name
        )

    def _freeze_unselected_params(self):
        """Freeze parameters not in selected_param_keys."""
        if self.selected_param_keys is None:
            return  # No selection, all params trainable

        for name, param in self.model.model.named_parameters():
            if "lora" in name.lower():
                if name not in self.selected_param_keys:
                    param.requires_grad = False
                else:
                    param.requires_grad = True

    def _unfreeze_all_params(self):
        """Unfreeze all LoRA parameters."""
        for name, param in self.model.model.named_parameters():
            if "lora" in name.lower():
                param.requires_grad = True

    def calibration_train(
        self,
        init_state: Dict[str, torch.Tensor],
        steps: int,
        batch_size: int = 4,
        grad_accum_steps: int = 2,
        show_progress: bool = True,
        progress_desc: Optional[str] = None,
        progress_position: Optional[int] = None,
        seed: Optional[int] = None
    ) -> Tuple[Dict[str, torch.Tensor], float]:
        """Perform calibration training from given initial state.

        Args:
            init_state: Initial LoRA state (m_k_t_new)
            steps: Number of training steps (E_cali)
            batch_size: Batch size
            grad_accum_steps: Gradient accumulation steps

        Returns:
            Tuple of (trained_state, training_loss)
        """
        # Load initial state
        self.model.set_lora_state_dict(init_state)

        # Freeze unselected parameters (LoRA参数选择)
        self._freeze_unselected_params()

        # Train (unified local training helper)
        loss = self.trainer.train_local(
            self.local_samples,
            self.collator,
            local_steps=steps,
            batch_size=batch_size,
            grad_accum_steps=grad_accum_steps,
            show_progress=show_progress,
            progress_desc=progress_desc,
            progress_position=progress_position,
            progress_leave=True,
            seed=seed,
            reset_optimizer=False,
        )

        # Restore all params to trainable (for next round)
        self._unfreeze_all_params()

        # Get trained state
        trained_state = self.model.get_lora_state_dict()
        
        # 简化清理：只清零梯度，不销毁优化器（避免频繁重建开销）
        if self.trainer.optimizer is not None:
            self.trainer.optimizer.zero_grad(set_to_none=True)
        
        return trained_state, loss


class DFUTrainer:
    """DFU (Decentralized Federated Unlearning) Trainer.

    Implements the calibration-based unlearning algorithm:
    1. Load historical DFL snapshots
    2. Remove target agent (agent 0) and optionally select subset of agents
    3. Build ring topology with selected agents
    4. For each time step, compute calibration direction and historical magnitude
    5. Aggregate updates with neighbors and update models

    节点选择策略：
    - full: 所有幸存agent参与DFU（默认）
    - random: 随机选择指定比例的agent
    - ours: 基于LP求解的贪心选择算法

    LoRA参数选择（可选）：
    - 启用后只对敏感度高的LoRA模块进行校准
    - 降低计算和通信开销
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
        removed_agents: Optional[List[int]] = None,
        calibration_steps: int = 3,
        calibration_interval: int = 1,
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
        # Retrain TR 分布（用于计算 Forget Quality）
        retrain_tr_path: Optional[str] = None,
        optimizer_name: str = "adamw",
        save_lora_states: bool = True,
    ):
        """Initialize DFU Trainer.

        Args:
            snapshot_loader: Loader for DFL snapshots (read-only)
            model: LoRA model wrapper
            collator: Data collator
            all_samples: All training samples
            val_samples: Validation samples
            test_samples: Test samples
            partition: Data partition info
            target_agent: Agent to forget (default: 0)
            removed_agents: Optional list of agents that should be excluded from
                the retained set for sequential-withdrawal probes.
            calibration_steps: Local calibration steps per interval (E_cali)
            calibration_interval: Interval between calibration steps
            lr: Learning rate for calibration
            label_names: Label names for classification
            task_type: Task type ("classification" or "generation")
            device: Device to use
            selection_strategy: Agent selection strategy ("full", "random", "ours")
            selection_ratio: Ratio of agents to select (random strategy)
            selection_count: Exact number of agents to select (random strategy)
            selection_seed: Random seed for agent selection
            selection_epsilon: Distribution error threshold for "ours" strategy
            enable_param_selection: Whether to enable LoRA parameter selection
            param_selection_ratio: Maximum ratio of modules to select
            param_epsilon_W: Energy threshold for parameter selection
            param_random_selection: Whether to use random LoRA module selection
            param_relative_sensitivity: Whether to use relative sensitivity weighting
            param_sensitivity_alpha: Exponent for relative sensitivity weighting
            param_selection_mode: LoRA selection mode ("epsilon" or "top_ratio")
            param_sensitivity_cache: Optional JSON path to cache/restore sensitivities
            retrain_tr_path: Path to retrain model's TR distribution (.npy file) for FQ calculation
        """
        self.snapshot_loader = snapshot_loader
        self.snapshot = snapshot_loader.snapshot
        self.model = model
        self.collator = collator
        self.all_samples = all_samples
        self.val_samples = val_samples
        self.test_samples = test_samples
        self.partition = partition
        self.target_agent = target_agent
        removed = sorted({int(a) for a in (removed_agents if removed_agents is not None else [target_agent])})
        if int(target_agent) not in removed:
            removed.append(int(target_agent))
            removed = sorted(set(removed))
        self.removed_agents = removed
        self.calibration_steps = calibration_steps
        self.calibration_interval = calibration_interval
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
        self.seed = selection_seed
        self.tdb_aggregation_scope = str(tdb_aggregation_scope or "local").lower().strip()
        # LoRA参数选择
        self.enable_param_selection = enable_param_selection
        self.param_selection_ratio = param_selection_ratio
        self.param_epsilon_W = param_epsilon_W
        self.param_random_selection = param_random_selection
        self.param_relative_sensitivity = param_relative_sensitivity
        self.param_sensitivity_alpha = param_sensitivity_alpha
        self.param_selection_mode = param_selection_mode
        self.param_sensitivity_cache = param_sensitivity_cache
        self.optimizer_name = optimizer_name
        self.save_lora_states = bool(save_lora_states)
        self.param_selection_result = None
        self.selected_param_keys = None
        
        # 加载 retrain TR 分布（用于计算 Forget Quality）
        self.retrain_tr = None
        if retrain_tr_path is not None:
            retrain_tr_path = Path(retrain_tr_path)
            if retrain_tr_path.exists():
                self.retrain_tr = np.load(retrain_tr_path)
                print(f"Loaded retrain TR distribution from {retrain_tr_path}")
                print(f"  Shape: {self.retrain_tr.shape}, Mean: {np.mean(self.retrain_tr):.4f}")
            else:
                print(f"Warning: retrain_tr_path not found: {retrain_tr_path}")

        # 执行LoRA参数选择（如果启用）
        if enable_param_selection:
            self._perform_param_selection()

        # 获取所有幸存agent（移除当前/历史 withdrawal agents 后）
        removed_set = set(self.removed_agents)
        all_surviving = [i for i in range(self.snapshot.num_agents) if i not in removed_set]

        # 执行节点选择
        num_classes = len(label_names) if label_names else 20  # fallback
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

        # Initialize modified topology with selected agents
        self.topology = DFURingTopology(
            self.snapshot.num_agents,
            removed_agent=target_agent,
            removed_agents=self.removed_agents,
            selected_agents=selected_agents,
            aggregation_weights=self.selection_result.weights,
            aggregation_scope=self.tdb_aggregation_scope,
        )

        # Initialize DFU agents for selected agents only
        self.agents: Dict[int, DFUAgent] = {}
        for agent_id in self.topology.surviving_agents:
            indices = partition.agent_indices[agent_id]
            local_samples = [all_samples[idx] for idx in indices]

            self.agents[agent_id] = DFUAgent(
                agent_id=agent_id,
                local_samples=local_samples,
                model=model,
                collator=collator,
                lr=lr,
                device=device,
                selected_param_keys=self.selected_param_keys,
                optimizer_name=optimizer_name
            )

        # Current calibrated models for surviving agents: m_k_t_new
        self.current_models: Dict[int, Dict[str, torch.Tensor]] = {}

        # Prepare forget/retain samples for unlearning verification
        forget_indices: List[int] = []
        for aid in self.removed_agents:
            forget_indices.extend(partition.agent_indices[aid])
        forget_indices = sorted(set(forget_indices))
        self.forget_samples = [all_samples[i] for i in forget_indices]

        # retain 使用所有非目标客户端的数据（完整保留集，与节点选择无关，保证不同实验可比）
        retain_indices: List[int] = []
        for aid in all_surviving:
            agent_indices = partition.agent_indices[aid]
            retain_indices.extend(agent_indices)
        retain_indices = sorted(retain_indices)
        self.retain_samples = [all_samples[i] for i in retain_indices]

        # Non-member pool for membership inference (MIA).
        # Prefer val/test splits (not used for training). This is separate from retain TRAIN samples.
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
            # Keep member/non-member sizes balanced for AUC stability.
            n_forget = len(self.forget_samples_eval)
            self.nonmember_samples_eval = self.nonmember_samples[: min(n_forget, len(self.nonmember_samples))]

        # Training history
        self.history = {
            "calibration_steps": [],
            "agent_losses": {i: [] for i in self.topology.surviving_agents},
            "agent_metrics": {i: [] for i in self.topology.surviving_agents},
            "avg_metrics": [],
            "avg_metrics_std": [],
            "best_metrics": [],
            "best_agent_ids": [],
            "calibration_norms": {i: [] for i in self.topology.surviving_agents},
            "historical_gammas": {i: [] for i in self.topology.surviving_agents},
            # Unlearning verification metrics
            "unlearning_metrics": []
        }
        
        # Initialize TOFU evaluator if task_type is 'tofu'
        if self.task_type == "tofu":
            self.init_tofu_evaluator()

        # Optional: official TOFU eval samples (retain_perturbed / forgetXX_perturbed).
        # If provided, evaluate_tofu() uses these instead of client training samples.
        self.tofu_external_retain_samples: Optional[List[Sample]] = None
        self.tofu_external_forget_samples: Optional[List[Sample]] = None

        # 默认评估仅使用参与客户端中的前3个（按ID排序）
        self.eval_agent_ids: Optional[List[int]] = None

    def _perform_param_selection(self):
        """Perform LoRA parameter selection based on module sensitivities."""
        print("\n--- LoRA Parameter Selection ---")
        
        if self.param_random_selection:
            print("Using RANDOM LoRA module selection...")
            # Random selection: get all modules and randomly select
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
            
            # Create result object
            from src.dfu.lora_param_selection import LoRASelectionResult
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
                # Default: epsilon-constrained greedy + optional ratio cap
                self.param_selection_result = select_lora_modules(
                    sensitivities=sensitivities,
                    epsilon_W=self.param_epsilon_W,
                    verbose=True,
                    max_ratio=self.param_selection_ratio
                )

        # Get state dict keys for selected modules
        sample_state = self.snapshot_loader.load_agent_state(
            self.snapshot.available_rounds[0], self.target_agent
        )
        self.selected_param_keys = get_lora_module_keys(
            sample_state,
            self.param_selection_result.selected_modules
        )

        print(f"Selected {len(self.param_selection_result.selected_modules)} / "
              f"{len(self.param_selection_result.all_modules)} modules")
        print(f"Selected param keys: {len(self.selected_param_keys)}")

    def _compute_delta_and_direction(
        self,
        state_before: Dict[str, torch.Tensor],
        state_after: Dict[str, torch.Tensor],
        selected_keys_only: bool = False
    ) -> Tuple[Dict[str, torch.Tensor], float, Dict[str, torch.Tensor]]:
        """Compute delta, its L2 norm, and unit direction.

        Args:
            state_before: LoRA state before training
            state_after: LoRA state after training
            selected_keys_only: If True and param selection is enabled,
                               only compute delta for selected modules

        Returns:
            Tuple of (delta, norm, direction)
        """
        delta = {}
        squared_sum = 0.0

        for key in state_before.keys():
            # Ensure both tensors are on CPU for computation
            before_tensor = state_before[key].float().cpu()
            after_tensor = state_after[key].float().cpu()

            # If param selection is enabled and selected_keys_only is True,
            # set delta to zero for non-selected modules
            if (selected_keys_only and self.enable_param_selection
                    and self.selected_param_keys is not None
                    and key not in self.selected_param_keys):
                d = torch.zeros_like(before_tensor)
            else:
                d = after_tensor - before_tensor

            delta[key] = d
            squared_sum += (d ** 2).sum().item()

        norm = np.sqrt(squared_sum)

        # Compute unit direction
        direction = {}
        if norm > 1e-10:
            for key in delta.keys():
                direction[key] = delta[key] / norm
        else:
            # Zero direction if norm is too small
            for key in delta.keys():
                direction[key] = torch.zeros_like(delta[key])

        return delta, norm, direction

    def _compute_historical_magnitude(
        self,
        agent_id: int,
        round_from: int,
        round_to: int
    ) -> float:
        """Compute historical update magnitude (gamma).

        根据设计文档，正确的公式为：
        gamma = ||round_{t+interval-1}-c_u_i - round_t-c_i||
        
        这是**单轮**的更新幅度，不是累积幅度！
        
        DFL保存逻辑：
        - round_0: 初始状态，只有 lora_state.pt
        - round_i (i≥1): 第i轮训练后，有 lora_state.pt（聚合后）和 lora_state_pre_agg.pt（训练后聚合前）
        
        关键理解：
        - round_t-c_u_i（第t轮训练后聚合前）保存在 round_{t+1}/lora_state_pre_agg.pt
        - 因为DFL在第t轮训练后、聚合后保存为 round_{t+1}
        
        所以对于校准步骤 (round_from=t, round_to=t+interval)：
        - round_{t+interval-1}-c_u_i 保存在 round_{t+interval-1+1}/lora_state_pre_agg.pt
          = round_{t+interval}/lora_state_pre_agg.pt
        - 但这是错误的！应该是 round_{t+1}/lora_state_pre_agg.pt（单轮更新）
        
        修正：使用 round_{t+1} 的 pre_agg 状态，即单轮更新幅度

        Args:
            agent_id: Agent ID
            round_from: Starting round index (≥0, 从DFL快照加载)
            round_to: Ending round index (> round_from)

        Returns:
            L2 norm of historical update (单轮)
        """
        # 加载 round_from 的 post-aggregation 模型 (round_t-c_i)
        state_from = self.snapshot_loader.load_agent_state(round_from, agent_id, pre_agg=False)
        
        # 加载 round_t-c_u_i（第 t 轮训练后聚合前的模型）
        # 它保存在 round_{t+1}/lora_state_pre_agg.pt
        # 这是**单轮**的更新，与设计文档一致
        gamma_round = round_from + 1  # 单轮更新
        state_to = self.snapshot_loader.load_agent_state(gamma_round, agent_id, pre_agg=True)

        _, norm, _ = self._compute_delta_and_direction(state_from, state_to)
        return norm

    def _scale_direction_by_magnitude(
        self,
        direction: Dict[str, torch.Tensor],
        gamma: float
    ) -> Dict[str, torch.Tensor]:
        """Scale direction vector by magnitude to get delta_hat.

        Args:
            direction: Unit direction vector
            gamma: Historical magnitude

        Returns:
            Scaled delta_hat = gamma * direction
        """
        return {key: gamma * v for key, v in direction.items()}

    def _add_delta_to_state(
        self,
        state: Dict[str, torch.Tensor],
        delta: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Add delta to state: new_state = state + delta."""
        result = {}
        for key in state.keys():
            # Ensure both tensors are on the same device (CPU for storage)
            state_tensor = state[key].cpu() if state[key].is_cuda else state[key]
            delta_tensor = delta[key].cpu() if delta[key].is_cuda else delta[key]
            result[key] = state_tensor + delta_tensor
        return result

    def initialize_models(self):
        """Initialize DFU models for all surviving agents."""
        print(f"\n[初始化] 从DFL快照加载初始模型")
        print(f"  快照路径: {self.snapshot.snapshot_dir}")
        print(f"  可用轮次: {self.snapshot.available_rounds}")
        print(f"  目标遗忘agent: {self.target_agent}")
        print(f"  已移除agents: {self.removed_agents}")
        print(f"  幸存agents: {self.topology.surviving_agents}")

        for agent_id in self.topology.surviving_agents:
            agent_init_state = self.snapshot_loader.load_agent_state(0, agent_id)
            self.current_models[agent_id] = {
                k: v.clone() for k, v in agent_init_state.items()
            }
            print(f"  Agent {agent_id}: 加载 round_0/agent_{agent_id}/lora_state.pt → current_models[{agent_id}]")

        print(f"[初始化完成] new_round_0 = round_0 (共{len(self.current_models)}个agent)\n")

    def _build_calibration_schedule(
        self, available_rounds: List[int]
    ) -> List[Tuple[int, int]]:
        """Build calibration schedule from available DFL rounds.

        Args:
            available_rounds: List of available round indices [0, 1, 2, ..., G]
                              其中 round_0 是初始LoRA，round_1 到 round_G 是训练后的模型

        Returns:
            List of (round_from, round_to) pairs

        关键逻辑：
        - 第一步从 round_0（初始LoRA）开始
        - 按 calibration_interval 间隔生成步骤
        - 最后一步即使间隔不足也要包含最终round

        示例（G=10, interval=2, available_rounds=[0,1,2,...,10]）:
        - (0, 2), (2, 4), (4, 6), (6, 8), (8, 10)  共5步

        示例（G=10, interval=1, available_rounds=[0,1,2,...,10]）:
        - (0, 1), (1, 2), ..., (9, 10)  共10步
        """
        if not available_rounds or len(available_rounds) < 2:
            return []

        calibration_pairs = []
        first_round = available_rounds[0]  # 应该是0（初始LoRA）
        last_round = available_rounds[-1]  # 应该是G（最终模型）
        current_round = first_round

        while current_round < last_round:
            # 找下一个目标round
            current_idx = available_rounds.index(current_round)
            next_round_idx = current_idx + self.calibration_interval

            if next_round_idx >= len(available_rounds):
                # 超出范围，直接跳到最后一个round
                next_round = last_round
            else:
                next_round = available_rounds[next_round_idx]

            calibration_pairs.append((current_round, next_round))
            current_round = next_round

        return calibration_pairs

    def _compute_state_norm(self, state: Dict[str, torch.Tensor]) -> float:
        """Compute L2 norm of a state dict."""
        squared_sum = 0.0
        for key, tensor in state.items():
            squared_sum += (tensor.float().cpu() ** 2).sum().item()
        return np.sqrt(squared_sum)

    def _compute_delta_norm(self, delta: Dict[str, torch.Tensor]) -> float:
        """Compute L2 norm of a delta dict."""
        return self._compute_state_norm(delta)

    def calibrate_step(
        self,
        step_idx: int,
        round_from: int,
        round_to: int,
        batch_size: int = 4,
        grad_accum_steps: int = 2,
        verbose: bool = True,
        show_progress: bool = True
    ):
        """Perform one calibration step for all surviving agents.

        Args:
            step_idx: Current calibration step index
            round_from: DFL snapshot round for t_j
            round_to: DFL snapshot round for t_{j+1}
            batch_size: Batch size for training
            grad_accum_steps: Gradient accumulation steps
            verbose: Print progress
        """
        import gc
        
        if verbose:
            print(f"\n[校准步骤 {step_idx}] round_{round_from} → round_{round_to}")

        # Store delta_hat for all agents (for aggregation)
        agent_delta_hats: Dict[int, Dict[str, torch.Tensor]] = {}
        step_losses = {}

        # Phase 1: For each agent, compute calibration direction and delta_hat
        for agent_id in self.topology.surviving_agents:
            agent = self.agents[agent_id]
            current_state = self.current_models[agent_id]
            train_seed = self.seed + step_idx * 100 + agent_id

            if verbose:
                print(f"  [Agent {agent_id}]")
                print(f"    1. 读取当前模型: current_models[{agent_id}] (即 new_round_{round_from}_agent_{agent_id})")
                print(f"    2. 校准训练: current_models[{agent_id}] → trained_state (训练{self.calibration_steps}步)")

            # 3.1 Local calibration training
            trained_state, loss = agent.calibration_train(
                init_state=current_state,
                steps=self.calibration_steps,
                batch_size=batch_size,
                grad_accum_steps=grad_accum_steps,
                show_progress=show_progress,
                progress_desc=f"Calib train A{agent_id}",
                progress_position=1 if show_progress else None,
                seed=train_seed
            )
            step_losses[agent_id] = loss

            # Compute calibration delta and direction
            delta_cali, norm_cali, dir_cali = self._compute_delta_and_direction(
                current_state, trained_state, selected_keys_only=True
            )

            if verbose:
                print(f"    3. 计算方向: direction = (trained_state - current_state) / ||...||")

            # 3.2 Compute historical magnitude (单轮更新幅度)
            gamma_round = round_from + 1  # 单轮更新
            if verbose:
                print(f"    4. 读取历史状态 (单轮更新):")
                print(f"       - round_{round_from}/agent_{agent_id}/lora_state.pt (post_agg)")
                print(f"       - round_{gamma_round}/agent_{agent_id}/lora_state_pre_agg.pt (pre_agg)")
            
            gamma = self._compute_historical_magnitude(agent_id, round_from, round_to)

            if verbose:
                print(f"    5. gamma = ||round_{gamma_round}_pre_agg - round_{round_from}_post_agg|| (单轮)")

            # 3.3 Construct delta_hat = gamma * dir_cali
            delta_hat = self._scale_direction_by_magnitude(dir_cali, gamma)
            agent_delta_hats[agent_id] = delta_hat

            if verbose:
                print(f"    6. δ_hat = gamma × direction")

            # Record history (在清理变量之前记录)
            self.history["agent_losses"][agent_id].append(loss)
            self.history["calibration_norms"][agent_id].append(norm_cali)
            self.history["historical_gammas"][agent_id].append(gamma)

            # 清理中间变量（不频繁调用cuda清理，避免性能损失）
            del trained_state
            del delta_cali
            del dir_cali

        # Phase 2: Compute local updated models (new_round_t-c_u_i)
        # new_round_t-c_u_i = new_round_t-c_i + delta_hat_i
        agent_updated_models: Dict[int, Dict[str, torch.Tensor]] = {}
        for agent_id in self.topology.surviving_agents:
            agent_updated_models[agent_id] = self._add_delta_to_state(
                self.current_models[agent_id], agent_delta_hats[agent_id]
            )

        # Phase 3: Aggregate updated models with neighbors
        # new_round_{t+interval}-c_i = avg(new_round_t-c_u_{neighbors})
        if verbose:
            print(f"  [聚合阶段]")

        new_models: Dict[int, Dict[str, torch.Tensor]] = {}
        for agent_id in self.topology.surviving_agents:
            neighbors = self.topology.get_neighbors(agent_id)
            all_ids = [agent_id] + neighbors  # Self + unique neighbors

            if verbose:
                inner = ", ".join([f"new_round_{round_from}_u_{i}" for i in all_ids])
                print(f"    Agent {agent_id}: new_round_{round_to} = avg({inner})")

            # Aggregate models (not deltas!)
            aggregated = {}
            weights = self.topology.get_aggregation_weights(agent_id, all_ids)
            for key in agent_updated_models[agent_id].keys():
                stacked = torch.stack([agent_updated_models[i][key] for i in all_ids])
                coeffs = torch.tensor(weights, dtype=stacked.dtype, device=stacked.device)
                aggregated[key] = (stacked * coeffs.view(-1, *([1] * (stacked.dim() - 1)))).sum(dim=0)
            
            new_models[agent_id] = aggregated

        # Update current_models with aggregated results
        for agent_id in self.topology.surviving_agents:
            self.current_models[agent_id] = new_models[agent_id]

        # 清理中间变量（只在校准步骤结束时清理一次）
        del agent_delta_hats
        del agent_updated_models
        del new_models

        if verbose:
            print(f"  [校准步骤 {step_idx} 完成]")

    def verify_unlearning(self, agent_id: int = None, verbose: bool = True) -> UnlearningMetrics:
        """Verify unlearning effectiveness on the current DFU model.

        Args:
            agent_id: Which agent's model to verify. If None, use first surviving agent.
            verbose: Print results

        Returns:
            UnlearningMetrics with verification results
        """
        if agent_id is None:
            agent_id = self.topology.surviving_agents[0]

        # Load the agent's current model
        self.model.set_lora_state_dict(self.current_models[agent_id])

        # Create verifier
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
            show_progress=verbose,
        )

        return metrics

    def evaluate_all_agents(self) -> Dict[int, Dict]:
        """Evaluate all surviving agents on test set."""
        eval_agent_ids = self._get_eval_agent_ids()
        lora_states = {int(aid): self.current_models[int(aid)] for aid in eval_agent_ids}

        per_agent, _summary = evaluate_lora_states(
            model=self.model,
            train_collator=self.collator,
            task_type=self.task_type,
            label_names=self.label_names,
            test_samples=self.test_samples,
            lora_states=lora_states,
            eval_agent_ids=eval_agent_ids,
            max_eval_samples=self.max_eval_samples,
            batch_size=16,
            max_new_tokens_classification=16,
            max_new_tokens_generation=64,
        )

        for aid, metrics in per_agent.items():
            self.history["agent_metrics"][int(aid)].append(metrics)

        return per_agent

    def _compute_avg_metrics(self, round_metrics: Dict[int, Dict]) -> Dict:
        """Compute average metrics across all surviving agents."""
        if not round_metrics:
            return {}

        keys = list(list(round_metrics.values())[0].keys())
        avg = {}
        for key in keys:
            values = [m[key] for m in round_metrics.values()
                     if not np.isnan(m.get(key, 0))]
            avg[key] = np.mean(values) if values else 0.0
        return avg

    def _compute_metrics_with_std(self, round_metrics: Dict[int, Dict]) -> Tuple[Dict, Dict]:
        """Compute average and std metrics across all surviving agents."""
        if not round_metrics:
            return {}, {}

        keys = list(list(round_metrics.values())[0].keys())
        avg: Dict[str, float] = {}
        std: Dict[str, float] = {}
        for key in keys:
            values = [m.get(key, np.nan) for m in round_metrics.values()]
            values = [v for v in values if isinstance(v, (int, float, np.floating)) and not np.isnan(v)]
            avg[key] = float(np.mean(values)) if values else 0.0
            std[key] = float(np.std(values)) if values else 0.0
        return avg, std

    @staticmethod
    def _format_mean_std(mean: float, std: float) -> str:
        return f"{mean:.4f}±{std:.4f}"

    def set_eval_agent_ids(self, agent_ids: List[int]) -> None:
        """设置评估时使用的agent模型ID列表。"""
        self.eval_agent_ids = list(agent_ids)

    def _get_eval_agent_ids(self) -> List[int]:
        """获取用于评估的agent列表（用于加速评估）。"""
        if self.eval_agent_ids is None:
            # Default: evaluate on ALL actually selected agents (selected-k).
            # This is important for sweep/ablation fairness; users can still override
            # via `set_eval_agent_ids()` to speed up debug runs.
            return sorted(list(self.topology.surviving_agents))
        candidate = [aid for aid in self.eval_agent_ids if aid in self.topology.surviving_agents]
        if not candidate:
            return sorted(list(self.topology.surviving_agents))
        return sorted(candidate)

    def _aggregate_unlearning_metrics(
        self,
        per_agent_metrics: Dict[int, UnlearningMetrics]
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Aggregate unlearning metrics across agents into mean and std."""
        if not per_agent_metrics:
            return {}, {}

        metric_keys = [
            "mia_auc",
            "mia_auc_sym",
            "mia_adv",
            "mia_tpr_at_1fpr",
            "mia_tpr_at_5fpr",
            "mia_ks_stat",
            "mia_ks_pvalue",
            "forget_loss",
            "retain_loss",
            "loss_gap",
            "forget_accuracy",
            "retain_accuracy",
            "accuracy_gap"
        ]

        mean_metrics: Dict[str, float] = {}
        std_metrics: Dict[str, float] = {}
        for key in metric_keys:
            values = [
                getattr(m, key)
                for m in per_agent_metrics.values()
                if getattr(m, key) is not None and not np.isnan(getattr(m, key))
            ]
            mean_metrics[key] = float(np.mean(values)) if values else 0.0
            std_metrics[key] = float(np.std(values)) if values else 0.0
        return mean_metrics, std_metrics

    def _evaluate_unlearning_all(
        self,
        verbose: bool = True
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Run unlearning verification for all agents and aggregate results."""
        per_agent_metrics: Dict[int, UnlearningMetrics] = {}
        eval_agent_ids = self._get_eval_agent_ids()
        per_agent_verbose = verbose and len(eval_agent_ids) == 1
        for agent_id in eval_agent_ids:
            per_agent_metrics[agent_id] = self.verify_unlearning(
                agent_id=agent_id,
                verbose=per_agent_verbose
            )

        mean_metrics, std_metrics = self._aggregate_unlearning_metrics(per_agent_metrics)

        if verbose and len(self.topology.surviving_agents) > 1:
            print("\n--- Unlearning Verification (Selected Agents) ---")
            print(
                f"  Forget Loss: {self._format_mean_std(mean_metrics.get('forget_loss', 0), std_metrics.get('forget_loss', 0))}"
            )
            print(
                f"  Retain Loss: {self._format_mean_std(mean_metrics.get('retain_loss', 0), std_metrics.get('retain_loss', 0))}"
            )
            print(
                f"  Loss Gap: {self._format_mean_std(mean_metrics.get('loss_gap', 0), std_metrics.get('loss_gap', 0))}"
            )
            print(
                f"  MIA AUC: {self._format_mean_std(mean_metrics.get('mia_auc', 0), std_metrics.get('mia_auc', 0))}"
            )
            print(
                f"  TPR@1%FPR: {self._format_mean_std(mean_metrics.get('mia_tpr_at_1fpr', 0), std_metrics.get('mia_tpr_at_1fpr', 0))}"
            )
            print(
                f"  TPR@5%FPR: {self._format_mean_std(mean_metrics.get('mia_tpr_at_5fpr', 0), std_metrics.get('mia_tpr_at_5fpr', 0))}"
            )
        return mean_metrics, std_metrics

    def run(
        self,
        batch_size: int = 4,
        grad_accum_steps: int = 2,
        save_dir: Optional[str] = None,
        eval_every: int = 1,
        show_progress: bool = True
    ):
        """Run full DFU calibration process.

        Args:
            batch_size: Batch size for calibration training
            grad_accum_steps: Gradient accumulation steps
            save_dir: Directory to save results
            eval_every: Evaluate every N steps
        """
        available_rounds = self.snapshot.available_rounds
        do_mid_eval = eval_every is not None and eval_every > 0

        if len(available_rounds) < 2:
            raise ValueError("Need at least 2 DFL snapshots for DFU")

        print(f"\n{'='*60}")
        print("Starting DFU (Decentralized Federated Unlearning)")
        print(f"{'='*60}")
        print(f"Target agent to forget: {self.target_agent}")
        if len(self.removed_agents) > 1:
            print(f"Removed agents for this request: {self.removed_agents}")

        # 显示节点选择信息
        print(f"\n--- Agent Selection ---")
        print(f"Selection strategy: {self.selection_strategy}")
        if self.selection_strategy != "full":
            print(f"Selection ratio: {self.selection_ratio}")
        print(f"All surviving agents: {self.topology.all_surviving_agents}")
        print(f"Selected agents: {self.topology.surviving_agents}")
        if self.selection_result.scores:
            print("Agent importance scores:")
            for agent_id, score in sorted(self.selection_result.scores.items()):
                selected = "✓" if agent_id in self.topology.surviving_agents else " "
                print(f"  Agent {agent_id}: {score:.4f} {selected}")
        if self.selection_result.distribution_error is not None:
            print(f"Distribution error: {self.selection_result.distribution_error:.4f}")

        print(f"\n--- DFU Configuration ---")
        print(f"Available DFL rounds: {available_rounds}")
        print(f"Calibration steps per interval: {self.calibration_steps}")
        print(f"Calibration interval: {self.calibration_interval}")

        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            curves_dir = save_path / "curves"
            curves_dir.mkdir(exist_ok=True)

        # Initialize models from DFL round_0 (m_k_0_new = m_k_0_old)
        self.initialize_models()

        # Initial evaluation (Step 0 - before calibration)
        if do_mid_eval:
            print("\n=== Initial Evaluation (Step 0) ===")
            print("Evaluating agents before DFU calibration...")
            
            # For TOFU, skip token_f1/exact_match evaluation, only use TOFU metrics
            if self.task_type == "tofu":
                self.history["calibration_steps"].append(0)
                # TOFU evaluation will be done below
            else:
                initial_metrics = self.evaluate_all_agents()
                primary = "macro_f1" if self.task_type == "classification" else "token_f1"
                initial_avg_metrics, initial_std_metrics, best_id, best_metrics, _ = summarize_per_agent_metrics(
                    initial_metrics, primary_metric=primary
                )
                self.history["avg_metrics"].append(initial_avg_metrics)
                self.history["avg_metrics_std"].append(initial_std_metrics)
                self.history["best_metrics"].append(best_metrics)
                self.history["best_agent_ids"].append(best_id)
                self.history["calibration_steps"].append(0)

                if self.task_type == "classification":
                    print(
                        f"Initial Avg Accuracy: "
                        f"{self._format_mean_std(initial_avg_metrics.get('accuracy', 0), initial_std_metrics.get('accuracy', 0))}, "
                        f"Macro-F1 (mean): {self._format_mean_std(initial_avg_metrics.get('macro_f1', 0), initial_std_metrics.get('macro_f1', 0))}, "
                        f"Macro-F1 (best): {best_metrics.get('macro_f1', 0):.4f} (agent {best_id})"
                    )
                else:
                    print(
                        f"Initial Avg Token-F1: "
                        f"{self._format_mean_std(initial_avg_metrics.get('token_f1', 0), initial_std_metrics.get('token_f1', 0))}, "
                        f"Exact Match (mean): {self._format_mean_std(initial_avg_metrics.get('exact_match', 0), initial_std_metrics.get('exact_match', 0))}, "
                        f"Token-F1 (best): {best_metrics.get('token_f1', 0):.4f} (agent {best_id})"
                    )

            # Initial unlearning verification (skip for TOFU; rely on TOFU metrics instead)
            if self.task_type != "tofu":
                initial_unlearn_mean, initial_unlearn_std = self._evaluate_unlearning_all(verbose=True)
                self.history["unlearning_metrics"].append({
                    "step": 0,
                    **initial_unlearn_mean,
                    **{f"{k}_std": v for k, v in initial_unlearn_std.items()}
                })
            
            # Initial TOFU evaluation
            if self.task_type == "tofu" and hasattr(self, 'tofu_evaluator'):
                initial_tofu_results = self.evaluate_tofu(verbose=True)
                initial_tofu_results["step"] = 0
                self.history["tofu_metrics"].append(initial_tofu_results)

            # Update curves with initial evaluation
            if save_dir:
                self._plot_curves(curves_dir)
        else:
            print("\nSkipping evaluations (eval_every<=0)")

        # Build calibration schedule
        # 从 round_0（初始LoRA）开始，按calibration_interval间隔进行
        calibration_pairs = self._build_calibration_schedule(available_rounds)

        print(f"\nCalibration schedule: {calibration_pairs}")
        print(f"  round_0 = 初始未训练LoRA, round_1...{available_rounds[-1]} = 训练后模型")

        # Run calibration
        last_eval_step = 0
        calib_iter = calibration_pairs
        calib_pbar = None
        if show_progress and calibration_pairs:
            calib_pbar = tqdm(
                calibration_pairs,
                desc="Calibration steps",
                dynamic_ncols=True,
                leave=True
            )
            calib_iter = calib_pbar

        for step_idx, (round_from, round_to) in enumerate(calib_iter, start=1):
            self.calibrate_step(
                step_idx=step_idx,
                round_from=round_from,
                round_to=round_to,
                batch_size=batch_size,
                grad_accum_steps=grad_accum_steps,
                show_progress=show_progress
            )

            # Evaluation
            if do_mid_eval and step_idx % eval_every == 0:
                print("Evaluating...")
                self.history["calibration_steps"].append(step_idx)
                
                # For TOFU, skip token_f1/exact_match evaluation, only use TOFU metrics
                if self.task_type == "tofu":
                    # TOFU-specific evaluation only
                    if hasattr(self, 'tofu_evaluator'):
                        tofu_results = self.evaluate_tofu(verbose=True)
                        tofu_results["step"] = step_idx
                        self.history["tofu_metrics"].append(tofu_results)
                else:
                    # Standard evaluation for other tasks
                    round_metrics = self.evaluate_all_agents()
                    primary = "macro_f1" if self.task_type == "classification" else "token_f1"
                    avg_metrics, std_metrics, best_id, best_metrics, _ = summarize_per_agent_metrics(
                        round_metrics, primary_metric=primary
                    )
                    self.history["avg_metrics"].append(avg_metrics)
                    self.history["avg_metrics_std"].append(std_metrics)
                    self.history["best_metrics"].append(best_metrics)
                    self.history["best_agent_ids"].append(best_id)
                    if self.task_type == "classification":
                        print(
                            f"Avg Accuracy: {self._format_mean_std(avg_metrics.get('accuracy', 0), std_metrics.get('accuracy', 0))}, "
                            f"Macro-F1 (mean): {self._format_mean_std(avg_metrics.get('macro_f1', 0), std_metrics.get('macro_f1', 0))}, "
                            f"Macro-F1 (best): {best_metrics.get('macro_f1', 0):.4f} (agent {best_id})"
                        )
                    else:
                        print(
                            f"Avg Token-F1: {self._format_mean_std(avg_metrics.get('token_f1', 0), std_metrics.get('token_f1', 0))}, "
                            f"Exact Match (mean): {self._format_mean_std(avg_metrics.get('exact_match', 0), std_metrics.get('exact_match', 0))}, "
                            f"Token-F1 (best): {best_metrics.get('token_f1', 0):.4f} (agent {best_id})"
                        )

                # Unlearning verification (skip for TOFU; rely on TOFU metrics instead)
                if self.task_type != "tofu":
                    unlearn_mean, unlearn_std = self._evaluate_unlearning_all(verbose=True)
                    self.history["unlearning_metrics"].append({
                        "step": step_idx,
                        **unlearn_mean,
                        **{f"{k}_std": v for k, v in unlearn_std.items()}
                    })
                last_eval_step = step_idx

            # Save intermediate results (only plots, not models to save disk space)
            if save_dir:
                # self._save_step(save_path, step_idx)  # Disabled to save disk space
                self._plot_curves(curves_dir)
                # Plot TOFU curves if available
                if self.task_type == "tofu" and hasattr(self, 'tofu_evaluator'):
                    self._plot_tofu_curves(curves_dir)

        if calib_pbar is not None:
            calib_pbar.close()

        # 如果未在最后一步评估且需要最终评估（例如 eval_every<=0），执行末轮评估
        total_steps = len(calibration_pairs)
        if not do_mid_eval or last_eval_step != total_steps:
            final_step = total_steps
            print(f"\n=== Final Evaluation (Step {final_step}) ===")
            self.history["calibration_steps"].append(final_step)
            
            # 初始化 test set 指标
            test_avg_metrics, test_std_metrics = {}, {}
            test_best_metrics, test_best_agent_id = {}, None
            
            if self.task_type == "tofu" and hasattr(self, 'tofu_evaluator'):
                tofu_results = self.evaluate_tofu(verbose=True)
                tofu_results["step"] = final_step
                self.history["tofu_metrics"].append(tofu_results)
            elif self.task_type != "tofu":
                round_metrics = self.evaluate_all_agents()
                primary = "macro_f1" if self.task_type == "classification" else "token_f1"
                avg_metrics, std_metrics, best_id, best_metrics, _ = summarize_per_agent_metrics(
                    round_metrics, primary_metric=primary
                )
                self.history["avg_metrics"].append(avg_metrics)
                self.history["avg_metrics_std"].append(std_metrics)
                self.history["best_metrics"].append(best_metrics)
                self.history["best_agent_ids"].append(best_id)
                # 保存 test set 指标用于添加到 unlearning_metrics
                test_avg_metrics = avg_metrics
                test_std_metrics = std_metrics
                test_best_metrics = best_metrics
                test_best_agent_id = best_id
                if self.task_type == "classification":
                    print(
                        f"Avg Accuracy: {self._format_mean_std(avg_metrics.get('accuracy', 0), std_metrics.get('accuracy', 0))}, "
                        f"Macro-F1 (mean): {self._format_mean_std(avg_metrics.get('macro_f1', 0), std_metrics.get('macro_f1', 0))}, "
                        f"Macro-F1 (best): {best_metrics.get('macro_f1', 0):.4f} (agent {best_id})"
                    )
                else:
                    print(
                        f"Avg Token-F1: {self._format_mean_std(avg_metrics.get('token_f1', 0), std_metrics.get('token_f1', 0))}, "
                        f"Exact Match (mean): {self._format_mean_std(avg_metrics.get('exact_match', 0), std_metrics.get('exact_match', 0))}, "
                        f"Token-F1 (best): {best_metrics.get('token_f1', 0):.4f} (agent {best_id})"
                    )

            if self.task_type != "tofu":
                # Unlearning verification
                unlearn_mean, unlearn_std = self._evaluate_unlearning_all(verbose=True)

                # 构建 unlearning_metrics，包含 test set 指标
                unlearn_entry = {
                    "step": final_step,
                    **unlearn_mean,
                    **{f"{k}_std": v for k, v in unlearn_std.items()},
                }
                # 添加 test set 指标到 unlearning_metrics
                if test_avg_metrics:
                    unlearn_entry.update({
                        "test_accuracy": test_avg_metrics.get("accuracy", 0),
                        "test_macro_f1": test_avg_metrics.get("macro_f1", 0),
                        "test_precision": test_avg_metrics.get("precision", 0),
                        "test_recall": test_avg_metrics.get("recall", 0),
                        "test_accuracy_std": test_std_metrics.get("accuracy", 0),
                        "test_macro_f1_std": test_std_metrics.get("macro_f1", 0),
                    })
                if test_best_metrics:
                    unlearn_entry.update({
                        "test_best_agent_id": test_best_agent_id,
                        "test_accuracy_best": test_best_metrics.get("accuracy", 0),
                        "test_macro_f1_best": test_best_metrics.get("macro_f1", 0),
                    })
                self.history["unlearning_metrics"].append(unlearn_entry)
            if save_dir:
                self._plot_curves(curves_dir)
                if self.task_type == "tofu" and hasattr(self, 'tofu_evaluator'):
                    self._plot_tofu_curves(curves_dir)

        # Final save
        if save_dir:
            self._save_history(save_path)
            if self.save_lora_states:
                try:
                    self._save_final(save_path)
                except Exception as e:  # noqa: BLE001
                    print(f"[WARN] Failed to save final LoRA states: {e}")

        print(f"\n{'='*60}")
        print("DFU completed!")
        print(f"{'='*60}")

    def _save_step(self, save_path: Path, step_idx: int):
        """Save all agent models for this calibration step."""
        step_dir = save_path / f"step_{step_idx}"
        step_dir.mkdir(exist_ok=True)

        for agent_id in self.topology.surviving_agents:
            agent_dir = step_dir / f"agent_{agent_id}"
            agent_dir.mkdir(exist_ok=True)
            torch.save(self.current_models[agent_id], agent_dir / "lora_state.pt")

    def _save_final(self, save_path: Path):
        """Save final DFU models."""
        final_dir = save_path / "final"
        final_dir.mkdir(exist_ok=True)

        for agent_id in self.topology.surviving_agents:
            agent_dir = final_dir / f"agent_{agent_id}"
            agent_dir.mkdir(exist_ok=True)
            torch.save(self.current_models[agent_id], agent_dir / "lora_state.pt")

    def _save_history(self, save_path: Path):
        """Save training history."""
        def safe_float(v):
            if isinstance(v, (int, float)):
                return None if np.isnan(v) else float(v)
            return v

        history_json = {
            "calibration_steps": self.history["calibration_steps"],
            "agent_losses": {
                str(k): [safe_float(v) for v in vals]
                for k, vals in self.history["agent_losses"].items()
            },
            "agent_metrics": {
                str(k): [{mk: safe_float(mv) for mk, mv in m.items()} for m in metrics]
                for k, metrics in self.history["agent_metrics"].items()
            },
            "avg_metrics": [
                {k: safe_float(v) for k, v in m.items()}
                for m in self.history["avg_metrics"]
            ],
            "avg_metrics_std": [
                {k: safe_float(v) for k, v in m.items()}
                for m in self.history.get("avg_metrics_std", [])
            ],
            "best_metrics": [
                {k: safe_float(v) for k, v in m.items()}
                for m in self.history.get("best_metrics", [])
            ],
            "best_agent_ids": [safe_float(v) for v in self.history.get("best_agent_ids", [])],
            "calibration_norms": {
                str(k): [safe_float(v) for v in vals]
                for k, vals in self.history["calibration_norms"].items()
            },
            "historical_gammas": {
                str(k): [safe_float(v) for v in vals]
                for k, vals in self.history["historical_gammas"].items()
            },
            "unlearning_metrics": [
                {k: safe_float(v) for k, v in m.items()}
                for m in self.history.get("unlearning_metrics", [])
            ],
            "tofu_metrics": [
                {k: safe_float(v) for k, v in m.items()}
                for m in self.history.get("tofu_metrics", [])
            ],
            "config": {
                "target_agent": self.target_agent,
                "removed_agents": self.removed_agents,
                "calibration_steps": self.calibration_steps,
                "calibration_interval": self.calibration_interval,
                "lr": self.lr,
                "surviving_agents": self.topology.surviving_agents,
                "all_surviving_agents": self.topology.all_surviving_agents,
                "dfl_snapshot": str(self.snapshot.snapshot_dir),
                # 节点选择配置
                "selection_strategy": self.selection_strategy,
                "selection_ratio": self.selection_ratio if self.selection_strategy != "full" else None,
                "selection_seed": self.selection_seed if self.selection_strategy == "random" else None,
                "selected_agents": self.selection_result.selected_agents,
                "selection_scores": {
                    str(k): safe_float(v)
                    for k, v in self.selection_result.scores.items()
                } if self.selection_result.scores else None,
                "distribution_error": safe_float(self.selection_result.distribution_error)
                    if self.selection_result.distribution_error is not None else None
            }
        }

        # 添加最终统计（最后一轮的平均值±标准差）
        if self.history["agent_metrics"] and self.history["avg_metrics"]:
            # 获取最后一轮的所有agent指标
            final_metrics = {}
            for agent_id, metrics_list in self.history["agent_metrics"].items():
                if metrics_list:
                    final_metrics[int(agent_id)] = metrics_list[-1]
            
            primary = "macro_f1" if self.task_type == "classification" else "token_f1"
            history_json["final_stats"] = build_final_stats(
                final_metrics,
                primary_metric=primary,
                eval_agent_ids=sorted(final_metrics.keys()),
            )
            
            # 添加MIA最终统计
            if self.history.get("unlearning_metrics"):
                final_unlearn = self.history["unlearning_metrics"][-1]
                history_json["final_stats"]["mia_auc"] = safe_float(final_unlearn.get("mia_auc", 0))
                history_json["final_stats"]["mia_tpr_at_1fpr"] = safe_float(final_unlearn.get("mia_tpr_at_1fpr", 0))
                history_json["final_stats"]["mia_tpr_at_5fpr"] = safe_float(final_unlearn.get("mia_tpr_at_5fpr", 0))
                history_json["final_stats"]["mia_auc_std"] = safe_float(final_unlearn.get("mia_auc_std", 0))
                history_json["final_stats"]["mia_tpr_at_1fpr_std"] = safe_float(final_unlearn.get("mia_tpr_at_1fpr_std", 0))
                history_json["final_stats"]["mia_tpr_at_5fpr_std"] = safe_float(final_unlearn.get("mia_tpr_at_5fpr_std", 0))

        with open(save_path / "history.json", "w") as f:
            json.dump(history_json, f, indent=2)

    def _plot_curves(self, curves_dir: Path):
        """Plot and save training curves."""
        curves_dir = Path(curves_dir)
        curves_dir.mkdir(parents=True, exist_ok=True)
        steps = self.history["calibration_steps"]
        if len(steps) < 1:
            return

        # 1. Loss curves (only plot if we have training losses)
        # Note: Step 0 has no training loss (initial evaluation only)
        if any(len(losses) > 0 for losses in self.history["agent_losses"].values()):
            fig, ax = plt.subplots(figsize=(10, 6))

            # Get the steps that have corresponding loss data
            max_loss_len = max(len(losses) for losses in self.history["agent_losses"].values())
            loss_steps = [s for s in steps if s > 0][:max_loss_len]
            if len(loss_steps) < max_loss_len:
                loss_steps = list(range(1, max_loss_len + 1))

            for agent_id, losses in self.history["agent_losses"].items():
                if losses and len(loss_steps) > 0:
                    ax.plot(loss_steps[:len(losses)], losses, alpha=0.5, label=f"Agent {agent_id}")

            # Average loss
            if len(loss_steps) > 0:
                avg_losses = []
                for i in range(len(loss_steps)):
                    agent_vals = [
                        self.history["agent_losses"][aid][i]
                        for aid in self.topology.surviving_agents
                        if i < len(self.history["agent_losses"][aid])
                        and not np.isnan(self.history["agent_losses"][aid][i])
                    ]
                    avg_losses.append(np.mean(agent_vals) if agent_vals else np.nan)
                ax.plot(loss_steps, avg_losses, 'k-', linewidth=2, label="Average")

            ax.set_xlabel("Calibration Step")
            ax.set_ylabel("Training Loss")
            ax.set_title("DFU Calibration Loss Curves")
            ax.legend(loc='upper right', fontsize='small', ncol=2)
            plt.tight_layout()
            plt.savefig(curves_dir / "loss_curves.png", dpi=150)
            plt.close()

        # 2. Calibration norms and gammas (only plot if we have calibration data)
        if any(len(norms) > 0 for norms in self.history["calibration_norms"].values()):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            # Get the steps that have corresponding calibration data
            max_norm_len = max(len(norms) for norms in self.history["calibration_norms"].values())
            max_gamma_len = max(len(gammas) for gammas in self.history["historical_gammas"].values())
            max_calib_len = max(max_norm_len, max_gamma_len)
            calib_steps = [s for s in steps if s > 0][:max_calib_len]
            if len(calib_steps) < max_calib_len:
                calib_steps = list(range(1, max_calib_len + 1))

            for agent_id in self.topology.surviving_agents:
                norms = self.history["calibration_norms"][agent_id]
                gammas = self.history["historical_gammas"][agent_id]
                if norms and len(calib_steps) > 0:
                    ax1.plot(calib_steps[:len(norms)], norms, alpha=0.5, label=f"Agent {agent_id}")
                if gammas and len(calib_steps) > 0:
                    ax2.plot(calib_steps[:len(gammas)], gammas, alpha=0.5, label=f"Agent {agent_id}")

            ax1.set_xlabel("Calibration Step")
            ax1.set_ylabel("Calibration Norm")
            ax1.set_title("Local Calibration Update Norms")
            ax1.legend(fontsize='small')

            ax2.set_xlabel("Calibration Step")
            ax2.set_ylabel("Historical Gamma")
            ax2.set_title("Historical Update Magnitudes")
            ax2.legend(fontsize='small')

            plt.tight_layout()
            plt.savefig(curves_dir / "calibration_stats.png", dpi=150)
            plt.close()

        # 3. Metrics curves
        if self.history["avg_metrics"]:
            if self.task_type == "classification":
                self._plot_metric_curve(curves_dir, "accuracy", "Accuracy")
                self._plot_metric_curve(curves_dir, "macro_f1", "Macro-F1")
            else:
                self._plot_metric_curve(curves_dir, "token_f1", "Token F1")
                self._plot_metric_curve(curves_dir, "exact_match", "Exact Match")

        # 4. MIA curves (unlearning verification)
        if self.history.get("unlearning_metrics"):
            self._plot_mia_curves(curves_dir)

    def _plot_mia_curves(self, curves_dir: Path):
        """Plot MIA (Membership Inference Attack) curves with final stats."""
        unlearn_metrics = self.history.get("unlearning_metrics", [])
        if not unlearn_metrics:
            return

        steps = [m["step"] for m in unlearn_metrics]
        mia_auc = [m.get("mia_auc", 0) for m in unlearn_metrics]
        mia_tpr_1fpr = [m.get("mia_tpr_at_1fpr", 0) for m in unlearn_metrics]
        mia_tpr_5fpr = [m.get("mia_tpr_at_5fpr", 0) for m in unlearn_metrics]

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(steps, mia_auc, 'b-o', linewidth=2, markersize=6, label="MIA AUC")
        ax.plot(steps, mia_tpr_1fpr, 'r-s', linewidth=2, markersize=6, label="TPR@1%FPR")
        ax.plot(steps, mia_tpr_5fpr, 'g-^', linewidth=2, markersize=6, label="TPR@5%FPR")

        # 添加最终统计注释
        if mia_auc:
            final_auc = mia_auc[-1]
            final_tpr_1 = mia_tpr_1fpr[-1]
            final_tpr_5 = mia_tpr_5fpr[-1]
            
            # 在图上添加最终统计文本框
            stats_text = f'Final MIA Stats:\nAUC: {final_auc:.4f}\nTPR@1%FPR: {final_tpr_1:.4f}\nTPR@5%FPR: {final_tpr_5:.4f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label="Random (0.5)")
        ax.set_xlabel("Calibration Step")
        ax.set_ylabel("MIA Metric")
        ax.set_title("MIA (Membership Inference Attack) Curves")
        ax.legend(loc='upper right', fontsize='small')
        ax.set_ylim(0, 1)
        plt.tight_layout()
        plt.savefig(curves_dir / "mia_curves.png", dpi=150)
        plt.close()

    def _plot_metric_curve(self, curves_dir: Path, metric_key: str, metric_name: str):
        """Plot a single metric curve for all agents with final stats annotation."""
        steps = self.history["calibration_steps"]

        fig, ax = plt.subplots(figsize=(10, 6))

        for agent_id, metrics_list in self.history["agent_metrics"].items():
            if metrics_list:
                values = [m.get(metric_key, 0) for m in metrics_list[:len(steps)]]
                ax.plot(range(len(values)), values, alpha=0.5, label=f"Agent {agent_id}")

        # Average
        if self.history["avg_metrics"]:
            avg_values = [m.get(metric_key, 0) for m in self.history["avg_metrics"]]
            ax.plot(range(len(avg_values)), avg_values, 'k-', linewidth=2, label="Average")

        # 计算最后一轮的统计并添加注释
        final_values = []
        for agent_id, metrics_list in self.history["agent_metrics"].items():
            if metrics_list:
                final_values.append(metrics_list[-1].get(metric_key, 0))
        
        if final_values:
            final_mean = np.mean(final_values)
            final_std = np.std(final_values)
            # 在图上添加最终统计注释
            ax.annotate(f'Final: {final_mean:.4f}±{final_std:.4f}',
                       xy=(len(avg_values)-1, final_mean),
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

        ax.set_xlabel("Evaluation Step")
        ax.set_ylabel(metric_name)
        ax.set_title(f"DFU {metric_name} Curves")
        ax.legend(loc='lower right', fontsize='small', ncol=2)
        plt.tight_layout()
        plt.savefig(curves_dir / f"{metric_key}_curves.png", dpi=150)
        plt.close()

    # ============================================================
    # TOFU Dataset Evaluation Support
    # ============================================================
    
    def init_tofu_evaluator(self):
        """Initialize TOFU evaluator if task_type is 'tofu'."""
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
        
        # Initialize TOFU-specific history
        self.history["tofu_metrics"] = []
    
    def evaluate_tofu(self, agent_id: int = None, verbose: bool = True) -> Dict:
        """Evaluate using TOFU metrics.
        
        Args:
            agent_id: Which agent's model to evaluate. If None, use first surviving agent.
            verbose: Print results
            
        Returns:
            Dictionary of TOFU metrics
        """
        if not hasattr(self, 'tofu_evaluator'):
            return {}

        if agent_id is None:
            agent_ids = self._get_eval_agent_ids()
        else:
            agent_ids = [agent_id]

        per_agent_results: Dict[int, Dict] = {}
        per_agent_verbose = verbose and len(agent_ids) == 1
        retain_samples = self.tofu_external_retain_samples or self.retain_samples
        forget_samples = self.tofu_external_forget_samples or self.forget_samples

        for aid in agent_ids:
            self.model.set_lora_state_dict(self.current_models[aid])
            results = self.tofu_evaluator.evaluate_all(
                retain_samples=retain_samples,
                forget_samples=forget_samples,
                retain_truth_ratios=self.retrain_tr,  # 传入 retrain TR 用于计算 FQ
                compute_generation=True,  # Enable ROUGE computation
                verbose=per_agent_verbose,
                print_summary=per_agent_verbose,
                show_progress=verbose,
                max_samples=self.max_eval_samples
            )
            per_agent_results[aid] = results.to_dict()

        aggregated = self._aggregate_tofu_metrics(per_agent_results)

        if verbose and len(agent_ids) > 1:
            self._print_tofu_summary(aggregated, len(agent_ids))

        return aggregated

    def set_tofu_external_eval_samples(self, *, retain_samples: List[Sample], forget_samples: List[Sample]) -> None:
        """Set official TOFU eval samples for retain/forget.

        This is useful when training samples do not contain perturbed answers.
        """
        self.tofu_external_retain_samples = list(retain_samples)
        self.tofu_external_forget_samples = list(forget_samples)

    def _aggregate_tofu_metrics(self, per_agent_results: Dict[int, Dict]) -> Dict:
        """Aggregate TOFU metrics across agents into mean and std."""
        if not per_agent_results:
            return {}

        keys = list(next(iter(per_agent_results.values())).keys())
        aggregated: Dict[str, float] = {}
        for key in keys:
            values = [m.get(key, np.nan) for m in per_agent_results.values()]
            values = [v for v in values if isinstance(v, (int, float, np.floating)) and not np.isnan(v)]
            mean = float(np.mean(values)) if values else 0.0
            std = float(np.std(values)) if values else 0.0
            aggregated[key] = mean
            aggregated[f"{key}_std"] = std
        return aggregated

    def _print_tofu_summary(self, metrics: Dict, num_agents: int) -> None:
        """Print TOFU metrics as mean±std."""
        def fmt(key: str) -> str:
            return self._format_mean_std(metrics.get(key, 0), metrics.get(f"{key}_std", 0))

        print(f"\n--- TOFU Evaluation (Selected Agents, n={num_agents}) ---")
        print(
            f"  Retain: ROUGE-L={fmt('retain_rougeL_recall')}, "
            f"Prob={fmt('retain_probability')}, TR={fmt('retain_truth_ratio')}"
        )
        print(
            f"  Forget: ROUGE-L={fmt('forget_rougeL_recall')}, "
            f"Prob={fmt('forget_probability')}, TR={fmt('forget_truth_ratio')}"
        )
        if "real_authors_rougeL_recall" in metrics:
            print(
                f"  Real Authors: ROUGE-L={fmt('real_authors_rougeL_recall')}, "
                f"Prob={fmt('real_authors_probability')}, TR={fmt('real_authors_truth_ratio')}"
            )
        if "real_world_rougeL_recall" in metrics:
            print(
                f"  Real World: ROUGE-L={fmt('real_world_rougeL_recall')}, "
                f"Prob={fmt('real_world_probability')}, TR={fmt('real_world_truth_ratio')}"
            )
        print(f"\n  Model Utility: {fmt('model_utility')}")
        if "forget_quality" in metrics:
            print(f"  Forget Quality: {fmt('forget_quality')}")
        print("--- TOFU Evaluation Complete ---\n")
    
    def _plot_tofu_curves(self, curves_dir: Path):
        """Plot TOFU-specific metric curves."""
        curves_dir = Path(curves_dir)
        curves_dir.mkdir(parents=True, exist_ok=True)
        if "tofu_metrics" not in self.history or not self.history["tofu_metrics"]:
            return
        
        tofu_metrics = self.history["tofu_metrics"]
        steps = [m.get("step", i) for i, m in enumerate(tofu_metrics)]
        
        # Plot Model Utility and Forget Quality
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Model Utility
        mu_values = [m.get("model_utility", 0) for m in tofu_metrics]
        ax1.plot(steps, mu_values, 'b-o', linewidth=2, markersize=6)
        ax1.set_xlabel("Calibration Step")
        ax1.set_ylabel("Model Utility")
        ax1.set_title("TOFU Model Utility")
        ax1.set_ylim(0, 1)
        if mu_values:
            ax1.annotate(f'Final: {mu_values[-1]:.4f}',
                        xy=(steps[-1], mu_values[-1]),
                        xytext=(10, 10), textcoords='offset points',
                        fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        # Forget Quality
        fq_values = [m.get("forget_quality", 0) for m in tofu_metrics]
        ax2.plot(steps, fq_values, 'r-o', linewidth=2, markersize=6)
        ax2.set_xlabel("Calibration Step")
        ax2.set_ylabel("Forget Quality (p-value)")
        ax2.set_title("TOFU Forget Quality")
        ax2.set_ylim(0, 1)
        if fq_values:
            ax2.annotate(f'Final: {fq_values[-1]:.4f}',
                        xy=(steps[-1], fq_values[-1]),
                        xytext=(10, 10), textcoords='offset points',
                        fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(curves_dir / "tofu_metrics.png", dpi=150)
        plt.close()
        
        # Plot individual metrics
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        metric_pairs = [
            ("retain_probability", "Retain Probability"),
            ("retain_truth_ratio", "Retain Truth Ratio"),
            ("forget_probability", "Forget Probability"),
            ("forget_truth_ratio", "Forget Truth Ratio")
        ]
        
        for ax, (metric_key, metric_name) in zip(axes.flat, metric_pairs):
            values = [m.get(metric_key, 0) for m in tofu_metrics]
            ax.plot(steps, values, 'g-o', linewidth=2, markersize=6)
            ax.set_xlabel("Calibration Step")
            ax.set_ylabel(metric_name)
            ax.set_title(metric_name)
            ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(curves_dir / "tofu_detailed_metrics.png", dpi=150)
        plt.close()
