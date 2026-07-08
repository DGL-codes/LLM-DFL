"""D-FedRecovery: 基于历史梯度校正的去中心化联邦遗忘方法。

参考: FedRecovery (baseline-reference/FedAU/baselines/fedrecovery_base.py)

核心思想：
1. 利用DFL训练过程中保存的历史模型快照
2. 计算每轮的全局更新和目标客户端更新
3. 通过梯度残差校正来移除目标客户端的贡献
4. 可选添加高斯噪声保护隐私

去中心化适配：
- 原始FedRecovery假设有中心服务器保存全局模型历史
- 在去中心化设置中，我们使用环状聚合后的模型作为"全局"模型
- 每个客户端独立计算校正，然后通过环状聚合传播

流程：
1. 校正阶段：基于历史梯度计算校正量，更新模型
2. 恢复阶段（可选）：移除目标客户端后，幸存客户端继续联邦训练
"""
import json
import torch
import numpy as np
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
from .trainer import DFURingTopology, DFUAgent
from ..models.lora_model import LoRAModelWrapper
from ..models.trainer import LLMTrainer
from ..models.multi_agent_eval import build_final_stats, evaluate_lora_states
from ..data.base import Sample
from ..data.collator import LLMCollator
from ..data.partitioner import PartitionInfo
from ..utils.determinism import derive_seed


class DFedRecoveryTrainer:
    """D-FedRecovery: 基于历史梯度校正的去中心化联邦遗忘方法。
    
    流程：
    1. 校正阶段：
       - 从DFL快照加载历史模型
       - 计算每轮的全局更新 delta_global 和目标客户端更新 delta_target
       - 计算梯度残差并累积校正量
       - 应用校正到最终模型
    2. 恢复阶段（可选）：
       - 移除目标客户端
       - 幸存客户端继续联邦训练恢复性能
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
        # D-FedRecovery 特有参数
        correction_weight: float = 5.0,  # 校正权重（原论文使用5）
        noise_std: float = 0.0,  # 高斯噪声标准差（0表示不加噪声）
        correction_mode: str = "decentralized_replay",
        recovery_rounds: int = 0,  # 恢复轮数（0表示不做恢复）
        recovery_epochs: int = 1,
        recovery_lr: float = 1e-4,
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
        """初始化 D-FedRecovery Trainer。"""
        self.snapshot_loader = snapshot_loader
        self.snapshot = snapshot_loader.snapshot
        self.model = model
        self.collator = collator
        self.all_samples = all_samples
        self.val_samples = val_samples
        self.test_samples = test_samples
        self.partition = partition
        self.target_agent = target_agent
        
        # D-FedRecovery 特有参数
        self.correction_weight = correction_weight
        self.noise_std = noise_std
        self.correction_mode = str(correction_mode or "decentralized_replay").lower().strip()
        if self.correction_mode not in {"decentralized_replay", "residual"}:
            raise ValueError(
                f"Unsupported FedRecovery correction_mode={correction_mode!r}; "
                "expected 'decentralized_replay' or 'residual'."
            )
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
        self.optimizer_name = optimizer_name
        self.save_lora_states = bool(save_lora_states)
        
        # 节点选择参数
        self.selection_strategy = selection_strategy
        self.selection_ratio = selection_ratio
        self.selection_count = selection_count
        self.selection_seed = selection_seed
        self.selection_epsilon = selection_epsilon
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
        
        # 执行节点选择（用于恢复阶段）
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
        
        # 初始化恢复阶段拓扑（移除目标客户端）
        self.recovery_topology = DFURingTopology(
            self.snapshot.num_agents,
            removed_agent=target_agent,
            selected_agents=selected_agents,
            aggregation_weights=self.selection_result.weights,
            aggregation_scope=self.tdb_aggregation_scope,
        )
        
        # 初始化 agents（用于恢复阶段）
        self.agents: Dict[int, DFUAgent] = {}
        for agent_id in self.recovery_topology.surviving_agents:
            indices = partition.agent_indices[agent_id]
            local_samples = [all_samples[idx] for idx in indices]
            
            self.agents[agent_id] = DFUAgent(
                agent_id=agent_id,
                local_samples=local_samples,
                model=model,
                collator=collator,
                lr=self.recovery_lr,
                device=device,
                selected_param_keys=self.selected_param_keys,
                optimizer_name=optimizer_name
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
            "correction_phase": {
                "rounds_processed": 0,
                "total_correction_norm": 0.0,
            },
            "recovery_phase": {
                "rounds": [],
                "agent_losses": {i: [] for i in selected_agents},
            },
            "unlearning_metrics": [],
            "tofu_metrics": [],
        }

        # Optional: official TOFU eval samples (retain_perturbed / forgetXX_perturbed).
        # If provided, evaluate_tofu() uses these instead of client training samples.
        self.tofu_external_retain_samples: Optional[List[Sample]] = None
        self.tofu_external_forget_samples: Optional[List[Sample]] = None
        
        # TOFU 评估器
        self.tofu_evaluator = None
        if self.task_type == "tofu":
            self.init_tofu_evaluator()
        
        # 评估用的 agent IDs
        self.eval_agent_ids: Optional[List[int]] = None

        # Cache per-agent test-set metrics for final_stats writing.
        self.last_test_per_agent_results: Optional[Dict[int, Dict[str, float]]] = None

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
            print(f"  TPR@1%FPR: {mean_metrics.get('mia_tpr_at_1fpr', 0):.4f}±{std_metrics.get('mia_tpr_at_1fpr', 0):.4f}")
            print(f"  TPR@5%FPR: {mean_metrics.get('mia_tpr_at_5fpr', 0):.4f}±{std_metrics.get('mia_tpr_at_5fpr', 0):.4f}")
            print(f"  Forget Loss: {mean_metrics.get('forget_loss', 0):.4f}±{std_metrics.get('forget_loss', 0):.4f}")
            print(f"  Retain Loss: {mean_metrics.get('retain_loss', 0):.4f}±{std_metrics.get('retain_loss', 0):.4f}")
        
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

        # Cache per-agent results (used when writing final_stats).
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

    def evaluate_tofu(self, agent_id: int = None, verbose: bool = True) -> Dict:
        """使用 TOFU 指标评估。"""
        if self.tofu_evaluator is None:
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

    def _compute_round_weights(self, num_rounds: int) -> List[float]:
        """计算每轮的权重（基于累积梯度范数）。
        
        参考 FedRecovery 原论文：
        weight_i = ||delta_i||^2 / sum_{j<i} ||delta_j||^2
        
        这里简化为均匀权重或基于轮次的递增权重。
        """
        # 简单实现：后面的轮次权重更大（因为模型更接近最终状态）
        weights = []
        cumsum = 0.0
        for i in range(1, num_rounds + 1):
            cumsum += i
        for i in range(1, num_rounds + 1):
            weights.append(i / cumsum)
        return weights

    def _compute_gradient_residual(
        self,
        global_delta: Dict[str, torch.Tensor],
        target_delta: Dict[str, torch.Tensor],
        num_agents: int
    ) -> Dict[str, torch.Tensor]:
        """计算梯度残差。
        
        公式: grad_residual = 1/(n-1) * (global_delta - target_delta/n)
        
        这个残差代表了移除目标客户端后，其他客户端的平均贡献。
        """
        residual = {}
        n = num_agents
        for key in global_delta.keys():
            # global_delta 是所有客户端聚合后的更新
            # target_delta 是目标客户端的更新
            # 残差 = (全局更新 - 目标客户端贡献) / (n-1)
            residual[key] = (global_delta[key] - target_delta[key] / n) / (n - 1)
        return residual

    def _correction_phase_decentralized_replay(self, verbose: bool = True):
        """Replay a target-free decentralized trajectory from saved local deltas.

        The original FedRecovery baseline is server-centric.  In the DFL ring,
        each retained node has its own trajectory, and the target node only
        affects other nodes through local ring aggregation and later diffusion.
        This replay starts from round_0 retained states, applies each selected
        retained node's saved local update delta, and aggregates only on the
        post-removal selected ring.  The resulting per-node states are therefore
        a lightweight approximation of the trajectory that would have occurred
        without the target agent.
        """
        print(f"\n{'='*60}")
        print("D-FedRecovery: 去中心化历史重放校正阶段")
        print(f"{'='*60}")
        print(f"  目标客户端: {self.target_agent}")
        print(f"  参与客户端: {self.recovery_topology.surviving_agents}")
        print(f"  校正模式: decentralized_replay")
        print(f"  校正权重参数: {self.correction_weight}")
        print(f"  噪声标准差: {self.noise_std}")

        available_rounds = self.snapshot.available_rounds
        if len(available_rounds) <= 1:
            print("  警告: 没有足够的历史轮次进行校正")
            return

        selected_agents = list(self.recovery_topology.surviving_agents)
        final_round = available_rounds[-1]
        round0 = available_rounds[0]

        final_states: Dict[int, Dict[str, torch.Tensor]] = {}
        replay_states: Dict[int, Dict[str, torch.Tensor]] = {}

        sample_state = self.snapshot_loader.load_agent_state(round0, selected_agents[0])
        tracked_keys = [
            key for key in sample_state.keys()
            if self.selected_param_keys is None or key in self.selected_param_keys
        ]
        if not tracked_keys:
            raise ValueError("FedRecovery decentralized replay has no tracked LoRA keys.")

        for agent_id in selected_agents:
            final_state = self.snapshot_loader.load_agent_state(final_round, agent_id)
            final_states[agent_id] = {k: v.clone().cpu() for k, v in final_state.items()}
            initial_state = self.snapshot_loader.load_agent_state(round0, agent_id)
            replay_states[agent_id] = {
                key: initial_state[key].clone().cpu()
                for key in tracked_keys
            }

        total_delta_norms: List[float] = []
        for round_idx in range(len(available_rounds) - 1):
            round_from = available_rounds[round_idx]
            round_to = available_rounds[round_idx + 1]
            if verbose:
                print(f"\n  重放轮次 {round_from} → {round_to}")

            local_pre_states: Dict[int, Dict[str, torch.Tensor]] = {}
            for agent_id in selected_agents:
                prev_actual = self.snapshot_loader.load_agent_state(round_from, agent_id)
                pre_actual = self.snapshot_loader.load_agent_state(round_to, agent_id, pre_agg=True)
                local_pre_states[agent_id] = {}
                for key in tracked_keys:
                    local_delta = pre_actual[key].cpu() - prev_actual[key].cpu()
                    local_pre_states[agent_id][key] = replay_states[agent_id][key] + local_delta

            next_replay_states: Dict[int, Dict[str, torch.Tensor]] = {}
            for agent_id in selected_agents:
                neighbors = self.recovery_topology.get_neighbors(agent_id)
                all_ids = [agent_id] + neighbors
                weights = self.recovery_topology.get_aggregation_weights(agent_id, all_ids)
                next_state: Dict[str, torch.Tensor] = {}
                for key in tracked_keys:
                    stacked = torch.stack([local_pre_states[aid][key].cpu() for aid in all_ids])
                    coeffs = torch.tensor(weights, dtype=stacked.dtype, device=stacked.device)
                    next_state[key] = (
                        stacked * coeffs.view(-1, *([1] * (stacked.dim() - 1)))
                    ).sum(dim=0)
                next_replay_states[agent_id] = next_state

            replay_states = next_replay_states

        for agent_id in selected_agents:
            correction_norm = 0.0
            for key in tracked_keys:
                correction = final_states[agent_id][key].cpu() - replay_states[agent_id][key].cpu()
                correction_norm += (correction.float() ** 2).sum().item()
                # FedRecovery uses a correction strength to remove the target
                # contribution from the retained trajectory.  The decentralized
                # replay state is the target-free proxy; applying the configured
                # weight preserves the original FedRecovery knob instead of
                # silently fixing it to one.
                final_states[agent_id][key] = (
                    final_states[agent_id][key].cpu()
                    - float(self.correction_weight) * correction
                )

                if self.noise_std > 0:
                    noise = torch.randn_like(final_states[agent_id][key]) * self.noise_std
                    final_states[agent_id][key] += noise
            total_delta_norms.append(float(np.sqrt(correction_norm)))
            self.current_models[agent_id] = final_states[agent_id]

        mean_norm = float(np.mean(total_delta_norms)) if total_delta_norms else 0.0
        self.history["correction_phase"]["rounds_processed"] = len(available_rounds) - 1
        self.history["correction_phase"]["total_correction_norm"] = mean_norm
        self.history["correction_phase"]["mode"] = "decentralized_replay"
        self.history["correction_phase"]["per_agent_correction_norm"] = {
            str(agent_id): float(norm)
            for agent_id, norm in zip(selected_agents, total_delta_norms)
        }

        print(f"\n  平均每节点校正范数: {mean_norm:.6f}")
        print("去中心化历史重放校正阶段完成")

    def _correction_phase_residual(self, verbose: bool = True):
        """旧版残差校正：保留作为兼容路径。"""
        print(f"\n{'='*60}")
        print("D-FedRecovery: 残差校正阶段")
        print(f"{'='*60}")
        print(f"  目标客户端: {self.target_agent}")
        print(f"  校正权重: {self.correction_weight}")
        print(f"  噪声标准差: {self.noise_std}")

        available_rounds = self.snapshot.available_rounds
        num_rounds = len(available_rounds) - 1

        if num_rounds <= 0:
            print("  警告: 没有足够的历史轮次进行校正")
            return

        print(f"  历史轮次数: {num_rounds}")

        weights = self._compute_round_weights(num_rounds)

        final_round = available_rounds[-1]

        for agent_id in self.recovery_topology.surviving_agents:
            final_state = self.snapshot_loader.load_agent_state(final_round, agent_id)
            self.current_models[agent_id] = {
                k: v.clone().cpu() for k, v in final_state.items()
            }

        total_correction: Dict[str, torch.Tensor] = None
        total_correction_norm = 0.0

        for round_idx in range(num_rounds):
            round_from = available_rounds[round_idx]
            round_to = available_rounds[round_idx + 1]
            weight = weights[round_idx]

            if verbose:
                print(f"\n  处理轮次 {round_from} → {round_to} (权重: {weight:.4f})")

            ref_agent = self.recovery_topology.surviving_agents[0]
            state_from = self.snapshot_loader.load_agent_state(round_from, ref_agent)
            state_to = self.snapshot_loader.load_agent_state(round_to, ref_agent)

            global_delta = {}
            for key in state_from.keys():
                global_delta[key] = (state_to[key].cpu() - state_from[key].cpu()).float()

            try:
                target_pre_agg = self.snapshot_loader.load_agent_state(
                    round_to, self.target_agent, pre_agg=True
                )
                target_state_from = self.snapshot_loader.load_agent_state(
                    round_from, self.target_agent
                )
                target_delta = {}
                for key in state_from.keys():
                    target_delta[key] = (
                        target_pre_agg[key].cpu() - target_state_from[key].cpu()
                    ).float()
            except FileNotFoundError:
                if verbose:
                    print(f"    警告: 轮次 {round_to} 没有 pre_agg 状态，使用近似")
                target_state_to = self.snapshot_loader.load_agent_state(
                    round_to, self.target_agent
                )
                target_state_from = self.snapshot_loader.load_agent_state(
                    round_from, self.target_agent
                )
                target_delta = {}
                for key in state_from.keys():
                    target_delta[key] = (
                        target_state_to[key].cpu() - target_state_from[key].cpu()
                    ).float()

            residual = self._compute_gradient_residual(
                global_delta, target_delta, self.snapshot.num_agents
            )

            if total_correction is None:
                total_correction = {
                    key: self.correction_weight * weight * residual[key]
                    for key in residual.keys()
                    if self.selected_param_keys is None or key in self.selected_param_keys
                }
            else:
                for key in residual.keys():
                    if self.selected_param_keys is not None and key not in self.selected_param_keys:
                        continue
                    total_correction[key] += self.correction_weight * weight * residual[key]

        if total_correction is not None:
            for key, tensor in total_correction.items():
                total_correction_norm += (tensor ** 2).sum().item()
            total_correction_norm = np.sqrt(total_correction_norm)

        print(f"\n  总校正量范数: {total_correction_norm:.6f}")

        for agent_id in self.recovery_topology.surviving_agents:
            for key in self.current_models[agent_id].keys():
                if key not in total_correction:
                    continue
                self.current_models[agent_id][key] = (
                    self.current_models[agent_id][key] - total_correction[key]
                )

                if self.noise_std > 0:
                    noise = torch.randn_like(self.current_models[agent_id][key]) * self.noise_std
                    self.current_models[agent_id][key] += noise

        self.history["correction_phase"]["rounds_processed"] = num_rounds
        self.history["correction_phase"]["total_correction_norm"] = total_correction_norm
        self.history["correction_phase"]["mode"] = "residual"

        print(f"\n校正阶段完成")

    def correction_phase(self, verbose: bool = True):
        """校正阶段：基于历史轨迹计算校正量。

        流程：
        1. 遍历DFL训练的每一轮
        2. 计算全局更新 delta_global = model_{t+1} - model_t
        3. 计算目标客户端更新 delta_target = model_{t+1}_pre_agg - model_t
        4. 计算梯度残差并累积校正量
        5. 应用校正到最终模型
        """
        if self.correction_mode == "residual":
            return self._correction_phase_residual(verbose=verbose)
        return self._correction_phase_decentralized_replay(verbose=verbose)

    def recovery_phase(
        self,
        batch_size: int = 4,
        grad_accum_steps: int = 2,
        verbose: bool = True,
        show_progress: bool = True,
        local_steps: int = None
    ):
        """恢复阶段：幸存客户端继续联邦训练恢复性能。
        
        Args:
            local_steps: 每轮每个agent的训练步数
        """
        if self.recovery_rounds <= 0:
            print("\n跳过恢复阶段 (recovery_rounds=0)")
            return
        
        import gc
        
        print(f"\n{'='*60}")
        print("D-FedRecovery: 恢复阶段")
        print(f"{'='*60}")
        print(f"  恢复轮数: {self.recovery_rounds}")
        effective_local_steps = local_steps if (local_steps is not None and int(local_steps) > 0) else None
        if effective_local_steps is not None:
            print(f"  每轮本地训练步数: {effective_local_steps}")
        else:
            print(f"  每轮训练轮数: {self.recovery_epochs}")
        print(f"  参与客户端: {self.recovery_topology.surviving_agents}")
        
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
                
                # 加载当前模型状态
                self.model.set_lora_state_dict(self.current_models[agent_id])
                
                # 使用固定种子保证可复现性（与不同 k/顺序无关）
                train_seed = derive_seed(
                    self.selection_seed,
                    salt="fedrecovery_recovery",
                    round_idx=round_idx,
                    agent_id=agent_id,
                )
                if self.selected_param_keys is not None:
                    agent._freeze_unselected_params()
                loss = agent.trainer.train_local(
                    agent.local_samples,
                    self.collator,
                    local_steps=effective_local_steps,
                    epochs=self.recovery_epochs if effective_local_steps is None else None,
                    batch_size=batch_size,
                    grad_accum_steps=grad_accum_steps,
                    show_progress=show_progress,
                    progress_desc=f"Recovery train A{agent_id}",
                    progress_position=1 if show_progress else None,
                    progress_leave=True,
                    seed=train_seed,
                    # Keep training behavior consistent with the previous implementation,
                    # which reset optimizer/scheduler each time via `setup_optimizer(...)`.
                    reset_optimizer=True,
                )
                if self.selected_param_keys is not None:
                    agent._unfreeze_all_params()
                
                trained_models[agent_id] = self.model.get_lora_state_dict()
                round_losses.append(loss)
                self.history["recovery_phase"]["agent_losses"][agent_id].append(loss)
            
            avg_loss = np.mean(round_losses) if round_losses else 0.0
            if verbose:
                print(f"  平均 loss = {avg_loss:.4f}")
            if round_pbar is not None:
                round_pbar.set_postfix({"loss": f"{avg_loss:.4f}"})
            
            # 环状聚合
            aggregated = self._ring_aggregate(trained_models)
            for agent_id in self.recovery_topology.surviving_agents:
                self.current_models[agent_id] = aggregated[agent_id]
            
            self.history["recovery_phase"]["rounds"].append(round_idx + 1)
            
            del trained_models, aggregated
        
        print(f"\n恢复阶段完成")
        if round_pbar is not None:
            round_pbar.close()

    def _ring_aggregate(
        self,
        agent_models: Dict[int, Dict[str, torch.Tensor]]
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """环状聚合：每个客户端与邻居平均。"""
        new_models = {}
        
        for agent_id in self.recovery_topology.surviving_agents:
            neighbors = self.recovery_topology.get_neighbors(agent_id)
            all_ids = [agent_id] + neighbors
            
            aggregated = {}
            weights = self.recovery_topology.get_aggregation_weights(agent_id, all_ids)
            for key in agent_models[agent_id].keys():
                tensors = [agent_models[i][key].cpu() for i in all_ids]
                stacked = torch.stack(tensors)
                coeffs = torch.tensor(weights, dtype=stacked.dtype, device=stacked.device)
                aggregated[key] = (stacked * coeffs.view(-1, *([1] * (stacked.dim() - 1)))).sum(dim=0)
            
            new_models[agent_id] = aggregated
        
        return new_models

    def _get_eval_agent_ids(self) -> List[int]:
        """获取用于评估的 agent 列表。"""
        if self.eval_agent_ids is not None:
            return [aid for aid in self.eval_agent_ids 
                    if aid in self.recovery_topology.surviving_agents]
        # Default: evaluate on ALL actually selected agents (selected-k).
        return sorted(self.recovery_topology.surviving_agents)

    def set_eval_agent_ids(self, agent_ids: List[int]) -> None:
        """设置评估时使用的 agent 模型 ID 列表。"""
        self.eval_agent_ids = list(agent_ids)

    def verify_unlearning(self, agent_id: int = None, verbose: bool = True) -> UnlearningMetrics:
        """验证遗忘效果。"""
        if agent_id is None:
            agent_id = self.recovery_topology.surviving_agents[0]
        
        self.model.set_lora_state_dict(self.current_models[agent_id])
        verifier = UnlearningVerifier(self.model, self.collator, self.device)
        
        if verbose:
            print(f"\n--- Unlearning Verification (Agent {agent_id}) ---")
        
        metrics = verifier.verify_unlearning(
            self.forget_samples_eval,
            self.retain_samples_eval,
            nonmember_samples=self.nonmember_samples_eval,
            batch_size=8,
            verbose=verbose,
            show_progress=verbose
        )
        return metrics

    def run(
        self,
        batch_size: int = 4,
        grad_accum_steps: int = 2,
        save_dir: str = None,
        eval_every: int = 0,
        recovery_local_steps: int = None,
        skip_final_eval: bool = False
    ):
        """运行完整的 D-FedRecovery 流程。"""
        print(f"\n{'='*60}")
        print("D-FedRecovery: 开始遗忘流程")
        print(f"{'='*60}")
        
        # 显示节点选择信息
        print(f"\n--- Agent Selection ---")
        print(f"Selection strategy: {self.selection_strategy}")
        print(f"Selected agents for recovery: {self.recovery_topology.surviving_agents}")
        
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
        
        # 1. 校正阶段
        self.correction_phase(verbose=True)
        
        # 2. 恢复阶段（可选）
        if self.recovery_rounds > 0:
            self.recovery_phase(
                batch_size=batch_size,
                grad_accum_steps=grad_accum_steps,
                verbose=True,
                show_progress=True,
                local_steps=recovery_local_steps
            )
        
        # 3. 最终评估
        if skip_final_eval:
            print("\n跳过最终评估（skip_final_eval=True）；后门审计会单独评估保存后的模型。")
        else:
            print(f"\n{'='*60}")
            print("最终评估")
            print(f"{'='*60}")
            
            if self.task_type == "tofu":
                # TOFU 数据集评估
                final_tofu = self.evaluate_tofu(verbose=True)
                final_tofu["phase"] = "final"
                self.history["tofu_metrics"].append(final_tofu)
            else:
                # 分类任务评估（如 20newsgroups）
                # 1. MIA/遗忘指标
                final_unlearn_mean, final_unlearn_std = self._evaluate_unlearning_all(verbose=True)

                # 2. Test set 分类性能
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
        
        # 4. 保存结果
        if save_dir:
            self._save_results(save_dir)
        
        print(f"\nD-FedRecovery 完成!")

    def _save_results(self, save_dir: str):
        """保存结果。"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        def safe_value(v):
            """转换值为可序列化类型。"""
            if isinstance(v, np.ndarray):
                return v.tolist()
            elif isinstance(v, (np.floating, np.integer)):
                return float(v) if not np.isnan(v) else None
            elif isinstance(v, dict):
                return {str(k): safe_value(vv) for k, vv in v.items()}
            elif isinstance(v, list):
                return [safe_value(item) for item in v]
            return v

        # Add final_stats (test-set metrics over eval_agent_ids, typically selected-k agents).
        if self.last_test_per_agent_results:
            per_agent = self.last_test_per_agent_results
            final_stats = build_final_stats(
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
                        final_stats[k] = last_unlearn.get(k)
                    std_k = f"{k}_std"
                    if std_k in last_unlearn:
                        final_stats[std_k] = last_unlearn.get(std_k)

            self.history["final_stats"] = final_stats
        
        # 保存历史
        history_serializable = safe_value(self.history)
        
        with open(save_path / "history.json", "w") as f:
            json.dump(history_serializable, f, indent=2)
        
        # 保存最终模型（可选；大规模 sweep 可关闭以节省磁盘）
        if self.save_lora_states:
            try:
                for agent_id in self.recovery_topology.surviving_agents:
                    agent_dir = save_path / f"agent_{agent_id}"
                    agent_dir.mkdir(exist_ok=True)
                    torch.save(
                        self.current_models[agent_id],
                        agent_dir / "lora_state.pt"
                    )
            except Exception as e:  # noqa: BLE001
                print(f"[WARN] Failed to save final LoRA states: {e}")
        
        print(f"Results saved to {save_path}")
