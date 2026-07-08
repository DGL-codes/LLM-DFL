"""DFL Trainer orchestrating multi-agent training.

快照索引说明：
- round_0 = 初始未训练的LoRA模型 (训练开始前)
- round_1 = 第1轮全局训练后的模型
- round_i (i≥1) = 第i轮全局训练后的模型

对于 G 个全局轮次的训练，共保存 G+1 个快照 (round_0 到 round_G)。
例如：G=10 时，保存 round_0 到 round_10，共11个快照。
"""
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm

from .agent import DFLAgent, RingTopology
from ..models.lora_model import LoRAModelWrapper
from ..models.multi_agent_eval import evaluate_lora_states
from ..data.base import Sample
from ..data.collator import LLMCollator
from ..data.partitioner import PartitionInfo

# TOFU评估支持
try:
    from ..dfu.tofu_evaluator import TOFUEvaluator
    TOFU_EVALUATOR_AVAILABLE = True
except ImportError:
    TOFU_EVALUATOR_AVAILABLE = False


class DFLTrainer:
    """Orchestrates DFL training across multiple agents."""

    def __init__(
        self,
        num_agents: int,
        model: LoRAModelWrapper,
        collator: LLMCollator,
        partition: PartitionInfo,
        all_samples: List[Sample],
        val_samples: List[Sample],
        test_samples: List[Sample],
        max_eval_samples: Optional[int] = None,
        tofu_local_dir: Optional[str] = None,
        tofu_eval_max_samples: int = 50,
        label_names: Optional[List[str]] = None,
        task_type: str = "classification",
        lr: float = 1e-4,
        device: str = "cuda",
        optimizer_name: str = "adamw",
        seed: int = 42
    ):
        self.num_agents = num_agents
        self.base_model = model
        self.collator = collator
        self.partition = partition
        self.val_samples = val_samples
        self.test_samples = test_samples
        self.max_eval_samples = max_eval_samples
        self.tofu_local_dir = tofu_local_dir
        self.tofu_eval_max_samples = int(tofu_eval_max_samples)
        self.label_names = label_names
        self.task_type = task_type
        self.device = device
        self.seed = seed

        # Initialize topology
        self.topology = RingTopology(num_agents)

        # Initialize agents with their local data
        self.agents: List[DFLAgent] = []
        for i in range(num_agents):
            indices = partition.agent_indices[i]
            local_samples = [all_samples[idx] for idx in indices]

            agent = DFLAgent(
                agent_id=i,
                local_samples=local_samples,
                model=model,
                collator=collator,
                lr=lr,
                device=device,
                optimizer_name=optimizer_name
            )
            self.agents.append(agent)

        # Training history with metrics for each agent
        self.history = {
            "global_rounds": [],
            "agent_losses": {i: [] for i in range(num_agents)},
            "agent_metrics": {i: [] for i in range(num_agents)},
            "avg_metrics": [],
            "avg_metrics_std": [],
            "val_losses": [],
            "tofu_metrics": []  # TOFU特定指标
        }
        
        # 外部forget样本（用于retrain场景）
        self._external_forget_samples = None
        # 外部retain样本（用于retrain/对齐场景）
        self._external_retain_samples = None
        # 默认评估目标：agent1 作为模型，agent0 作为 forget，retain=除 forget 外的所有客户端
        self.eval_agent_id = 1
        # 默认评估仅使用参与客户端中的前3个（按ID排序）
        self.eval_agent_ids: Optional[List[int]] = None
        self.eval_forget_agent_id = 0
        
        # 初始化TOFU评估器（如果是TOFU任务）
        self.tofu_evaluator = None
        if self.task_type == "tofu" and TOFU_EVALUATOR_AVAILABLE:
            self._init_tofu_evaluator()
    
    def train(
        self,
        global_rounds: int,
        local_steps: int,
        batch_size: int = 4,
        grad_accum_steps: int = 4,
        save_dir: Optional[str] = None,
        eval_every: int = 1,
        save_only_final: bool = False,
        optimizer_schedule: Optional[object] = None,
        show_progress: bool = True
    ):
        """Run DFL training.

        快照保存说明：
        - round_0: 训练开始前的初始LoRA状态
        - round_1 到 round_{global_rounds}: 每轮训练后的模型状态
        - 共保存 global_rounds + 1 个快照
        """
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            self.partition.save(str(save_path / "partition.json"))

            # Create curves directory
            curves_dir = save_path / "curves"
            curves_dir.mkdir(exist_ok=True)

            # Generate partition visualization
            self._plot_partition_distribution(save_path)

            if not save_only_final:
                # 保存初始状态为 round_0（训练开始前）
                print("Saving initial LoRA state as round_0...")
                self._save_round(save_path, round_idx=0)

        # 初始评估（Round 0 - 训练开始前），允许跳过
        if eval_every and eval_every > 0:
            print("\n=== Initial Evaluation (Round 0) ===")
            print("Evaluating agents on test set before training...")
            
            # For TOFU, use TOFU-specific evaluation
            if self.task_type == "tofu":
                print("TOFU task: Using TOFU-specific evaluation metrics")
                self.history["global_rounds"].append(0)
                # TOFU评估
                if self.tofu_evaluator is not None:
                    tofu_results = self._evaluate_tofu(
                        verbose=True, 
                        external_forget_samples=self._external_forget_samples,
                        external_retain_samples=self._external_retain_samples
                    )
                    tofu_results["round"] = 0
                    self.history["tofu_metrics"].append(tofu_results)
            else:
                initial_metrics = self._evaluate_all_agents()
                initial_avg_metrics, initial_std_metrics = self._compute_metrics_with_std(initial_metrics)
                self.history["avg_metrics"].append(initial_avg_metrics)
                self.history["avg_metrics_std"].append(initial_std_metrics)
                self.history["global_rounds"].append(0)

                # Print initial metrics
                if self.task_type == "classification":
                    print(
                        f"Initial Average Accuracy: "
                        f"{self._format_mean_std(initial_avg_metrics.get('accuracy', 0), initial_std_metrics.get('accuracy', 0))}, "
                        f"Macro-F1: {self._format_mean_std(initial_avg_metrics.get('macro_f1', 0), initial_std_metrics.get('macro_f1', 0))}"
                    )
                else:
                    print(
                        f"Initial Average Token-F1: "
                        f"{self._format_mean_std(initial_avg_metrics.get('token_f1', 0), initial_std_metrics.get('token_f1', 0))}, "
                        f"Exact Match: {self._format_mean_std(initial_avg_metrics.get('exact_match', 0), initial_std_metrics.get('exact_match', 0))}"
                    )

            # Update curves with initial evaluation
            if save_dir and not save_only_final:
                self._plot_curves(curves_dir)
                if self.task_type == "tofu":
                    self._plot_tofu_curves(curves_dir)
        else:
            print("\nSkipping evaluations (eval_every<=0)")
            # 仍记录起始轮次
            self.history["global_rounds"].append(0)
        last_eval_round = 0 if (eval_every and eval_every > 0) else None

        round_pbar = None
        if show_progress and global_rounds > 0:
            round_pbar = tqdm(
                range(global_rounds),
                desc="Global rounds",
                dynamic_ncols=True,
                leave=True
            )
            round_iter = round_pbar
        else:
            round_iter = range(global_rounds)

        for round_idx in round_iter:
            print(f"\n=== Global Round {round_idx + 1}/{global_rounds} ===")

            # Optional optimizer schedule per global round
            if optimizer_schedule is not None:
                if callable(optimizer_schedule):
                    round_optimizer = optimizer_schedule(round_idx)
                else:
                    if round_idx < len(optimizer_schedule):
                        round_optimizer = optimizer_schedule[round_idx]
                    else:
                        round_optimizer = optimizer_schedule[-1]
                if round_optimizer:
                    print(f"Using optimizer: {round_optimizer}")
                    for agent in self.agents:
                        agent.set_optimizer_name(round_optimizer)

            # Phase 1: Local training
            round_losses = {}
            for agent in self.agents:
                train_seed = self.seed + round_idx * 100 + agent.agent_id
                loss = agent.local_train(
                    steps=local_steps,
                    batch_size=batch_size,
                    grad_accum_steps=grad_accum_steps,
                    show_progress=show_progress,
                    progress_desc=f"Local train A{agent.agent_id}",
                    progress_position=1 if show_progress else None,
                    seed=train_seed
                )
                round_losses[agent.agent_id] = loss
                self.history["agent_losses"][agent.agent_id].append(loss)

            print(f"Agent losses: {[f'{l:.4f}' if not np.isnan(l) else 'nan' for l in round_losses.values()]}")

            # Phase 2: Ring aggregation
            print("Aggregating parameters...")
            # 保存训练后、聚合前的参数 (pre-aggregation: round_t-c_u_i)
            pre_agg_params = {
                agent.agent_id: agent.get_lora_params()
                for agent in self.agents
            }

            new_params = {}
            for agent in self.agents:
                new_params[agent.agent_id] = self.topology.aggregate(
                    agent.agent_id, pre_agg_params
                )

            # Update agents with aggregated parameters
            for agent in self.agents:
                agent.set_lora_params(new_params[agent.agent_id])

            # Evaluation on test set for each agent（可跳过）
            do_eval = eval_every and eval_every > 0 and ((round_idx + 1) % eval_every == 0)
            if do_eval:
                if self.task_type == "tofu":
                    print("TOFU task: Using TOFU-specific evaluation metrics")
                    # TOFU评估
                    if self.tofu_evaluator is not None:
                        tofu_results = self._evaluate_tofu(
                            verbose=True,
                            external_forget_samples=self._external_forget_samples,
                            external_retain_samples=self._external_retain_samples
                        )
                        tofu_results["round"] = round_idx + 1
                        self.history["tofu_metrics"].append(tofu_results)
                        last_eval_round = round_idx + 1
                else:
                    print("Evaluating agents on test set...")
                    round_metrics = self._evaluate_all_agents()

                    # Compute average metrics
                    avg_metrics, std_metrics = self._compute_metrics_with_std(round_metrics)
                    self.history["avg_metrics"].append(avg_metrics)
                    self.history["avg_metrics_std"].append(std_metrics)
                    last_eval_round = round_idx + 1

                    # Print metrics
                    if self.task_type == "classification":
                        print(
                            f"Average Accuracy: "
                            f"{self._format_mean_std(avg_metrics.get('accuracy', 0), std_metrics.get('accuracy', 0))}, "
                            f"Macro-F1: {self._format_mean_std(avg_metrics.get('macro_f1', 0), std_metrics.get('macro_f1', 0))}"
                        )
                    else:
                        print(
                            f"Average Token-F1: "
                            f"{self._format_mean_std(avg_metrics.get('token_f1', 0), std_metrics.get('token_f1', 0))}, "
                            f"Exact Match: {self._format_mean_std(avg_metrics.get('exact_match', 0), std_metrics.get('exact_match', 0))}"
                        )

            self.history["global_rounds"].append(round_idx + 1)

            # Save checkpoints and update curves
            # round_idx=0 表示第1轮训练，保存为 round_1
            # 同时保存 pre-aggregation 参数 (训练后、聚合前)
            if save_dir and not save_only_final:
                self._save_round(save_path, round_idx + 1, pre_agg_params=pre_agg_params)
                if eval_every and eval_every > 0:
                    self._plot_curves(curves_dir)
                    if self.task_type == "tofu":
                        self._plot_tofu_curves(curves_dir)
                # 每轮都保存history，便于实时监控
                self._save_history(save_path)

        if round_pbar is not None:
            round_pbar.close()

        if save_dir and not save_only_final:
            self._save_history(save_path)

        # 如果未在最后一轮评估且需要最终评估（例如 eval_every<=0），执行一次末轮评估
        if last_eval_round != global_rounds:
            print(f"\n=== Final Evaluation (Round {global_rounds}) ===")
            if self.task_type == "tofu":
                if self.tofu_evaluator is not None:
                    tofu_results = self._evaluate_tofu(
                        verbose=True,
                        external_forget_samples=self._external_forget_samples,
                        external_retain_samples=self._external_retain_samples
                    )
                    tofu_results["round"] = global_rounds
                    self.history["tofu_metrics"].append(tofu_results)
            else:
                round_metrics = self._evaluate_all_agents()
                avg_metrics, std_metrics = self._compute_metrics_with_std(round_metrics)
                self.history["avg_metrics"].append(avg_metrics)
                self.history["avg_metrics_std"].append(std_metrics)
                if self.task_type == "classification":
                    print(
                        f"Average Accuracy: "
                        f"{self._format_mean_std(avg_metrics.get('accuracy', 0), std_metrics.get('accuracy', 0))}, "
                        f"Macro-F1: {self._format_mean_std(avg_metrics.get('macro_f1', 0), std_metrics.get('macro_f1', 0))}"
                    )
                else:
                    print(
                        f"Average Token-F1: "
                        f"{self._format_mean_std(avg_metrics.get('token_f1', 0), std_metrics.get('token_f1', 0))}, "
                        f"Exact Match: {self._format_mean_std(avg_metrics.get('exact_match', 0), std_metrics.get('exact_match', 0))}"
                    )
            # 保存/绘图
            if save_dir and not save_only_final:
                self._save_history(save_path)
                self._plot_curves(curves_dir)
                if self.task_type == "tofu":
                    self._plot_tofu_curves(curves_dir)

        if save_dir and save_only_final:
            # 只保存最终模型
            self._save_round(save_path, global_rounds)
            self._save_history(save_path)
            self._plot_curves(curves_dir)
            if self.task_type == "tofu":
                self._plot_tofu_curves(curves_dir)
    
    def _evaluate_all_agents(self) -> Dict[int, Dict]:
        """Evaluate all agents on test set."""
        eval_agent_ids = self._get_eval_agent_ids()
        lora_states: Dict[int, Dict] = {}
        for aid in eval_agent_ids:
            agent = self.agents[aid]
            if agent.lora_state is None:
                agent.lora_state = agent.get_lora_params()
            lora_states[int(aid)] = agent.lora_state

        per_agent, _summary = evaluate_lora_states(
            model=self.base_model,
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
        """Compute average metrics across all agents."""
        if not round_metrics:
            return {}

        keys = list(round_metrics[0].keys())
        avg = {}
        for key in keys:
            values = [m[key] for m in round_metrics.values() if not np.isnan(m.get(key, 0))]
            avg[key] = np.mean(values) if values else 0.0
        return avg

    def _compute_metrics_with_std(self, round_metrics: Dict[int, Dict]) -> Tuple[Dict, Dict]:
        """Compute average and std metrics across all agents."""
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

    def _plot_partition_distribution(self, save_path: Path):
        """Plot and save partition distribution visualization."""
        # Collect label distribution per agent
        labels_per_agent = {}
        all_labels = set()

        for agent in self.agents:
            label_counts = {}
            for sample in agent.local_samples:
                label = sample.label if sample.label is not None else 0
                label_counts[label] = label_counts.get(label, 0) + 1
                all_labels.add(label)
            labels_per_agent[agent.agent_id] = label_counts

        all_labels = sorted(all_labels)
        num_labels = len(all_labels)

        # Create stacked bar chart
        fig, ax = plt.subplots(figsize=(max(10, self.num_agents * 0.8), 6))

        x = np.arange(self.num_agents)
        width = 0.8
        bottom = np.zeros(self.num_agents)

        colors = plt.cm.tab20(np.linspace(0, 1, num_labels))

        for i, label in enumerate(all_labels):
            heights = [labels_per_agent[agent_id].get(label, 0) for agent_id in range(self.num_agents)]
            label_name = self.label_names[label] if self.label_names and label < len(self.label_names) else f"Class {label}"
            ax.bar(x, heights, width, bottom=bottom, label=label_name[:15], color=colors[i])
            bottom += heights

        ax.set_xlabel("Agent ID")
        ax.set_ylabel("Number of Samples")
        ax.set_title(f"Data Partition Distribution (α={self.partition.alpha})")
        ax.set_xticks(x)
        ax.set_xticklabels([f"A{i}" for i in range(self.num_agents)])

        # Legend outside plot if too many labels
        if num_labels <= 10:
            ax.legend(loc='upper right', fontsize='small')
        else:
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='x-small', ncol=1)

        plt.tight_layout()
        plt.savefig(save_path / "partition_distribution.png", dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_curves(self, curves_dir: Path):
        """Plot and save training curves."""
        rounds = self.history["global_rounds"]
        if len(rounds) < 1:
            return

        # 1. Loss curves (only plot if we have training losses)
        # Note: Round 0 has no training loss (initial evaluation only)
        if any(len(losses) > 0 for losses in self.history["agent_losses"].values()):
            fig, ax = plt.subplots(figsize=(10, 6))

            # Get the rounds that have corresponding loss data
            # Loss data starts from round 1 (after first training)
            max_loss_len = max(len(losses) for losses in self.history["agent_losses"].values())
            loss_rounds = [r for r in rounds if r > 0][:max_loss_len]

            for agent_id, losses in self.history["agent_losses"].items():
                if len(losses) > 0:
                    valid_losses = [l if not np.isnan(l) else None for l in losses[:len(loss_rounds)]]
                    ax.plot(loss_rounds, valid_losses, alpha=0.5, label=f"Agent {agent_id}")

            # Average loss
            avg_losses = []
            for i in range(len(loss_rounds)):
                agent_vals = [self.history["agent_losses"][aid][i] for aid in range(self.num_agents)
                             if i < len(self.history["agent_losses"][aid]) and not np.isnan(self.history["agent_losses"][aid][i])]
                avg_losses.append(np.mean(agent_vals) if agent_vals else np.nan)
            ax.plot(loss_rounds, avg_losses, 'k-', linewidth=2, label="Average")

            ax.set_xlabel("Global Round")
            ax.set_ylabel("Training Loss")
            ax.set_title("Training Loss Curves")
            ax.legend(loc='upper right', fontsize='small', ncol=2)
            plt.tight_layout()
            plt.savefig(curves_dir / "loss_curves.png", dpi=150)
            plt.close()

        # 2. Metrics curves (skip for TOFU task)
        if self.task_type == "tofu":
            pass  # TOFU uses specialized metrics in DFU
        elif self.task_type == "classification":
            self._plot_metric_curve(curves_dir, "accuracy", "Accuracy")
            self._plot_metric_curve(curves_dir, "macro_f1", "Macro-F1")
        else:
            self._plot_metric_curve(curves_dir, "token_f1", "Token F1")
            self._plot_metric_curve(curves_dir, "exact_match", "Exact Match")

    def _plot_metric_curve(self, curves_dir: Path, metric_key: str, metric_name: str):
        """Plot a single metric curve for all agents."""
        rounds = self.history["global_rounds"]
        if not rounds:
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        for agent_id, metrics_list in self.history["agent_metrics"].items():
            if not metrics_list:
                continue
            values = [m.get(metric_key, 0) for m in metrics_list]
            plot_rounds = rounds[:len(values)]
            ax.plot(plot_rounds, values, alpha=0.5, label=f"Agent {agent_id}")

        # Average
        avg_values = [m.get(metric_key, 0) for m in self.history["avg_metrics"]]
        if avg_values:
            ax.plot(rounds[:len(avg_values)], avg_values, 'k-', linewidth=2, label="Average")

        ax.set_xlabel("Global Round")
        ax.set_ylabel(metric_name)
        ax.set_title(f"{metric_name} Curves")
        ax.legend(loc='lower right', fontsize='small', ncol=2)
        plt.tight_layout()
        plt.savefig(curves_dir / f"{metric_key}_curves.png", dpi=150)
        plt.close()

    def _save_round(self, save_path: Path, round_idx: int, pre_agg_params: Dict[int, Dict] = None):
        """Save all agent checkpoints for this round.

        Args:
            save_path: Base save directory
            round_idx: Round index
            pre_agg_params: Pre-aggregation parameters for each agent (optional)
                           If provided, saves both lora_state.pt (post-agg) and 
                           lora_state_pre_agg.pt (pre-agg)
        
        使用 get_lora_params() 确保即使是初始状态也能正确保存。
        """
        round_dir = save_path / f"round_{round_idx}"
        round_dir.mkdir(exist_ok=True)

        for agent in self.agents:
            agent_dir = round_dir / f"agent_{agent.agent_id}"
            agent_dir.mkdir(exist_ok=True)
            # 保存聚合后模型 (post-aggregation)
            lora_state = agent.get_lora_params()
            torch.save(lora_state, agent_dir / "lora_state.pt")
            
            # 如果提供了 pre-aggregation 参数，也保存
            if pre_agg_params and agent.agent_id in pre_agg_params:
                torch.save(pre_agg_params[agent.agent_id], agent_dir / "lora_state_pre_agg.pt")

    def _save_history(self, save_path: Path):
        """Save training history."""
        def safe_float(v):
            if isinstance(v, (int, float)):
                return None if np.isnan(v) else float(v)
            return v

        history_json = {
            "global_rounds": self.history["global_rounds"],
            "agent_losses": {
                str(k): [safe_float(v) for v in vals]
                for k, vals in self.history["agent_losses"].items()
            },
            "agent_metrics": {
                str(k): [{mk: safe_float(mv) for mk, mv in m.items()} for m in metrics]
                for k, metrics in self.history["agent_metrics"].items()
            },
            "avg_metrics": [{k: safe_float(v) for k, v in m.items()} for m in self.history["avg_metrics"]],
            "avg_metrics_std": [{k: safe_float(v) for k, v in m.items()} for m in self.history["avg_metrics_std"]],
            "val_losses": [safe_float(v) for v in self.history["val_losses"]],
            "tofu_metrics": [
                {k: safe_float(v) for k, v in m.items()}
                for m in self.history.get("tofu_metrics", [])
            ]
        }
        with open(save_path / "history.json", "w") as f:
            json.dump(history_json, f, indent=2)
    
    # ============================================================
    # TOFU Dataset Evaluation Support
    # ============================================================
    
    def _init_tofu_evaluator(self):
        """初始化TOFU评估器。"""
        if not TOFU_EVALUATOR_AVAILABLE:
            print("Warning: TOFU evaluator not available")
            return
        
        self.tofu_evaluator = TOFUEvaluator(
            model=self.base_model,
            tokenizer=self.base_model.tokenizer,
            collator=self.collator,
            device=self.device,
            tofu_local_dir=self.tofu_local_dir,
        )
        
        # 准备评估样本（使用test_samples的一部分）
        # 假设test_samples已经按label分组
        self._tofu_eval_max_samples = self.tofu_eval_max_samples
        self._tofu_eval_samples = self.test_samples[:min(self._tofu_eval_max_samples, len(self.test_samples))]
        print(f"TOFU evaluator initialized with {len(self._tofu_eval_samples)} eval samples")
    
    def _evaluate_tofu(
        self, 
        agent_id: int = None, 
        verbose: bool = True,
        external_forget_samples: List[Sample] = None,
        external_retain_samples: List[Sample] = None
    ) -> Dict:
        """使用TOFU指标评估模型（DFL训练期间的简化版本）。
        
        DFL训练期间只评估Retain和Forget集，不评估Real Authors/World。
        
        Args:
            agent_id: 要评估的agent ID（用于加载模型）
            verbose: 是否显示详细信息
            external_forget_samples: 外部提供的forget样本（用于retrain场景）
                如果提供，则使用这些样本作为forget集，而不是从agent获取
            external_retain_samples: 外部提供的retain样本（用于retrain/对齐场景）
                如果提供，则使用这些样本作为retain集，而不是从agent获取
            
        Returns:
            TOFU指标字典
        """
        if self.tofu_evaluator is None:
            return {}

        # 确定评估模型的agent集合
        if agent_id is None:
            agent_ids = self._get_eval_agent_ids()
        else:
            agent_ids = [agent_id if agent_id < len(self.agents) else 0]
        
        # 确定forget样本来源
        if external_forget_samples is not None:
            # Retrain场景：使用外部提供的被排除客户端数据
            forget_samples = external_forget_samples
            print(f"  Using external forget samples: {len(forget_samples)} samples")
        else:
            # DFL场景：使用指定的forget agent本地数据（默认agent0）
            forget_agent_id = self.eval_forget_agent_id
            if forget_agent_id >= len(self.agents):
                forget_agent_id = agent_id
            forget_samples = self.agents[forget_agent_id].local_samples
        
        if external_retain_samples is not None:
            retain_samples = external_retain_samples
            print(f"  Using external retain samples: {len(retain_samples)} samples")
        else:
            # DFL场景：使用除 forget agent 外的所有客户端数据
            retain_samples = []
            for ag in self.agents:
                if ag.agent_id == forget_agent_id:
                    continue
                retain_samples.extend(ag.local_samples)
        
        # 逐个agent评估，并聚合为均值±标准差
        per_agent_results: Dict[int, Dict] = {}
        per_agent_verbose = verbose and len(agent_ids) == 1
        for aid in agent_ids:
            self.base_model.set_lora_state_dict(self.agents[aid].lora_state)
            results = self.tofu_evaluator.evaluate_all(
                retain_samples=retain_samples,
                forget_samples=forget_samples,
                compute_generation=True,
                verbose=per_agent_verbose,
                print_summary=per_agent_verbose,
                show_progress=verbose,
                max_samples=getattr(self, "_tofu_eval_max_samples", 50)
            )
            per_agent_results[aid] = results.to_dict()

        aggregated = self._aggregate_tofu_metrics(per_agent_results)

        if verbose and len(agent_ids) > 1:
            self._print_tofu_summary(aggregated, len(agent_ids))

        return aggregated

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
    
    def set_external_forget_samples(self, samples: List[Sample]):
        """设置外部forget样本（用于retrain场景）。
        
        Args:
            samples: 被排除客户端的原始样本列表
        """
        self._external_forget_samples = samples
        print(f"Set external forget samples: {len(samples)} samples")

    def set_external_retain_samples(self, samples: List[Sample]):
        """设置外部retain样本（用于retrain/对齐场景）。"""
        self._external_retain_samples = samples
        print(f"Set external retain samples: {len(samples)} samples")

    def set_eval_agent_id(self, agent_id: int):
        """设置评估时加载的agent模型ID。"""
        self.eval_agent_id = agent_id
        self.eval_agent_ids = [agent_id]

    def set_eval_agent_ids(self, agent_ids: List[int]):
        """设置评估时使用的agent模型ID列表。"""
        self.eval_agent_ids = list(agent_ids)

    def set_eval_forget_agent_id(self, agent_id: int):
        """设置评估时的forget agent ID。"""
        self.eval_forget_agent_id = agent_id

    def set_eval_retain_agent_id(self, agent_id: int):
        """设置评估时的retain agent ID。"""
        self.eval_retain_agent_id = agent_id

    def _get_eval_agent_ids(self) -> List[int]:
        """获取用于评估的agent列表（用于加速评估）。"""
        if self.eval_agent_ids is None:
            all_ids = sorted([ag.agent_id for ag in self.agents])
            return all_ids[:3]
        candidate = [aid for aid in self.eval_agent_ids if 0 <= aid < len(self.agents)]
        if not candidate:
            all_ids = sorted([ag.agent_id for ag in self.agents])
            return all_ids[:3]
        return sorted(candidate)
    
    def _plot_tofu_curves(self, curves_dir: Path):
        """绘制TOFU指标曲线。"""
        if "tofu_metrics" not in self.history or not self.history["tofu_metrics"]:
            return
        
        tofu_metrics = self.history["tofu_metrics"]
        rounds = [m.get("round", i) for i, m in enumerate(tofu_metrics)]
        
        # 绘制Model Utility曲线
        fig, ax = plt.subplots(figsize=(10, 6))
        
        mu_values = [m.get("model_utility", 0) for m in tofu_metrics]
        ax.plot(rounds, mu_values, 'b-o', linewidth=2, markersize=6, label="Model Utility")
        
        ax.set_xlabel("Global Round")
        ax.set_ylabel("Model Utility")
        ax.set_title("TOFU Model Utility During DFL Training")
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if mu_values:
            ax.annotate(f'Final: {mu_values[-1]:.4f}',
                       xy=(rounds[-1], mu_values[-1]),
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(curves_dir / "tofu_model_utility.png", dpi=150)
        plt.close()
        
        # 绘制详细指标曲线
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        metric_pairs = [
            ("retain_rouge1_recall", "Retain ROUGE-1"),
            ("retain_rougeL_recall", "Retain ROUGE-L"),
            ("retain_probability", "Retain Probability"),
            ("retain_truth_ratio", "Retain Truth Ratio")
        ]
        
        for ax, (metric_key, metric_name) in zip(axes.flat, metric_pairs):
            values = [m.get(metric_key, 0) for m in tofu_metrics]
            ax.plot(rounds, values, 'g-o', linewidth=2, markersize=6)
            ax.set_xlabel("Global Round")
            ax.set_ylabel(metric_name)
            ax.set_title(metric_name)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(curves_dir / "tofu_detailed_metrics.png", dpi=150)
        plt.close()
