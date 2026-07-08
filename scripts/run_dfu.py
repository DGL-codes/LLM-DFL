"""DFU (Decentralized Federated Unlearning) script for a single DFL snapshot."""
import argparse
import json
import secrets
import torch
from pathlib import Path
from datetime import datetime
from typing import List, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.datasets import (
    NewsGroupsDataset, DBpediaDataset, YahooAnswersDataset, YahooSubsetDataset,
    AlpacaGPT4Dataset, FinGPTSentimentDataset,
    MedicalFlashcardsDataset, CodeAlpacaDataset, TOFUDataset
)
from src.utils.gpu_guard import guard_gpu_or_raise
from src.data.backdoor_wrapper import BackdoorPoisonSpec, poison_samples_by_indices
from src.data.partitioner import PartitionInfo
from src.data.collator import LLMCollator
from src.data.base import Sample
from src.models.lora_model import LoRAModelWrapper
from src.dfu.snapshot_loader import SnapshotLoader
from src.dfu.trainer import DFUTrainer
from src.dfu.d_oblivionis import DOblivionisTrainer
from src.dfu.d_fedosd import DFedOSDTrainer
from src.dfu.d_fedrecovery import DFedRecoveryTrainer

import run_retrain as retrain_module

DATASET_MAP = {
    "20newsgroups": NewsGroupsDataset,
    "dbpedia": DBpediaDataset,
    "yahoo": YahooAnswersDataset,
    "yahoo_subset": YahooSubsetDataset,
    "alpaca": AlpacaGPT4Dataset,
    "fingpt": FinGPTSentimentDataset,
    "medical": MedicalFlashcardsDataset,
    "code": CodeAlpacaDataset,
    "tofu": TOFUDataset,
}

TOFU_FORGET_TO_RETAIN = {
    "forget01": "retain99",
    "forget05": "retain95",
    "forget10": "retain90",
}


def _load_json_or_jsonl(path: Path):
    # Detect JSON array vs JSONL.
    with open(path, "r", encoding="utf-8") as f:
        head = ""
        while True:
            ch = f.read(1)
            if not ch:
                break
            if not ch.isspace():
                head = ch
                break
    if head == "[":
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def load_tofu_perturbed_as_samples(*, tofu_local_dir: str, config_name: str) -> List[Sample]:
    """Load TOFU *perturbed* json/jsonl as Sample objects (with perturb fields)."""
    path = Path(tofu_local_dir) / f"{config_name}.json"
    data = _load_json_or_jsonl(path)
    out: List[Sample] = []
    for item in data:
        q = str(item.get("question", "")).strip()
        a = str(item.get("answer", "")).strip()
        if not q or not a:
            continue
        paraphrased = str(item.get("paraphrased_answer", a))
        pert = item.get("perturbed_answer", item.get("perturbed_answers", []))
        if isinstance(pert, str):
            pert_list = [pert]
        else:
            pert_list = list(pert or [])
        out.append(
            Sample(
                instruction=TOFUDataset.INSTRUCTION,
                input_text=f"Question: {q}",
                output_text=a,
                label=None,
                paraphrased_answer=paraphrased,
                perturbed_answers=pert_list,
            )
        )
    return out

FIXED_EVAL_AGENT_IDS = {
    # Deprecated: kept for backward compatibility / reference.
    # Previously used for consistent evaluation across sweeps/ablations (always evaluate on the same 3 agents).
    "20newsgroups": [1, 8, 9],
    "yahoo_subset": [1, 4, 9],
    "tofu": [2, 5, 9],
}


def _replay_backdoor_poisoning_if_needed(
    *,
    dfl_config: dict,
    train_samples: List[Sample],
    partition: PartitionInfo,
    label_names: Optional[List[str]],
    task_type: str,
    requested_target_agent: int,
    fallback_seed: int,
) -> dict:
    """Reconstruct the exact poisoned training samples used by a backdoor DFL run.

    DFL backdoor audits poison the target agent before decentralized training.
    DFU reloads the raw dataset, so the target agent's forget samples must be
    poisoned again to match the original DFL training distribution.
    """
    trigger = str(dfl_config.get("backdoor_trigger") or "").strip()
    try:
        rate = float(dfl_config.get("backdoor_rate") or 0.0)
    except Exception:
        rate = 0.0

    info = {
        "enabled": False,
        "reason": "no_backdoor_config",
        "trigger": trigger or None,
        "poison_rate": rate,
    }
    if not trigger or rate <= 0.0:
        return info

    if task_type != "classification":
        info["reason"] = f"unsupported_task_type:{task_type}"
        return info
    if not label_names:
        raise ValueError("Backdoor DFU replay requires label_names for classification datasets.")

    poison_agent = int(dfl_config.get("backdoor_target_agent", requested_target_agent))
    if poison_agent not in partition.agent_indices:
        raise ValueError(
            f"Backdoor DFU replay target agent {poison_agent} not found in partition. "
            f"Available agents: {sorted(partition.agent_indices)}"
        )

    seed_raw = dfl_config.get("backdoor_seed")
    poison_seed = int(seed_raw) if seed_raw is not None else int(dfl_config.get("seed", fallback_seed))
    spec = BackdoorPoisonSpec(
        trigger=trigger,
        poison_rate=rate,
        target_label=int(dfl_config.get("backdoor_target_label", 0)),
        position=str(dfl_config.get("backdoor_position") or "prefix"),
        seed=poison_seed,
    )
    poisoned = poison_samples_by_indices(
        train_samples,
        partition.agent_indices[poison_agent],
        label_names=label_names,
        spec=spec,
    )
    info.update(
        {
            "enabled": True,
            "reason": "replayed_from_dfl_config",
            "poison_agent": poison_agent,
            "requested_target_agent": int(requested_target_agent),
            "target_label": int(spec.target_label),
            "position": spec.position,
            "seed": poison_seed,
            "poisoned": int(poisoned),
            "agent_sample_count": int(len(partition.agent_indices[poison_agent])),
            "target_matches_forget_agent": poison_agent == int(requested_target_agent),
        }
    )
    print(
        "Backdoor DFU replay enabled: "
        f"agent={poison_agent}, rate={spec.poison_rate}, target_label={spec.target_label}, "
        f"poisoned={poisoned}/{len(partition.agent_indices[poison_agent])}, "
        f"target_matches_forget_agent={poison_agent == int(requested_target_agent)}"
    )
    return info


def main(args):
    import time
    import random
    import numpy as np
    start_time = time.time()

    # Enforce physical GPU 2/3 only (args.gpu is LOGICAL within CUDA_VISIBLE_DEVICES)
    visible_physical = guard_gpu_or_raise(gpu=args.gpu)
    
    # 设置全局随机种子（未指定时自动生成）
    seed = args.seed
    if seed is None or seed < 0:
        seed = secrets.randbits(32)
        print(f"Random seed: {seed}")
    args.seed = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # 支持指定GPU
    if args.gpu is not None:
        device = f"cuda:{args.gpu}"
        torch.cuda.set_device(args.gpu)
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"CUDA_VISIBLE_DEVICES (physical): {visible_physical}")

    removed_agents = None
    if getattr(args, "removed_agents", None):
        removed_agents = sorted({
            int(x.strip())
            for x in str(args.removed_agents).split(",")
            if x.strip()
        })
        if int(args.target_agent) not in removed_agents:
            removed_agents.append(int(args.target_agent))
            removed_agents = sorted(set(removed_agents))
        print(f"Sequential/cumulative removal set: {removed_agents}")

    # Load DFL snapshot
    print(f"\nLoading DFL snapshot from: {args.dfl_snapshot}")
    snapshot_loader = SnapshotLoader(args.dfl_snapshot)
    snapshot = snapshot_loader.snapshot
    
    print(f"  Dataset: {snapshot.dataset}")
    print(f"  Num agents: {snapshot.num_agents}")
    print(f"  Global rounds: {snapshot.global_rounds}")
    print(f"  Available rounds: {snapshot.available_rounds}")

    # 分支：retrain / rapidretrain（直接复用 run_retrain 逻辑）
    if args.dfu_algorithm in {"retrain", "rapidretrain"}:
        algo_name = args.dfu_algorithm
        print(f"\nDFU algorithm = {algo_name}")
        output_dir = args.output_dir
        if algo_name == "retrain" and output_dir == "dfu_checkpoints":
            output_dir = "retrain_checkpoints"
        retrain_args = argparse.Namespace(
            dfl_checkpoint=args.dfl_snapshot,
            output_dir=output_dir,
            target_agent=args.target_agent,
            eval_every=args.eval_every,
            max_train_samples=args.max_train_samples,
            max_eval_samples=args.max_eval_samples,
            gpu=args.gpu,
            optimizer="sgd" if algo_name == "rapidretrain" else "adamw",
            save_all_rounds=False,
            method_name=algo_name,
            jo=args.jo,
            t0=args.t0,
        )
        retrain_module.main(retrain_args)
        return
    
    # Create DFU output directory
    # Structure: dfu_checkpoints/{dataset}/K{total_agents}/G{rounds}_L{steps}/alpha{alpha}/strategy_{strategy}_ratio{ratio}/{dfl_snapshot_name}/dfu_{timestamp}
    dfl_snapshot_name = snapshot_loader.get_snapshot_name()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    dfl_config = snapshot_loader.config
    snapshot_local_steps = int(dfl_config.get("local_steps") or 5)

    # Align per-round local training budgets across methods:
    # - DFL uses `local_steps` from the snapshot config
    # - For D-Oblivionis we default to the same `local_steps` (instead of full-epoch training)
    if args.dfu_algorithm == "d-oblivionis":
        if args.unlearn_local_steps is None:
            args.unlearn_local_steps = snapshot_local_steps
        if args.propagation_local_steps is None:
            args.propagation_local_steps = snapshot_local_steps

    # Build strategy directory name
    # 格式: strategy_{strategy}[_count{X}][_lora{Y}_eps{Z}]
    if args.selection_strategy == "full":
        strategy_dir = "strategy_full"
    elif args.selection_strategy == "random":
        if args.selection_count is not None:
            strategy_dir = f"strategy_random_count{args.selection_count}"
        else:
            strategy_dir = f"strategy_random_ratio{args.selection_ratio}"
    elif args.selection_strategy == "ours":
        if args.selection_count is not None:
            strategy_dir = f"strategy_ours_count{args.selection_count}"
        else:
            strategy_dir = f"strategy_ours_ratio{args.selection_ratio}"
    elif args.selection_strategy == "tdb":
        if args.selection_count is not None:
            strategy_dir = f"strategy_tdb_count{args.selection_count}"
        elif args.selection_ratio is not None:
            strategy_dir = f"strategy_tdb_ratio{args.selection_ratio}"
        else:
            strategy_dir = "strategy_tdb"
    else:
        strategy_dir = f"strategy_{args.selection_strategy}"

    # 如果启用了LoRA参数选择，添加到目录名
    # 区分random和ours的LoRA选择策略
    if args.enable_param_selection:
        lora_strategy = "random" if args.param_random_selection else "ours"
        mode = (args.param_selection_mode or "epsilon").lower()
        if mode == "top_ratio":
            strategy_dir += f"_lora{args.param_selection_ratio}_topratio_{lora_strategy}"
        else:
            strategy_dir += f"_lora{args.param_selection_ratio}_eps{args.param_epsilon_W}_{lora_strategy}"

    output_dir = (
        Path(args.output_dir) /
        dfl_config.get("dataset", "unknown") /
        args.dfu_algorithm /
        strategy_dir /
        f"K{snapshot.num_agents}" /
        f"G{dfl_config.get('global_rounds', 0)}_L{dfl_config.get('local_steps', 0)}" /
        f"alpha{dfl_config.get('alpha', 0.5)}" /
        dfl_snapshot_name /
        f"dfu_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nDFU output directory: {output_dir}")
    
    # Save DFU config
    optimizer_name = "adahessian" if args.dfu_algorithm == "rapidretrain" else "adamw"
    print(f"\nDFU algorithm = {args.dfu_algorithm} (optimizer={optimizer_name})")

    dfu_config = {
        "dfl_snapshot": args.dfl_snapshot,
        "target_agent": args.target_agent,
        "removed_agents": removed_agents if removed_agents is not None else [args.target_agent],
        "calibration_steps": args.calibration_steps,
        "calibration_interval": args.calibration_interval,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "max_train_samples": args.max_train_samples,
        "max_eval_samples": args.max_eval_samples,
        "dfu_algorithm": args.dfu_algorithm,
        "selection_strategy": args.selection_strategy,
        "selection_ratio": args.selection_ratio,
        "selection_count": args.selection_count,
        "selection_epsilon": args.selection_epsilon,
        "tdb_sketch_dim": args.tdb_sketch_dim,
        "tdb_max_intervals": args.tdb_max_intervals,
        "tdb_round_stride": args.tdb_round_stride,
        "tdb_alpha_u": args.tdb_alpha_u,
        "tdb_alpha_p": args.tdb_alpha_p,
        "tdb_alpha_q": args.tdb_alpha_q,
        "tdb_epsilon_u": args.tdb_epsilon_u,
        "tdb_epsilon_p": args.tdb_epsilon_p,
        "tdb_tau_q": args.tdb_tau_q,
        "tdb_exposure_rho": args.tdb_exposure_rho,
        "tdb_use_target_similarity": args.tdb_use_target_similarity,
        "tdb_time_limit": args.tdb_time_limit,
        "tdb_aggregation_scope": args.tdb_aggregation_scope,
        "seed": seed,
        "timestamp": timestamp,
        "optimizer": optimizer_name,
        # LoRA参数选择配置
        "enable_param_selection": args.enable_param_selection,
        "param_selection_ratio": args.param_selection_ratio,
        "param_epsilon_W": args.param_epsilon_W,
        "param_random_selection": args.param_random_selection,
        "param_relative_sensitivity": args.param_relative_sensitivity,
        "param_sensitivity_alpha": args.param_sensitivity_alpha,
        "param_selection_mode": args.param_selection_mode,
        "param_sensitivity_cache": args.param_sensitivity_cache,
        # Artifact saving (disable in large sweeps to avoid huge disk usage)
        "save_lora_states": args.save_lora_states,
    }

    # Record algorithm-specific hyperparameters for reproducibility.
    if args.dfu_algorithm == "d-oblivionis":
        dfu_config.update({
            "unlearn_rounds": args.unlearn_rounds,
            "unlearn_epochs": args.unlearn_epochs,
            "unlearn_local_steps": args.unlearn_local_steps,
            "unlearn_lr": args.unlearn_lr,
            "propagation_rounds": args.propagation_rounds,
            "propagation_epochs": args.propagation_epochs,
            "propagation_local_steps": args.propagation_local_steps,
            "propagation_lr": args.propagation_lr,
        })
    elif args.dfu_algorithm == "d-fedosd":
        dfu_config.update({
            "unlearn_rounds": args.unlearn_rounds,
            "unlearn_lr": args.unlearn_lr,
            "recovery_rounds": args.recovery_rounds,
            "recovery_epochs": args.recovery_epochs,
            "recovery_local_steps": args.recovery_local_steps,
            "recovery_lr": args.recovery_lr,
            "retain_grad_samples": args.retain_grad_samples,
            "forget_grad_samples": args.forget_grad_samples,
            "fedosd_retain_subspace_mode": args.fedosd_retain_subspace_mode,
            "fedosd_orthogonal_update_norm": args.fedosd_orthogonal_update_norm,
            "fedosd_projection_strength": args.fedosd_projection_strength,
            "fedosd_forget_loss": args.fedosd_forget_loss,
            "cache_retain_grads": args.cache_retain_grads,
        })
    elif args.dfu_algorithm == "d-fedrecovery":
        dfu_config.update({
            "correction_weight": args.correction_weight,
            "noise_std": args.noise_std,
            "fedrecovery_correction_mode": args.fedrecovery_correction_mode,
            "recovery_rounds": args.recovery_rounds,
            "recovery_epochs": args.recovery_epochs,
            "recovery_local_steps": args.recovery_local_steps,
            "recovery_lr": args.recovery_lr,
        })
    with open(output_dir / "dfu_config.json", "w") as f:
        json.dump(dfu_config, f, indent=2)
    
    # Load dataset
    dataset_name = snapshot.dataset
    print(f"\nLoading {dataset_name} dataset...")
    
    DatasetClass = DATASET_MAP.get(dataset_name, NewsGroupsDataset)
    
    # TOFU数据集需要特殊处理
    tofu_fed = False
    tofu_local_dir: Optional[str] = None
    tofu_forget_rate: Optional[str] = None
    if dataset_name == "tofu":
        tofu_fed = bool(dfl_config.get("tofu_fed", False))

        if tofu_fed:
            tofu_local_dir = str(dfl_config.get("tofu_local_dir") or "").strip()
            tofu_forget_rate = str(dfl_config.get("forget_rate") or "").strip()
            tofu_retain_split = str(
                dfl_config.get("retain_split") or TOFU_FORGET_TO_RETAIN.get(tofu_forget_rate or "")
            ).strip()

            if not tofu_local_dir:
                raise ValueError("TOFU fed snapshot missing config field: tofu_local_dir")
            if not tofu_forget_rate:
                raise ValueError("TOFU fed snapshot missing config field: forget_rate")
            if not tofu_retain_split:
                raise ValueError(
                    f"TOFU fed snapshot missing/invalid retain_split for forget_rate={tofu_forget_rate!r}"
                )

            num_authors = int(dfl_config.get("num_authors", args.num_authors))
            group_size = int(dfl_config.get("group_size", 2))
            qa_per_author = int(dfl_config.get("qa_per_author", 20))

            if args.max_train_samples is not None:
                print(
                    "[WARN] --max_train_samples is ignored for tofu_fed snapshots "
                    "(partition indices require the full forget+retain ordering)."
                )

            forget_ds = DatasetClass(
                split="train",
                max_samples=None,
                num_authors=num_authors,
                group_size=group_size,
                tofu_local_dir=tofu_local_dir,
                tofu_split=tofu_forget_rate,
                qa_per_author=qa_per_author,
            )
            retain_ds = DatasetClass(
                split="train",
                max_samples=None,
                num_authors=num_authors,
                group_size=group_size,
                tofu_local_dir=tofu_local_dir,
                tofu_split=tofu_retain_split,
                qa_per_author=qa_per_author,
            )
            train_ds = retain_ds  # keep for label_names, etc.
            train_samples = list(forget_ds.samples) + list(retain_ds.samples)
            val_samples = []
            test_samples = []
            task_type = "tofu"

            print(
                f"TOFU fed: forget={tofu_forget_rate} ({len(forget_ds.samples)}), "
                f"retain={tofu_retain_split} ({len(retain_ds.samples)}), total={len(train_samples)}"
            )
        else:
            tofu_fed = False
            num_authors = int(dfl_config.get("num_authors", args.num_authors))
            group_size = int(dfl_config.get("group_size", 2))
            train_ds = DatasetClass(
                split="train",
                max_samples=args.max_train_samples,
                num_authors=num_authors,
                group_size=group_size
            )
            # TOFU遗忘任务：全部数据用于训练，不需要val/test（评估由TOFU evaluator处理）
            train_samples = train_ds.samples
            val_samples = []
            test_samples = []
            task_type = "tofu"
    else:
        train_ds = DatasetClass(split="train", max_samples=args.max_train_samples)
        test_ds = DatasetClass(split="test", max_samples=args.max_eval_samples)
        task_type = DatasetClass.TASK_TYPE
        # Split train into train/val (consistent with DFL)
        val_size = int(len(train_ds) * 0.1)
        train_samples = train_ds.samples[val_size:]
        val_samples = train_ds.samples[:val_size]
        test_samples = test_ds.samples
    
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")
    
    # Load partition from DFL snapshot
    partition_data = snapshot_loader.load_partition_info()
    if partition_data:
        partition = PartitionInfo(
            dataset_name=partition_data["dataset_name"],
            num_agents=partition_data["num_agents"],
            alpha=partition_data.get("alpha"),
            seed=partition_data["seed"],
            partition_type=partition_data["partition_type"],
            agent_indices={int(k): v for k, v in partition_data["agent_indices"].items()},
            agent_weights=partition_data.get("agent_weights")
        )
    else:
        raise ValueError("Partition info not found in DFL snapshot")

    backdoor_replay_info = _replay_backdoor_poisoning_if_needed(
        dfl_config=dfl_config,
        train_samples=train_samples,
        partition=partition,
        label_names=getattr(train_ds, "label_names", None),
        task_type=task_type,
        requested_target_agent=args.target_agent,
        fallback_seed=seed,
    )
    dfu_config["backdoor_forget_replay"] = backdoor_replay_info
    
    # Initialize model
    print("\nInitializing model...")
    model = LoRAModelWrapper(
        lora_r=dfl_config.get("lora_r", 8),
        lora_alpha=dfl_config.get("lora_alpha", 16),
        device=device
    )
    model.load_base_model()
    model.init_lora()
    
    collator = LLMCollator(model.tokenizer, max_length=dfl_config.get("max_length", 512))
    
    # Initialize DFU trainer
    print("\nInitializing DFU trainer...")
    
    # 根据算法选择不同的 Trainer
    if args.dfu_algorithm == "d-oblivionis":
        print(f"DFU algorithm = d-oblivionis")
        dfu_trainer = DOblivionisTrainer(
            snapshot_loader=snapshot_loader,
            model=model,
            collator=collator,
            all_samples=train_samples,
            val_samples=val_samples,
            test_samples=test_samples,
            partition=partition,
            target_agent=args.target_agent,
            # D-Oblivionis 特有参数
            unlearn_rounds=args.unlearn_rounds,
            unlearn_epochs=args.unlearn_epochs,
            unlearn_lr=args.unlearn_lr,
            propagation_rounds=args.propagation_rounds,
            propagation_epochs=args.propagation_epochs,
            propagation_lr=args.propagation_lr,
            # 通用参数
            lr=args.lr,
            label_names=getattr(train_ds, 'label_names', None),
            task_type=task_type,
            device=device,
            max_eval_samples=args.max_eval_samples,
            mia_nonmember_source=args.mia_nonmember_source,
            tofu_local_dir=tofu_local_dir,
            # 节点选择参数
            selection_strategy=args.selection_strategy,
            selection_ratio=args.selection_ratio,
            selection_count=args.selection_count,
            selection_seed=seed,
            selection_epsilon=args.selection_epsilon,
            tdb_sketch_dim=args.tdb_sketch_dim,
            tdb_max_intervals=args.tdb_max_intervals,
            tdb_round_stride=args.tdb_round_stride,
            tdb_alpha_u=args.tdb_alpha_u,
            tdb_alpha_p=args.tdb_alpha_p,
            tdb_alpha_q=args.tdb_alpha_q,
            tdb_epsilon_u=args.tdb_epsilon_u,
            tdb_epsilon_p=args.tdb_epsilon_p,
            tdb_tau_q=args.tdb_tau_q,
            tdb_exposure_rho=args.tdb_exposure_rho,
            tdb_use_target_similarity=args.tdb_use_target_similarity,
            tdb_time_limit=args.tdb_time_limit,
            tdb_aggregation_scope=args.tdb_aggregation_scope,
            # LoRA参数选择参数
            enable_param_selection=args.enable_param_selection,
            param_selection_ratio=args.param_selection_ratio,
            param_epsilon_W=args.param_epsilon_W,
            param_random_selection=args.param_random_selection,
            param_relative_sensitivity=args.param_relative_sensitivity,
            param_sensitivity_alpha=args.param_sensitivity_alpha,
            param_selection_mode=args.param_selection_mode,
            param_sensitivity_cache=args.param_sensitivity_cache,
            retrain_tr_path=args.retrain_tr_path,
            optimizer_name="adamw",
            save_lora_states=args.save_lora_states,
        )
    elif args.dfu_algorithm == "d-fedosd":
        print(f"DFU algorithm = d-fedosd")
        dfu_trainer = DFedOSDTrainer(
            snapshot_loader=snapshot_loader,
            model=model,
            collator=collator,
            all_samples=train_samples,
            val_samples=val_samples,
            test_samples=test_samples,
            partition=partition,
            target_agent=args.target_agent,
            # D-FedOSD 特有参数
            unlearn_rounds=args.unlearn_rounds,
            unlearn_lr=args.unlearn_lr,
            recovery_rounds=args.recovery_rounds,
            recovery_epochs=args.recovery_epochs,
            recovery_lr=args.recovery_lr,
            retain_subspace_mode=args.fedosd_retain_subspace_mode,
            orthogonal_update_norm=args.fedosd_orthogonal_update_norm,
            projection_strength=args.fedosd_projection_strength,
            forget_loss=args.fedosd_forget_loss,
            # 通用参数
            lr=args.lr,
            label_names=getattr(train_ds, 'label_names', None),
            task_type=task_type,
            device=device,
            max_eval_samples=args.max_eval_samples,
            mia_nonmember_source=args.mia_nonmember_source,
            tofu_local_dir=tofu_local_dir,
            # 节点选择参数
            selection_strategy=args.selection_strategy,
            selection_ratio=args.selection_ratio,
            selection_count=args.selection_count,
            selection_seed=seed,
            selection_epsilon=args.selection_epsilon,
            tdb_sketch_dim=args.tdb_sketch_dim,
            tdb_max_intervals=args.tdb_max_intervals,
            tdb_round_stride=args.tdb_round_stride,
            tdb_alpha_u=args.tdb_alpha_u,
            tdb_alpha_p=args.tdb_alpha_p,
            tdb_alpha_q=args.tdb_alpha_q,
            tdb_epsilon_u=args.tdb_epsilon_u,
            tdb_epsilon_p=args.tdb_epsilon_p,
            tdb_tau_q=args.tdb_tau_q,
            tdb_exposure_rho=args.tdb_exposure_rho,
            tdb_use_target_similarity=args.tdb_use_target_similarity,
            tdb_time_limit=args.tdb_time_limit,
            tdb_aggregation_scope=args.tdb_aggregation_scope,
            # LoRA参数选择参数
            enable_param_selection=args.enable_param_selection,
            param_selection_ratio=args.param_selection_ratio,
            param_epsilon_W=args.param_epsilon_W,
            param_random_selection=args.param_random_selection,
            param_relative_sensitivity=args.param_relative_sensitivity,
            param_sensitivity_alpha=args.param_sensitivity_alpha,
            param_selection_mode=args.param_selection_mode,
            param_sensitivity_cache=args.param_sensitivity_cache,
            retrain_tr_path=args.retrain_tr_path,
            optimizer_name="adamw",
            save_lora_states=args.save_lora_states,
        )
    elif args.dfu_algorithm == "d-fedrecovery":
        print(f"DFU algorithm = d-fedrecovery")
        dfu_trainer = DFedRecoveryTrainer(
            snapshot_loader=snapshot_loader,
            model=model,
            collator=collator,
            all_samples=train_samples,
            val_samples=val_samples,
            test_samples=test_samples,
            partition=partition,
            target_agent=args.target_agent,
            # D-FedRecovery 特有参数
            correction_weight=args.correction_weight,
            noise_std=args.noise_std,
            correction_mode=args.fedrecovery_correction_mode,
            recovery_rounds=args.recovery_rounds,
            recovery_epochs=args.recovery_epochs,
            recovery_lr=args.recovery_lr,
            # 通用参数
            lr=args.lr,
            label_names=getattr(train_ds, 'label_names', None),
            task_type=task_type,
            device=device,
            max_eval_samples=args.max_eval_samples,
            mia_nonmember_source=args.mia_nonmember_source,
            tofu_local_dir=tofu_local_dir,
            # 节点选择参数
            selection_strategy=args.selection_strategy,
            selection_ratio=args.selection_ratio,
            selection_count=args.selection_count,
            selection_seed=seed,
            selection_epsilon=args.selection_epsilon,
            tdb_sketch_dim=args.tdb_sketch_dim,
            tdb_max_intervals=args.tdb_max_intervals,
            tdb_round_stride=args.tdb_round_stride,
            tdb_alpha_u=args.tdb_alpha_u,
            tdb_alpha_p=args.tdb_alpha_p,
            tdb_alpha_q=args.tdb_alpha_q,
            tdb_epsilon_u=args.tdb_epsilon_u,
            tdb_epsilon_p=args.tdb_epsilon_p,
            tdb_tau_q=args.tdb_tau_q,
            tdb_exposure_rho=args.tdb_exposure_rho,
            tdb_use_target_similarity=args.tdb_use_target_similarity,
            tdb_time_limit=args.tdb_time_limit,
            tdb_aggregation_scope=args.tdb_aggregation_scope,
            # LoRA参数选择参数
            enable_param_selection=args.enable_param_selection,
            param_selection_ratio=args.param_selection_ratio,
            param_epsilon_W=args.param_epsilon_W,
            param_random_selection=args.param_random_selection,
            param_relative_sensitivity=args.param_relative_sensitivity,
            param_sensitivity_alpha=args.param_sensitivity_alpha,
            param_selection_mode=args.param_selection_mode,
            param_sensitivity_cache=args.param_sensitivity_cache,
            retrain_tr_path=args.retrain_tr_path,
            optimizer_name="adamw",
            save_lora_states=args.save_lora_states,
        )
    else:
        # d-federaser 或 rapidretrain
        dfu_trainer = DFUTrainer(
            snapshot_loader=snapshot_loader,
            model=model,
            collator=collator,
            all_samples=train_samples,
            val_samples=val_samples,
            test_samples=test_samples,
            partition=partition,
            target_agent=args.target_agent,
            removed_agents=removed_agents,
            calibration_steps=args.calibration_steps,
            calibration_interval=args.calibration_interval,
            lr=args.lr,
            label_names=getattr(train_ds, 'label_names', None),
            task_type=task_type,
            device=device,
            max_eval_samples=args.max_eval_samples,
            mia_nonmember_source=args.mia_nonmember_source,
            tofu_local_dir=tofu_local_dir,
            # 节点选择参数
            selection_strategy=args.selection_strategy,
            selection_ratio=args.selection_ratio,
            selection_count=args.selection_count,
            selection_seed=seed,
            selection_epsilon=args.selection_epsilon,
            tdb_sketch_dim=args.tdb_sketch_dim,
            tdb_max_intervals=args.tdb_max_intervals,
            tdb_round_stride=args.tdb_round_stride,
            tdb_alpha_u=args.tdb_alpha_u,
            tdb_alpha_p=args.tdb_alpha_p,
            tdb_alpha_q=args.tdb_alpha_q,
            tdb_epsilon_u=args.tdb_epsilon_u,
            tdb_epsilon_p=args.tdb_epsilon_p,
            tdb_tau_q=args.tdb_tau_q,
            tdb_exposure_rho=args.tdb_exposure_rho,
            tdb_use_target_similarity=args.tdb_use_target_similarity,
            tdb_time_limit=args.tdb_time_limit,
            tdb_aggregation_scope=args.tdb_aggregation_scope,
            # LoRA参数选择参数
            enable_param_selection=args.enable_param_selection,
            param_selection_ratio=args.param_selection_ratio,
            param_epsilon_W=args.param_epsilon_W,
            param_random_selection=args.param_random_selection,
            param_relative_sensitivity=args.param_relative_sensitivity,
            param_sensitivity_alpha=args.param_sensitivity_alpha,
            param_selection_mode=args.param_selection_mode,
            param_sensitivity_cache=args.param_sensitivity_cache,
            # Retrain TR 分布（用于计算 Forget Quality）
            retrain_tr_path=args.retrain_tr_path,
            optimizer_name=optimizer_name,
            save_lora_states=args.save_lora_states,
        )

    # For tofu_fed snapshots, force official perturbed eval splits for retain/forget.
    if dataset_name == "tofu" and tofu_fed and tofu_local_dir and tofu_forget_rate:
        try:
            retain_eval_samples = load_tofu_perturbed_as_samples(
                tofu_local_dir=tofu_local_dir,
                config_name="retain_perturbed",
            )
            forget_eval_samples = load_tofu_perturbed_as_samples(
                tofu_local_dir=tofu_local_dir,
                config_name=f"{tofu_forget_rate}_perturbed",
            )
            if hasattr(dfu_trainer, "set_tofu_external_eval_samples"):
                dfu_trainer.set_tofu_external_eval_samples(
                    retain_samples=retain_eval_samples,
                    forget_samples=forget_eval_samples,
                )
                print(
                    f"TOFU official eval splits set: retain_perturbed({len(retain_eval_samples)}), "
                    f"{tofu_forget_rate}_perturbed({len(forget_eval_samples)})"
                )
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] Could not set TOFU official eval splits: {e}")

    # Evaluate on the actually selected agents (selected-k), so results reflect deployed DFU participants.
    #
    # NOTE: TOFU evaluation is *very* expensive (generation + multiple forward passes). By default we
    # only evaluate a single representative agent to keep TOFU-fed grids tractable.
    if hasattr(dfu_trainer, "set_eval_agent_ids") and hasattr(dfu_trainer, "selection_result"):
        selected = list(getattr(dfu_trainer.selection_result, "selected_agents", []) or [])
        selected = [aid for aid in selected if aid != args.target_agent]
        if selected:
            eval_ids = selected
            if dataset_name == "tofu":
                eval_ids = [min(selected)]
            dfu_trainer.set_eval_agent_ids(eval_ids)
            print(f"Eval agent IDs: {eval_ids} (selected={selected})")
    
    # Update config with actual selection results
    # 不同 Trainer 的拓扑属性名不同
    selection_weights = {
        str(k): float(v) for k, v in dfu_trainer.selection_result.weights.items()
    } if dfu_trainer.selection_result.weights else None
    selection_diagnostics = (
        json.loads(json.dumps(dfu_trainer.selection_result.diagnostics, default=str))
        if dfu_trainer.selection_result.diagnostics else None
    )
    if args.dfu_algorithm in ["d-oblivionis", "d-fedosd", "d-fedrecovery"]:
        topology = dfu_trainer.propagation_topology if args.dfu_algorithm == "d-oblivionis" else dfu_trainer.recovery_topology
        dfu_config.update({
            "total_agents": snapshot.num_agents,
            "surviving_agents_count": snapshot.num_agents - 1,
            "selected_agents_count": len(dfu_trainer.selection_result.selected_agents),
            "selected_agents": dfu_trainer.selection_result.selected_agents,
            "selection_scores": {
                str(k): float(v) for k, v in dfu_trainer.selection_result.scores.items()
            } if dfu_trainer.selection_result.scores else None,
            "selection_weights": selection_weights,
            "selection_diagnostics": selection_diagnostics,
            "distribution_error": float(dfu_trainer.selection_result.distribution_error)
                if dfu_trainer.selection_result.distribution_error is not None else None
        })
    else:
        dfu_config.update({
            "total_agents": snapshot.num_agents,
            "surviving_agents_count": len(dfu_trainer.topology.all_surviving_agents),
            "selected_agents_count": len(dfu_trainer.selection_result.selected_agents),
            "selected_agents": dfu_trainer.selection_result.selected_agents,
            "selection_scores": {
                str(k): float(v) for k, v in dfu_trainer.selection_result.scores.items()
            } if dfu_trainer.selection_result.scores else None,
            "selection_weights": selection_weights,
            "selection_diagnostics": selection_diagnostics,
            "distribution_error": float(dfu_trainer.selection_result.distribution_error)
                if dfu_trainer.selection_result.distribution_error is not None else None
        })

    # Add LoRA parameter selection results if enabled
    if dfu_trainer.param_selection_result is not None:
        dfu_config.update({
            "param_selection_result": {
                "selected_modules": dfu_trainer.param_selection_result.selected_modules,
                "total_modules": len(dfu_trainer.param_selection_result.all_modules),
                "selection_ratio": dfu_trainer.param_selection_result.selection_ratio,
                "total_sensitivity": dfu_trainer.param_selection_result.total_sensitivity,
                "covered_sensitivity": dfu_trainer.param_selection_result.covered_sensitivity,
            }
        })

    with open(output_dir / "dfu_config.json", "w") as f:
        json.dump(dfu_config, f, indent=2)

    # Run DFU
    print("\nStarting DFU...")
    if args.dfu_algorithm == "d-fedosd":
        dfu_trainer.run(
            batch_size=args.batch_size,
            grad_accum_steps=args.grad_accum_steps,
            save_dir=str(output_dir),
            eval_every=args.eval_every,
            retain_grad_samples=args.retain_grad_samples,
            forget_grad_samples=args.forget_grad_samples,
            cache_retain_grads=args.cache_retain_grads,
            recovery_local_steps=args.recovery_local_steps,
            skip_final_eval=args.skip_final_eval
        )
    elif args.dfu_algorithm == "d-oblivionis":
        dfu_trainer.run(
            batch_size=args.batch_size,
            grad_accum_steps=args.grad_accum_steps,
            save_dir=str(output_dir),
            eval_every=args.eval_every,
            propagation_local_steps=args.propagation_local_steps,
            unlearn_local_steps=args.unlearn_local_steps,
            skip_final_eval=args.skip_final_eval
        )
    elif args.dfu_algorithm == "d-fedrecovery":
        dfu_trainer.run(
            batch_size=args.batch_size,
            grad_accum_steps=args.grad_accum_steps,
            save_dir=str(output_dir),
            eval_every=args.eval_every,
            recovery_local_steps=args.recovery_local_steps,
            skip_final_eval=args.skip_final_eval
        )
    else:
        # D-FedEraser
        dfu_trainer.run(
            batch_size=args.batch_size,
            grad_accum_steps=args.grad_accum_steps,
            save_dir=str(output_dir),
            eval_every=args.eval_every
        )

    print(f"\nDFU completed! Results saved to {output_dir}")
    print(f"  Total agents: {snapshot.num_agents}")
    if args.dfu_algorithm in ["d-oblivionis", "d-fedosd", "d-fedrecovery"]:
        print(f"  Surviving agents (after removing target): {snapshot.num_agents - 1}")
    else:
        print(f"  Surviving agents (after removing target): {len(dfu_trainer.topology.all_surviving_agents)}")
    print(f"  Selected agents for DFU: {len(dfu_trainer.selection_result.selected_agents)}")
    if dfu_trainer.param_selection_result is not None:
        print(f"  Selected LoRA modules: {len(dfu_trainer.param_selection_result.selected_modules)} / "
              f"{len(dfu_trainer.param_selection_result.all_modules)}")
        selected_module_scores = {
            module: float(dfu_trainer.param_selection_result.sensitivities.get(module, 0.0))
            for module in dfu_trainer.param_selection_result.selected_modules
        }
        top_sensitivity_modules = [
            {"module": module, "score": float(score)}
            for module, score in sorted(
                dfu_trainer.param_selection_result.sensitivities.items(),
                key=lambda item: (-float(item[1]), item[0]),
            )[:20]
        ]
        dfu_config["param_selection_result"] = {
            "selected_modules": dfu_trainer.param_selection_result.selected_modules,
            "selected_module_scores": selected_module_scores,
            "total_modules": len(dfu_trainer.param_selection_result.all_modules),
            "selection_ratio": dfu_trainer.param_selection_result.selection_ratio,
            "total_sensitivity": float(dfu_trainer.param_selection_result.total_sensitivity),
            "covered_sensitivity": float(dfu_trainer.param_selection_result.covered_sensitivity),
            "covered_sensitivity_ratio": (
                float(dfu_trainer.param_selection_result.covered_sensitivity)
                / float(dfu_trainer.param_selection_result.total_sensitivity)
                if float(dfu_trainer.param_selection_result.total_sensitivity) > 0
                else 0.0
            ),
            "top_sensitivity_modules": top_sensitivity_modules,
        }
        with open(output_dir / "dfu_config.json", "w") as f:
            json.dump(dfu_config, f, indent=2)

    # 记录实验结束时的时间和GPU内存使用
    elapsed_time = time.time() - start_time
    elapsed_min = elapsed_time / 60
    print(f"\n{'='*50}")
    print(f"实验完成统计:")
    print(f"  总运行时间: {elapsed_time:.2f}秒 ({elapsed_min:.2f}分钟)")
    if torch.cuda.is_available():
        gpu_id = args.gpu if args.gpu is not None else 0
        allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
        max_allocated = torch.cuda.max_memory_allocated(gpu_id) / 1024**3
        reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
        print(f"  GPU {gpu_id} 内存使用:")
        print(f"    当前分配: {allocated:.2f} GB")
        print(f"    峰值分配: {max_allocated:.2f} GB")
        print(f"    总预留: {reserved:.2f} GB")
    print(f"{'='*50}\n")

    # 实验结束后清理显存，避免影响下一个实验
    del dfu_trainer
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    print("GPU memory cleared.")

    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DFU on a single DFL snapshot")
    
    # Required arguments
    parser.add_argument("--dfl_snapshot", type=str, required=True,
                       help="Path to DFL snapshot directory")
    parser.add_argument("--dfu_algorithm", type=str, default="d-federaser",
                       choices=["d-federaser", "retrain", "rapidretrain", "d-oblivionis", "d-fedosd", "d-fedrecovery"],
                       help="Unlearning algorithm: d-federaser (default), retrain, rapidretrain, d-oblivionis, d-fedosd, or d-fedrecovery")
    
    # DFU parameters
    parser.add_argument("--output_dir", type=str, default="dfu_checkpoints",
                       help="Base output directory for DFU results")
    parser.add_argument(
        "--no_save_lora_states",
        action="store_false",
        dest="save_lora_states",
        default=True,
        help="Do not save LoRA state .pt files (history.json is still saved); useful for large sweeps.",
    )
    parser.add_argument("--target_agent", type=int, default=0,
                       help="Agent ID to forget (default: 0)")
    parser.add_argument(
        "--removed_agents",
        type=str,
        default=None,
        help=(
            "Comma-separated cumulative removed agents for sequential-withdrawal probes, "
            "for example '0,1,2'. The current --target_agent is added if missing. "
            "This is currently used by the FedEraser/DFUTrainer path."
        ),
    )
    parser.add_argument("--calibration_steps", type=int, default=3,
                       help="Local calibration training steps (E_cali)")
    parser.add_argument("--calibration_interval", type=int, default=2,
                       help="Calibration interval (how many DFL rounds per step, default=2 for sparse)")
    
    # Training parameters
    parser.add_argument("--lr", type=float, default=1e-3,
                       help="Learning rate for calibration")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for calibration")
    parser.add_argument("--grad_accum_steps", type=int, default=2,
                       help="Gradient accumulation steps")
    parser.add_argument("--eval_every", type=int, default=0,
                       help="Evaluate every N calibration steps (<=0 to skip middle evals; final eval always runs)")
    
    # Data limits
    parser.add_argument("--max_train_samples", type=int, default=None,
                       help="Max training samples")
    parser.add_argument("--max_eval_samples", type=int, default=100,
                       help="Max evaluation samples (default: 100)")
    parser.add_argument("--skip_final_eval", action="store_true",
                       help="Skip trainer-internal final verification and save final DFU states directly; useful for external audit diagnostics.")
    parser.add_argument(
        "--mia_nonmember_source",
        type=str,
        default="test",
        choices=["test", "val"],
        help="Non-member pool for MIA audit: 'test' (default) or held-out 'val' split.",
    )
    
    # TOFU dataset specific parameters
    parser.add_argument("--num_authors", type=int, default=20,
                       help="Number of authors to use for TOFU dataset (default: 20)")

    # Agent selection parameters
    parser.add_argument("--selection_strategy", type=str, default="full",
                       choices=["full", "random", "ours", "tdb"],
                       help="Agent selection strategy")
    parser.add_argument("--selection_ratio", type=float, default=None,
                       help="Ratio of agents to select (for random/ours strategies, used if count not set)")
    parser.add_argument("--selection_count", type=int, default=None,
                       help="Exact number of agents to select (takes priority over ratio)")
    parser.add_argument("--selection_epsilon", type=float, default=0.1,
                       help="Distribution error threshold for ours strategy")
    parser.add_argument("--tdb_sketch_dim", type=int, default=64,
                       help="TDB-AS sketch dimension per retained interval")
    parser.add_argument("--tdb_max_intervals", type=int, default=3,
                       help="Maximum retained intervals used for TDB-AS sketches")
    parser.add_argument("--tdb_round_stride", type=int, default=1,
                       help="Stride over DFL retained intervals for TDB-AS sketches")
    parser.add_argument("--tdb_alpha_u", type=float, default=1.0,
                       help="TDB-AS trajectory matching weight")
    parser.add_argument("--tdb_alpha_p", type=float, default=1.0,
                       help="TDB-AS label balance weight")
    parser.add_argument("--tdb_alpha_q", type=float, default=0.1,
                       help="TDB-AS target exposure reward weight")
    parser.add_argument("--tdb_epsilon_u", type=float, default=None,
                       help="Optional trajectory L1 threshold for TDB-AS min-cost mode")
    parser.add_argument("--tdb_epsilon_p", type=float, default=None,
                       help="Optional label L1 threshold for TDB-AS min-cost mode")
    parser.add_argument("--tdb_tau_q", type=float, default=0.0,
                       help="Optional target exposure lower bound for TDB-AS")
    parser.add_argument("--tdb_exposure_rho", type=float, default=0.8,
                       help="Temporal decay for TDB-AS ring exposure")
    parser.add_argument("--tdb_use_target_similarity", action="store_true",
                       help="Multiply topology exposure by target/update sketch similarity")
    parser.add_argument("--tdb_time_limit", type=float, default=30.0,
                       help="TDB-AS MILP solver time limit in seconds")
    parser.add_argument("--tdb_aggregation_scope", type=str, default="local",
                       choices=["local", "global"],
                       help="Use local ring neighborhoods or the full selected overlay for TDB weighted aggregation")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for agent selection (omit or set -1 for random)")
    parser.add_argument("--gpu", type=int, default=None,
                       help="GPU device ID to use (default: auto)")

    # LoRA parameter selection parameters
    parser.add_argument("--enable_param_selection", action="store_true",
                       help="Enable LoRA parameter selection for DFU")
    parser.add_argument("--param_selection_ratio", type=float, default=0.5,
                       help="Maximum ratio of LoRA modules to select")
    parser.add_argument("--param_epsilon_W", type=float, default=0.1,
                       help="Energy threshold for LoRA parameter selection (absolute)")
    parser.add_argument("--param_random_selection", action="store_true",
                       help="Use random LoRA module selection instead of sensitivity-based")
    parser.add_argument("--param_relative_sensitivity", action="store_true",
                       help="Use relative sensitivity weighting for LoRA selection")
    parser.add_argument("--param_sensitivity_alpha", type=float, default=1.0,
                       help="Exponent alpha for relative sensitivity weighting")
    parser.add_argument("--param_selection_mode", type=str, default="epsilon",
                       choices=["epsilon", "top_ratio"],
                       help="LoRA module selection mode: epsilon-greedy (default) or top_ratio")
    parser.add_argument("--param_sensitivity_cache", type=str, default=None,
                       help="Optional JSON cache path for computed LoRA sensitivities")
    
    # Retrain TR distribution for Forget Quality calculation
    parser.add_argument("--retrain_tr_path", type=str, default=None,
                       help="Path to retrain model's TR distribution (.npy file) for FQ calculation")

    # Rapidretrain schedule parameters (global rounds)
    parser.add_argument("--jo", type=int, default=10,
                       help="Rapidretrain warmup rounds j0 (global round index)")
    parser.add_argument("--t0", type=int, default=5,
                       help="Rapidretrain SGD period T0 (global rounds)")

    # D-Oblivionis specific parameters
    parser.add_argument("--unlearn_rounds", type=int, default=3,
                       help="Number of unlearn rounds for D-Oblivionis/D-FedOSD (default: 3)")
    parser.add_argument("--unlearn_epochs", type=int, default=1,
                       help="Number of epochs per unlearn round for D-Oblivionis (default: 1)")
    parser.add_argument("--unlearn_local_steps", type=int, default=None,
                       help="Number of local training steps per unlearn round for D-Oblivionis "
                            "(default: use snapshot local_steps; set <=0 to use unlearn_epochs)")
    parser.add_argument("--unlearn_lr", type=float, default=1e-4,
                       help="Learning rate for unlearn phase (default: 1e-4)")
    parser.add_argument("--propagation_rounds", type=int, default=5,
                       help="Number of propagation rounds for D-Oblivionis (default: 5)")
    parser.add_argument("--propagation_epochs", type=int, default=1,
                       help="Number of epochs per propagation round for D-Oblivionis (default: 1, ignored if propagation_local_steps is set)")
    parser.add_argument("--propagation_local_steps", type=int, default=None,
                       help="Number of local training steps per propagation round for D-Oblivionis "
                            "(default: use snapshot local_steps; set <=0 to use propagation_epochs)")
    parser.add_argument("--propagation_lr", type=float, default=1e-4,
                       help="Learning rate for propagation phase (default: 1e-4)")

    # D-FedOSD specific parameters
    parser.add_argument("--recovery_rounds", type=int, default=5,
                       help="Number of recovery rounds for D-FedOSD/D-FedRecovery (default: 5)")
    parser.add_argument("--recovery_epochs", type=int, default=1,
                       help="Number of epochs per recovery round for D-FedOSD/D-FedRecovery (default: 1, ignored if recovery_local_steps is set)")
    parser.add_argument("--recovery_local_steps", type=int, default=5,
                       help="Number of local training steps per recovery round (default: 5, same as DFL local_steps)")
    parser.add_argument("--recovery_lr", type=float, default=1e-5,
                       help="Learning rate for recovery phase (default: 1e-5)")
    parser.add_argument("--retain_grad_samples", type=int, default=50,
                       help="Number of samples per agent for retain gradient computation in D-FedOSD (default: 50)")
    parser.add_argument("--forget_grad_samples", type=int, default=None,
                       help="Number of target-agent forget samples for D-FedOSD unlearn gradient (default: all)")
    parser.add_argument("--fedosd_retain_subspace_mode", type=str, default="separate",
                       choices=["separate", "mean"],
                       help="FedOSD retain subspace construction: separate neighbor gradients or their local mean (default: separate)")
    parser.add_argument("--fedosd_orthogonal_update_norm", type=float, default=None,
                       help="Optional target norm for FedOSD orthogonal unlearning direction. Default keeps raw projected norm.")
    parser.add_argument("--fedosd_projection_strength", type=float, default=1.0,
                       help="FedOSD projection strength gamma in d=g_u-gamma*Proj_A(g_u). 1.0 keeps standard full projection.")
    parser.add_argument("--fedosd_forget_loss", type=str, default="uce",
                       choices=["uce", "grad_ascent"],
                       help="Forget loss used by D-FedOSD. 'uce' keeps the original UCE objective; "
                            "'grad_ascent' keeps the OSD projection but uses CE gradient ascent for LLM classification-as-generation.")
    parser.add_argument("--cache_retain_grads", action="store_true", default=True,
                       help="Cache retain gradients in unlearn phase (default: True, surviving agents' models don't change)")
    parser.add_argument("--no_cache_retain_grads", action="store_false", dest="cache_retain_grads",
                       help="Disable retain gradient caching (recompute every round)")

    # D-FedRecovery specific parameters
    parser.add_argument("--correction_weight", type=float, default=5.0,
                       help="Correction weight for D-FedRecovery (default: 5.0)")
    parser.add_argument("--noise_std", type=float, default=0.0,
                       help="Gaussian noise std for D-FedRecovery privacy (default: 0.0, no noise)")
    parser.add_argument("--fedrecovery_correction_mode", type=str, default="residual",
                       choices=["decentralized_replay", "residual"],
                       help="D-FedRecovery correction mode: decentralized target-free ring replay or legacy residual correction")

    args = parser.parse_args()
    main(args)
