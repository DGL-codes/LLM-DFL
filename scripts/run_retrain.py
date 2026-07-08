"""Retrain-from-scratch baseline script.

从DFL checkpoint读取配置，排除目标客户端后重新训练，作为遗忘的理想基线。
支持 20newsgroups, yahoo_subset, tofu 数据集。
"""
import argparse
import json
import torch
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.datasets import (
    NewsGroupsDataset, YahooSubsetDataset, TOFUDataset
)
from src.data.partitioner import DirichletPartitioner, PartitionInfo
from src.data.collator import LLMCollator
from src.data.base import Sample
from src.models.lora_model import LoRAModelWrapper
from src.dfl.trainer import DFLTrainer
from src.utils.gpu_guard import guard_gpu_or_raise


DATASET_MAP = {
    "20newsgroups": NewsGroupsDataset,
    "yahoo_subset": YahooSubsetDataset,
    "tofu": TOFUDataset,
}


def _load_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
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
    items: List[Dict[str, Any]] = []
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
    if not path.exists():
        raise FileNotFoundError(path)
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
            pert_list = [pert] if pert else []
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


def build_rapidretrain_schedule(global_rounds: int, jo: int, t0: int):
    """Build optimizer schedule for rapidretrain (per global round)."""
    if global_rounds <= 0:
        return [], jo, t0
    if jo >= global_rounds:
        adapted_jo = max(1, global_rounds // 2 - 1)
        if adapted_jo >= global_rounds:
            adapted_jo = max(0, global_rounds - 2)
        adapted_t0 = max(2, global_rounds // 3)
        print(
            f"Rapidretrain: j0={jo} >= global_rounds={global_rounds}, "
            f"use j0={adapted_jo}, T0={adapted_t0}"
        )
        jo = adapted_jo
        t0 = adapted_t0
    if t0 <= 0:
        print("Rapidretrain: T0<=0, set T0=1")
        t0 = 1

    schedule = []
    for i in range(global_rounds):
        use_sgd = (i <= jo) or ((i - jo) % t0 == 0)
        schedule.append("sgd" if use_sgd else "adahessian")
    return schedule, jo, t0


def load_dfl_config(dfl_checkpoint: str) -> dict:
    """加载DFL checkpoint的配置文件。"""
    config_path = Path(dfl_checkpoint) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"DFL config not found: {config_path}")
    
    with open(config_path, "r") as f:
        return json.load(f)


def load_partition_info(dfl_checkpoint: str) -> dict:
    """加载DFL checkpoint的partition信息。"""
    # 尝试两种可能的文件名
    partition_path = Path(dfl_checkpoint) / "partition_info.json"
    if not partition_path.exists():
        partition_path = Path(dfl_checkpoint) / "partition.json"
    if not partition_path.exists():
        raise FileNotFoundError(f"Partition info not found in: {dfl_checkpoint}")
    
    with open(partition_path, "r") as f:
        return json.load(f)


def create_retrain_partition(
    original_partition: dict,
    target_agent: int,
    labels: list,
    excluded_indices: set = None
) -> PartitionInfo:
    """创建排除目标客户端后的partition。
    
    Args:
        original_partition: 原始partition信息
        target_agent: 要排除的客户端ID
        labels: 所有样本的标签列表
        excluded_indices: 被排除的样本索引集合（如val_samples的索引）
        
    Returns:
        新的PartitionInfo，不包含目标客户端
    """
    original_indices = {int(k): v for k, v in original_partition["agent_indices"].items()}
    num_agents = original_partition["num_agents"]
    
    # 如果有被排除的索引，需要重新映射
    if excluded_indices:
        # 创建旧索引到新索引的映射
        old_to_new = {}
        new_idx = 0
        max_old_idx = max(max(indices) for indices in original_indices.values())
        for old_idx in range(max_old_idx + 1):
            if old_idx not in excluded_indices:
                old_to_new[old_idx] = new_idx
                new_idx += 1
    else:
        old_to_new = None
    
    # 排除目标客户端
    new_agent_indices = {}
    new_agent_id = 0
    for agent_id in range(num_agents):
        if agent_id == target_agent:
            continue
        
        if old_to_new:
            # 重新映射索引，排除不在映射中的索引
            new_indices = [old_to_new[idx] for idx in original_indices[agent_id] if idx in old_to_new]
        else:
            new_indices = original_indices[agent_id]
        
        new_agent_indices[new_agent_id] = new_indices
        new_agent_id += 1
    
    # 计算新的agent_weights
    total_samples = sum(len(indices) for indices in new_agent_indices.values())
    new_agent_weights = {
        k: len(v) / total_samples for k, v in new_agent_indices.items()
    }
    
    return PartitionInfo(
        dataset_name=original_partition["dataset_name"],
        num_agents=num_agents - 1,
        alpha=original_partition.get("alpha"),
        seed=original_partition["seed"],
        partition_type=original_partition["partition_type"],
        agent_indices=new_agent_indices,
        agent_weights=new_agent_weights
    )


def create_tofu_fed_retrain_partition(
    *,
    original_partition: dict,
    target_agent: int,
    forget_offset: int,
) -> PartitionInfo:
    """Create a retrain partition for tofu_fed snapshots.

    tofu_fed DFL ordering is: all_samples = forget_samples + retain_samples.
    Retain client indices are therefore offset by `forget_offset`.
    This retrain partition:
      - removes `target_agent` (usually agent0)
      - keeps the original retain client assignments
      - remaps sample indices to the retain-only training set (subtract offset)
      - remaps agent ids to 0..K-2
    """
    original_indices = {int(k): v for k, v in original_partition["agent_indices"].items()}
    num_agents = int(original_partition["num_agents"])
    if forget_offset <= 0:
        raise ValueError(f"Invalid forget_offset: {forget_offset}")
    if target_agent not in original_indices:
        raise ValueError(f"target_agent {target_agent} not found in original partition")

    new_agent_indices: Dict[int, List[int]] = {}
    new_aid = 0
    for old_aid in range(num_agents):
        if old_aid == target_agent:
            continue
        old_idxs = [int(i) for i in original_indices.get(old_aid, [])]
        new_idxs = [i - int(forget_offset) for i in old_idxs if i >= int(forget_offset)]
        new_agent_indices[int(new_aid)] = new_idxs
        new_aid += 1

    total_samples = sum(len(v) for v in new_agent_indices.values())
    new_agent_weights = (
        {k: len(v) / total_samples for k, v in new_agent_indices.items()} if total_samples > 0 else None
    )

    return PartitionInfo(
        dataset_name=str(original_partition.get("dataset_name", "tofu_fed")),
        num_agents=num_agents - 1,
        alpha=original_partition.get("alpha"),
        seed=original_partition.get("seed", 42),
        partition_type=str(original_partition.get("partition_type", "tofu_fed_dirichlet")),
        agent_indices=new_agent_indices,
        agent_weights=new_agent_weights,
    )


def main(args):
    import time
    start_time = time.time()

    visible_physical = guard_gpu_or_raise(gpu=args.gpu)
    
    # 设置设备
    if args.gpu is not None:
        device = f"cuda:{args.gpu}"
        torch.cuda.set_device(args.gpu)
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"CUDA_VISIBLE_DEVICES (physical): {visible_physical}")
    
    # 加载DFL配置
    print(f"\nLoading DFL config from: {args.dfl_checkpoint}")
    dfl_config = load_dfl_config(args.dfl_checkpoint)
    original_partition = load_partition_info(args.dfl_checkpoint)
    
    dataset_name = dfl_config["dataset"]
    num_agents = dfl_config["num_agents"]
    alpha = dfl_config["alpha"]
    seed = dfl_config["seed"]
    global_rounds = dfl_config["global_rounds"]
    local_steps = dfl_config["local_steps"]

    # Allow overriding memory-sensitive knobs for retrain runs.
    # (Useful when the original DFL settings are near OOM limits.)
    batch_size = int(args.batch_size) if args.batch_size is not None else int(dfl_config.get("batch_size", 4))
    grad_accum_steps = (
        int(args.grad_accum_steps) if args.grad_accum_steps is not None else int(dfl_config.get("grad_accum_steps", 4))
    )
    max_length = int(args.max_length) if args.max_length is not None else int(dfl_config.get("max_length", 512))
    
    print(f"  Dataset: {dataset_name}")
    print(f"  Original agents: {num_agents}")
    print(f"  Target agent to exclude: {args.target_agent}")
    print(f"  Alpha: {alpha}")
    print(f"  Global rounds: {global_rounds}")
    print(f"  Local steps: {local_steps}")
    
    # rapidretrain optimizer schedule (per global round)
    optimizer_schedule = None
    effective_jo = args.jo
    effective_t0 = args.t0
    if args.method_name == "rapidretrain":
        optimizer_schedule, effective_jo, effective_t0 = build_rapidretrain_schedule(
            global_rounds, args.jo, args.t0
        )
        if optimizer_schedule:
            sgd_rounds = [i + 1 for i, opt in enumerate(optimizer_schedule) if opt == "sgd"]
            ada_rounds = [i + 1 for i, opt in enumerate(optimizer_schedule) if opt == "adahessian"]
            print("\nRapidretrain schedule (global rounds, 1-based):")
            print(f"  SGD: {sgd_rounds}")
            print(f"  AdaHessian: {ada_rounds}")

    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dfl_snapshot_name = Path(args.dfl_checkpoint).name
    
    if args.method_name == "retrain":
        output_dir = (
            Path(args.output_dir) /
            dataset_name /
            f"K{num_agents}" /
            f"G{global_rounds}_L{local_steps}" /
            f"alpha{alpha}" /
            "strategy_retrain" /
            dfl_snapshot_name /
            f"retrain_{timestamp}"
        )
    else:
        output_dir = (
            Path(args.output_dir) /
            dataset_name /
            args.method_name /
            f"K{num_agents}" /
            f"G{global_rounds}_L{local_steps}" /
            f"alpha{alpha}" /
            f"strategy_{args.method_name}" /
            dfl_snapshot_name /
            f"{args.method_name}_{timestamp}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nRetrain output directory: {output_dir}")
    
    # 加载数据集
    print(f"\nLoading {dataset_name} dataset...")
    DatasetClass = DATASET_MAP.get(dataset_name)
    if DatasetClass is None:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Supported: {list(DATASET_MAP.keys())}")
    
    # TOFU数据集需要特殊处理
    tofu_fed = False
    tofu_local_dir: Optional[str] = None
    tofu_forget_rate: Optional[str] = None
    if dataset_name == "tofu":
        tofu_fed = bool(dfl_config.get("tofu_fed", False))
        if tofu_fed:
            tofu_local_dir = str(args.tofu_local_dir or dfl_config.get("tofu_local_dir") or "").strip()
            tofu_forget_rate = str(dfl_config.get("forget_rate") or "").strip()
            tofu_retain_split = str(dfl_config.get("retain_split") or "").strip()
            if not tofu_retain_split and tofu_forget_rate:
                tofu_retain_split = {
                    "forget01": "retain99",
                    "forget05": "retain95",
                    "forget10": "retain90",
                }.get(tofu_forget_rate, "")
            if not tofu_local_dir:
                raise ValueError("TOFU fed snapshot missing tofu_local_dir (pass --tofu_local_dir or set in snapshot config)")
            if not tofu_forget_rate:
                raise ValueError("TOFU fed snapshot missing forget_rate in config.json")
            if not tofu_retain_split:
                raise ValueError(f"TOFU fed snapshot missing/invalid retain_split for forget_rate={tofu_forget_rate!r}")

            num_authors = int(dfl_config.get("num_authors", 200))
            group_size = int(dfl_config.get("group_size", 2))
            qa_per_author = int(dfl_config.get("qa_per_author", 20))
            if args.max_train_samples is not None:
                print(
                    "[WARN] --max_train_samples is ignored for tofu_fed retrain "
                    "(partition indices require the full retain split ordering)."
                )

            train_ds = DatasetClass(
                split="train",
                max_samples=None,
                num_authors=num_authors,
                group_size=group_size,
                tofu_local_dir=tofu_local_dir,
                tofu_split=tofu_retain_split,
                qa_per_author=qa_per_author,
            )
            task_type = "tofu"
            train_samples = train_ds.samples
            val_samples = []
            test_samples = []
            print(
                f"TOFU fed retrain: train(retain)={tofu_retain_split} ({len(train_samples)} samples), "
                f"eval(forget)={tofu_forget_rate}_perturbed"
            )
        else:
            num_authors = int(dfl_config.get("num_authors", 20))
            group_size = int(dfl_config.get("group_size", 2))
            train_ds = DatasetClass(
                split="train",
                max_samples=args.max_train_samples,
                num_authors=num_authors,
                group_size=group_size
            )
            task_type = "tofu"
            # 与 train_dfl.py 对齐：全部训练样本，不划分 val/test
            train_samples = train_ds.samples
            val_samples = []
            test_samples = []
    else:
        train_ds = DatasetClass(split="train", max_samples=args.max_train_samples)
        test_ds = DatasetClass(split="test", max_samples=args.max_eval_samples)
        task_type = DatasetClass.TASK_TYPE
        # 划分train/val
        val_size = int(len(train_ds) * 0.1)
        train_samples = train_ds.samples[val_size:]
        val_samples = train_ds.samples[:val_size]
        test_samples = test_ds.samples
    
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")
    
    # 创建排除目标客户端后的partition
    if dataset_name == "tofu" and tofu_fed:
        forget_offset = len(original_partition["agent_indices"][str(args.target_agent)])
        retrain_partition = create_tofu_fed_retrain_partition(
            original_partition=original_partition,
            target_agent=int(args.target_agent),
            forget_offset=int(forget_offset),
        )
    else:
        labels = [s.label for s in train_samples]
        retrain_partition = create_retrain_partition(
            original_partition, args.target_agent, labels
        )
    
    print(f"\nRetrain partition (excluding agent {args.target_agent}):")
    total_samples = 0
    for i, indices in retrain_partition.agent_indices.items():
        print(f"  Agent {i}: {len(indices)} samples")
        total_samples += len(indices)
    print(f"  Total: {total_samples} samples")
    
    # 保存retrain配置
    save_only_final = not args.save_all_rounds
    optimizer_label = args.optimizer
    if args.method_name == "rapidretrain":
        optimizer_label = "rapidretrain"

    retrain_config = {
        "dfl_checkpoint": args.dfl_checkpoint,
        "target_agent": args.target_agent,
        "dataset": dataset_name,
        "num_agents": num_agents - 1,  # 排除目标客户端后
        "original_num_agents": num_agents,
        "alpha": alpha,
        "seed": seed,
        "global_rounds": global_rounds,
        "local_steps": local_steps,
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum_steps,
        "lr": dfl_config.get("lr", 1e-4),
        "lora_r": dfl_config.get("lora_r", 8),
        "lora_alpha": dfl_config.get("lora_alpha", 16),
        "max_length": max_length,
        "timestamp": timestamp,
        "excluded_samples": len(original_partition["agent_indices"][str(args.target_agent)]),
        "retrain_samples": total_samples,
        "optimizer": optimizer_label,
        "save_only_final": save_only_final,
        "method_name": args.method_name,
        "jo": effective_jo,
        "t0": effective_t0,
        "optimizer_schedule": optimizer_schedule
    }
    
    with open(output_dir / "retrain_config.json", "w") as f:
        json.dump(retrain_config, f, indent=2)
    
    # 初始化模型
    print("\nInitializing model...")
    model = LoRAModelWrapper(
        lora_r=dfl_config.get("lora_r", 8),
        lora_alpha=dfl_config.get("lora_alpha", 16),
        device=device
    )
    model.load_base_model()
    model.init_lora()
    
    collator = LLMCollator(model.tokenizer, max_length=dfl_config.get("max_length", 512))
    if max_length != int(dfl_config.get("max_length", 512)):
        collator = LLMCollator(model.tokenizer, max_length=max_length)
    
    # 初始化DFL trainer（用于retrain）
    print("\nInitializing DFL trainer for retrain...")
    init_optimizer = args.optimizer
    if optimizer_schedule:
        init_optimizer = optimizer_schedule[0]

    dfl_trainer = DFLTrainer(
        num_agents=num_agents - 1,
        model=model,
        collator=collator,
        partition=retrain_partition,
        all_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        max_eval_samples=args.max_eval_samples,
        tofu_local_dir=tofu_local_dir,
        label_names=getattr(train_ds, 'label_names', None),
        task_type=task_type,
        lr=dfl_config.get("lr", 1e-4),
        device=device,
        optimizer_name=init_optimizer,
        seed=seed
    )
    
    # 对于TOFU数据集：设置外部 eval retain/forget（使用带 perturbed_answer 的官方口径）
    if dataset_name == "tofu":
        if tofu_fed:
            if not tofu_local_dir or not tofu_forget_rate:
                raise ValueError("tofu_fed retrain requires tofu_local_dir and tofu_forget_rate")

            external_retain_samples = load_tofu_perturbed_as_samples(
                tofu_local_dir=tofu_local_dir, config_name="retain_perturbed"
            )
            external_forget_samples = load_tofu_perturbed_as_samples(
                tofu_local_dir=tofu_local_dir, config_name=f"{tofu_forget_rate}_perturbed"
            )
            print(f"\nTOFU fed eval splits:")
            print(f"  retain_perturbed: {len(external_retain_samples)} samples")
            print(f"  {tofu_forget_rate}_perturbed: {len(external_forget_samples)} samples")
            dfl_trainer.set_external_retain_samples(external_retain_samples)
            dfl_trainer.set_external_forget_samples(external_forget_samples)

            retrain_config.update(
                {
                    "tofu_fed": True,
                    "tofu_local_dir": tofu_local_dir,
                    "forget_rate": tofu_forget_rate,
                    "retain_split": str(dfl_config.get("retain_split") or ""),
                }
            )
            with open(output_dir / "retrain_config.json", "w", encoding="utf-8") as f:
                json.dump(retrain_config, f, ensure_ascii=False, indent=2)
        else:
            # Legacy (repo-local) TOFU: use the excluded client's original training samples for eval.
            original_target_indices = original_partition["agent_indices"][str(args.target_agent)]
            print(f"\nOriginal target agent {args.target_agent} had {len(original_target_indices)} samples")

            full_train_ds = DatasetClass(split="train", max_samples=None, num_authors=num_authors)
            external_forget_samples = [
                full_train_ds.samples[idx]
                for idx in original_target_indices
                if idx < len(full_train_ds.samples)
            ]
            print(f"Loaded {len(external_forget_samples)} external forget samples for evaluation")

            external_retain_samples: List[Sample] = []
            for aid, idx_list in original_partition["agent_indices"].items():
                aid_int = int(aid)
                if aid_int == args.target_agent:
                    continue
                external_retain_samples.extend(
                    [full_train_ds.samples[idx] for idx in idx_list if idx < len(full_train_ds.samples)]
                )
            print(
                f"Loaded {len(external_retain_samples)} external retain samples for evaluation "
                f"(all non-target agents)"
            )

            dfl_trainer.set_external_forget_samples(external_forget_samples)
            dfl_trainer.set_external_retain_samples(external_retain_samples)

            retrain_config["excluded_sample_indices"] = original_target_indices
            with open(output_dir / "retrain_config.json", "w", encoding="utf-8") as f:
                json.dump(retrain_config, f, ensure_ascii=False, indent=2)
    
    # 运行retrain
    print("\nStarting retrain from scratch...")
    dfl_trainer.train(
        global_rounds=global_rounds,
        local_steps=local_steps,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        save_dir=str(output_dir),
        eval_every=args.eval_every,
        save_only_final=save_only_final,
        optimizer_schedule=optimizer_schedule
    )
    
    print(f"\nRetrain completed! Results saved to {output_dir}")
    
    # 记录时间和内存
    elapsed_time = time.time() - start_time
    elapsed_min = elapsed_time / 60
    print(f"\n{'='*50}")
    print(f"Retrain完成统计:")
    print(f"  总运行时间: {elapsed_time:.2f}秒 ({elapsed_min:.2f}分钟)")
    if torch.cuda.is_available():
        gpu_id = args.gpu if args.gpu is not None else 0
        allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
        max_allocated = torch.cuda.max_memory_allocated(gpu_id) / 1024**3
        print(f"  GPU {gpu_id} 内存使用:")
        print(f"    当前分配: {allocated:.2f} GB")
        print(f"    峰值分配: {max_allocated:.2f} GB")
    print(f"{'='*50}\n")
    
    # 清理显存
    del dfl_trainer
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("GPU memory cleared.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrain from scratch baseline")
    
    # 必需参数
    parser.add_argument("--dfl_checkpoint", type=str, required=True,
                       help="Path to DFL checkpoint directory")
    
    # 可选参数
    parser.add_argument("--output_dir", type=str, default="retrain_checkpoints",
                       help="Base output directory for retrain results")
    parser.add_argument("--target_agent", type=int, default=0,
                       help="Agent ID to exclude (default: 0)")
    parser.add_argument("--eval_every", type=int, default=0,
                       help="Evaluate every N global rounds (<=0 to skip middle evals; final eval always runs)")
    parser.add_argument("--max_train_samples", type=int, default=None,
                       help="Max training samples")
    parser.add_argument("--max_eval_samples", type=int, default=None,
                       help="Max evaluation samples")
    parser.add_argument("--batch_size", type=int, default=None,
                       help="Override batch size (default: use DFL config)")
    parser.add_argument("--grad_accum_steps", type=int, default=None,
                       help="Override grad accumulation steps (default: use DFL config)")
    parser.add_argument("--max_length", type=int, default=None,
                       help="Override max sequence length (default: use DFL config)")
    parser.add_argument("--tofu_local_dir", type=str, default=None,
                       help="Local TOFU dir containing json/jsonl splits (used for tofu_fed checkpoints).")
    parser.add_argument("--optimizer", type=str, default="adamw",
                       choices=["adamw", "adahessian", "sgd"],
                       help="Optimizer to use for retrain")
    parser.add_argument("--method_name", type=str, default="retrain",
                       choices=["retrain", "rapidretrain"],
                       help="Method name (retrain or rapidretrain)")
    parser.add_argument("--jo", type=int, default=10,
                       help="Rapidretrain warmup rounds j0 (global round index)")
    parser.add_argument("--t0", type=int, default=5,
                       help="Rapidretrain SGD period T0 (global rounds)")
    parser.add_argument("--save_all_rounds", action="store_true",
                       help="Save all rounds (default: only final round)")
    parser.add_argument("--gpu", type=int, default=None,
                       help="GPU device ID to use (default: auto)")
    
    args = parser.parse_args()
    main(args)
