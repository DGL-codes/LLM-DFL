"""DFL training script."""
import argparse
import json
import torch
import random
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.datasets import (
    NewsGroupsDataset, DBpediaDataset, YahooAnswersDataset, YahooSubsetDataset,
    AlpacaGPT4Dataset, FinGPTSentimentDataset,
    MedicalFlashcardsDataset, CodeAlpacaDataset, TOFUDataset
)
from src.utils.gpu_guard import guard_gpu_or_raise
from src.data.partitioner import DirichletPartitioner, PartitionInfo
from src.data.collator import LLMCollator
from src.data.backdoor_wrapper import BackdoorPoisonSpec, poison_samples_by_indices
from src.models.lora_model import LoRAModelWrapper
from src.dfl.trainer import DFLTrainer


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


def main(args):
    # Enforce physical GPU 2/3 only (args.gpu is LOGICAL within CUDA_VISIBLE_DEVICES)
    visible_physical = guard_gpu_or_raise(gpu=args.gpu)

    # Reproducibility
    seed = args.seed
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

    # Create hierarchical output directory
    # checkpoints/{dataset}/K{num_agents}/G{global}_L{local}/alpha{alpha}/seed{seed}_{timestamp}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(args.output_dir) /
        args.dataset /
        f"K{args.num_agents}" /
        f"G{args.global_rounds}_L{args.local_steps}" /
        f"alpha{args.alpha}" /
        f"seed{args.seed}_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = vars(args)
    config["timestamp"] = timestamp
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Load dataset
    print(f"Loading {args.dataset} dataset...")
    DatasetClass = DATASET_MAP[args.dataset]
    
    # TOFU数据集：800条全部用于训练，不需要val/test划分
    if args.dataset == "tofu":
        train_ds = DatasetClass(
            split="train", 
            max_samples=args.max_train_samples, 
            num_authors=args.num_authors,
            group_size=args.group_size
        )
        # TOFU遗忘任务：全部数据用于训练，评估使用官方perturbed数据
        train_samples = train_ds.samples
        val_samples = []  # 不需要val
        test_ds = train_ds  # 评估时使用tofu_evaluator，这里只是占位
    else:
        train_ds = DatasetClass(split="train", max_samples=args.max_train_samples)
        test_ds = DatasetClass(split="test", max_samples=args.max_eval_samples)
        # Split train into train/val
        val_size = int(len(train_ds) * 0.1)
        train_samples = train_ds.samples[val_size:]
        val_samples = train_ds.samples[:val_size]
    
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_ds)}")
    
    # Create partition
    print(f"Creating Dirichlet partition (alpha={args.alpha})...")
    labels = [s.label for s in train_samples]
    partitioner = DirichletPartitioner(
        num_agents=args.num_agents,
        alpha=args.alpha,
        seed=args.seed
    )
    partition = partitioner.partition(labels, args.dataset)

    # Print partition statistics and verify total
    total_partitioned = 0
    for i, indices in partition.agent_indices.items():
        print(f"  Agent {i}: {len(indices)} samples")
        total_partitioned += len(indices)

    # Verify partition correctness
    assert total_partitioned == len(train_samples), \
        f"Partition error: {total_partitioned} != {len(train_samples)}"
    print(f"  Total: {total_partitioned} samples (verified)")

    # Optional: backdoor poisoning (unlearning audit). Only supported for classification tasks.
    if args.backdoor_trigger and float(args.backdoor_rate) > 0:
        if DatasetClass.TASK_TYPE != "classification":
            raise ValueError("Backdoor poisoning is only supported for classification tasks in this repo.")
        if not getattr(train_ds, "label_names", None):
            raise ValueError("label_names is required for classification backdoor poisoning.")
        target_agent = int(args.backdoor_target_agent)
        if target_agent < 0 or target_agent >= int(args.num_agents):
            raise ValueError(f"Invalid --backdoor_target_agent={target_agent} for K={args.num_agents}")

        poison_seed = int(args.backdoor_seed) if args.backdoor_seed is not None else int(args.seed)
        spec = BackdoorPoisonSpec(
            trigger=str(args.backdoor_trigger),
            poison_rate=float(args.backdoor_rate),
            target_label=int(args.backdoor_target_label),
            position=str(args.backdoor_position),
            seed=poison_seed,
        )
        poisoned = poison_samples_by_indices(
            train_samples,
            partition.agent_indices[target_agent],
            label_names=train_ds.label_names,
            spec=spec,
        )
        print(
            f"Backdoor poisoning enabled: agent={target_agent}, rate={spec.poison_rate}, "
            f"target_label={spec.target_label}, poisoned={poisoned}/{len(partition.agent_indices[target_agent])}"
        )
    
    # Initialize model
    print("Initializing model...")
    model = LoRAModelWrapper(
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        device=device
    )
    model.load_base_model()
    model.init_lora()
    
    collator = LLMCollator(model.tokenizer, max_length=args.max_length)
    
    # Initialize DFL trainer
    print("Initializing DFL trainer...")
    dfl_trainer = DFLTrainer(
        num_agents=args.num_agents,
        model=model,
        collator=collator,
        partition=partition,
        all_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_ds.samples,
        max_eval_samples=args.max_eval_samples,
        label_names=getattr(train_ds, 'label_names', None),
        task_type=DatasetClass.TASK_TYPE,
        lr=args.lr,
        device=device,
        seed=args.seed
    )
    if args.dataset == "tofu":
        # DFL场景对齐：agent0作为forget，agent1作为retain，评估使用agent1模型
        dfl_trainer.set_eval_forget_agent_id(0)
        dfl_trainer.set_eval_retain_agent_id(1)
        dfl_trainer.set_eval_agent_id(1)

    # Run DFL training
    print("\nStarting DFL training...")
    dfl_trainer.train(
        global_rounds=args.global_rounds,
        local_steps=args.local_steps,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        save_dir=str(output_dir),
        eval_every=args.eval_every
    )
    
    print(f"\nDFL training completed! Results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=list(DATASET_MAP.keys()))
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--num_agents", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--global_rounds", type=int, default=10)
    parser.add_argument("--local_steps", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--eval_every", type=int, default=1)
    # TOFU数据集专用参数
    parser.add_argument("--num_authors", type=int, default=40, help="TOFU数据集使用的作者数量（默认40）")
    parser.add_argument("--group_size", type=int, default=2, help="TOFU数据集作者分组大小（默认2，即20组）")
    # Backdoor audit (classification only)
    parser.add_argument("--backdoor_trigger", type=str, default=None, help="Trigger string to inject (enables poisoning when set).")
    parser.add_argument("--backdoor_rate", type=float, default=0.0, help="Poisoning rate within the target agent's local data.")
    parser.add_argument("--backdoor_target_label", type=int, default=0, help="Target label index for the backdoor.")
    parser.add_argument("--backdoor_target_agent", type=int, default=0, help="Which agent to poison (default: 0).")
    parser.add_argument("--backdoor_seed", type=int, default=None, help="Poison RNG seed (default: use --seed).")
    parser.add_argument("--backdoor_position", type=str, default="prefix", choices=["prefix", "suffix"], help="Where to insert the trigger.")
    parser.add_argument("--gpu", type=int, default=None, help="GPU device ID to use (default: auto)")
    
    args = parser.parse_args()
    main(args)
