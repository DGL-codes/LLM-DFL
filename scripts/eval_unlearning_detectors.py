#!/usr/bin/env python3
"""Audit unlearning detectors (MIA) on DFL vs DFU vs Retrain checkpoints.

This script is designed to answer a concrete question:
  "Is the current MIA detector actually capable of separating members from
   non-members on the *original* (DFL) model?"

If the detector cannot distinguish members/non-members on DFL (AUC_sym ~ 0.5),
then it is not suitable as an unlearning verifier; switch to a different audit
signal (e.g., backdoor ASR).

Outputs
-------
Writes JSON + plots under:
  artifacts/unlearning_audit/mia/<tag>/

GPU
---
Must run with physical GPU 2/3 only:
  CUDA_VISIBLE_DEVICES=2,3  (then --gpu is logical 0/1)
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_ROOT = Path(os.environ.get("LLMDFL_ARTIFACT_ROOT", "artifacts"))

import sys

sys.path.insert(0, str(ROOT))

from src.data.datasets import NewsGroupsDataset, YahooSubsetDataset  # noqa: E402
from src.data.partitioner import PartitionInfo  # noqa: E402
from src.data.collator import LLMCollator  # noqa: E402
from src.models.lora_model import LoRAModelWrapper  # noqa: E402
from src.dfu.snapshot_loader import SnapshotLoader  # noqa: E402
from src.dfu.unlearning_detectors import MIADetectorResult, run_mia_detector  # noqa: E402
from src.utils.gpu_guard import guard_gpu_or_raise  # noqa: E402


DATASET_MAP = {
    "20newsgroups": NewsGroupsDataset,
    "yahoo_subset": YahooSubsetDataset,
}


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _find_dfu_lora_state(run_dir: Path, agent_id: int) -> Optional[Path]:
    candidates = [
        run_dir / "final" / f"agent_{agent_id}" / "lora_state.pt",
        run_dir / f"agent_{agent_id}" / "lora_state.pt",
    ]
    round_dirs = sorted(
        [d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("round_")],
        key=lambda p: int(p.name.split("_", 1)[1]) if p.name.split("_", 1)[1].isdigit() else -1,
    ) if run_dir.exists() else []
    for rd in reversed(round_dirs):
        candidates.append(rd / f"agent_{agent_id}" / "lora_state.pt")
    for p in candidates:
        if p.exists():
            return p
    return None


def _find_retrain_lora_state(retrain_dir: Path, *, round_idx: int, agent_id: int) -> Optional[Path]:
    p = retrain_dir / f"round_{round_idx}" / f"agent_{agent_id}" / "lora_state.pt"
    return p if p.exists() else None


def _load_train_test(
    dataset: str,
    *,
    max_train_samples: Optional[int],
    max_test_samples: Optional[int],
) -> Tuple[List, List, List, List[str]]:
    Dataset = DATASET_MAP[dataset]
    train_ds = Dataset(split="train", max_samples=max_train_samples)
    test_ds = Dataset(split="test", max_samples=max_test_samples)
    val_size = int(len(train_ds) * 0.1)
    train_samples = train_ds.samples[val_size:]
    val_samples = train_ds.samples[:val_size]
    test_samples = test_ds.samples
    label_names = getattr(train_ds, "label_names", [])
    return train_samples, val_samples, test_samples, label_names


def _select_equal_prefix(a: List, b: List, n: int) -> Tuple[List, List]:
    n_eff = min(n, len(a), len(b))
    return a[:n_eff], b[:n_eff]


def _plot_hist(member_scores: List[float], nonmember_scores: List[float], title: str, out_path: Path) -> None:
    if not member_scores or not nonmember_scores:
        return
    _ensure_dir(out_path.parent)
    plt.figure(figsize=(7, 4))
    plt.hist(member_scores, bins=40, alpha=0.5, label="member", density=True)
    plt.hist(nonmember_scores, bins=40, alpha=0.5, label="non-member", density=True)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _evaluate_one(
    *,
    name: str,
    model: LoRAModelWrapper,
    collator: LLMCollator,
    member_samples: List,
    nonmember_samples: List,
    device: str,
    batch_size: int,
    k_percent: float,
    show_progress: bool,
    out_dir: Path,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"name": name, "methods": {}}
    for method in ["loss", "min_k"]:
        res, debug = run_mia_detector(
            model=model,
            collator=collator,
            member_samples=member_samples,
            nonmember_samples=nonmember_samples,
            device=device,
            method=method,
            batch_size=batch_size,
            k_percent=k_percent,
            show_progress=show_progress,
            progress_prefix=f"{name} ",
        )
        out["methods"][method] = {
            "result": asdict(res),
            "debug": {k: v[:500] if isinstance(v, list) else v for k, v in debug.items()},
        }

        # Plot distribution on the membership score (higher => member).
        if method == "loss":
            mem_scores = [-x for x in debug.get("member_loss", [])]
            non_scores = [-x for x in debug.get("nonmember_loss", [])]
        else:
            mem_scores = [-x for x in debug.get("member_min_k_logprob", [])]
            non_scores = [-x for x in debug.get("nonmember_min_k_logprob", [])]
        _plot_hist(
            mem_scores,
            non_scores,
            title=f"{name} / {method} (auc_sym={res.auc_sym:.3f}, adv={res.adv:.3f})",
            out_path=out_dir / f"hist_{name}_{method}.png",
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dfl_snapshot", type=str, required=True)
    parser.add_argument("--dfu_dir", type=str, default=None, help="Optional DFU run dir (with final/agent_*/lora_state.pt).")
    parser.add_argument("--retrain_dir", type=str, default=None, help="Optional retrain run dir (with round_G/agent_*/lora_state.pt).")
    parser.add_argument("--target_agent", type=int, default=0)
    parser.add_argument("--eval_agent_id", type=int, default=1, help="Agent id to evaluate for DFL/DFU (original IDs).")
    parser.add_argument("--round", type=int, default=None, help="DFL round index to load (default: last available).")
    parser.add_argument("--max_samples", type=int, default=200, help="Max samples per side (member/non-member).")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--k_percent", type=float, default=20.0, help="Min-K%% for min_k detector.")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_test_samples", type=int, default=None)
    parser.add_argument(
        "--nonmember_source",
        type=str,
        default="test",
        choices=["test", "val", "retain"],
        help="Non-member pool: 'test' (default), held-out 'val', or historical 'retain' (train minus forget).",
    )
    parser.add_argument("--gpu", type=int, default=None, help="Logical GPU id within CUDA_VISIBLE_DEVICES (0/1).")
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--show_progress", action="store_true", default=False)
    args = parser.parse_args()

    # Enforce physical GPU 2/3 only.
    visible_physical = guard_gpu_or_raise(gpu=args.gpu)
    if args.gpu is not None:
        device = f"cuda:{args.gpu}"
        torch.cuda.set_device(args.gpu)
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"CUDA_VISIBLE_DEVICES (physical): {visible_physical}")

    # Reduce tokenizer thread noise.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTHONHASHSEED", "42")

    loader = SnapshotLoader(args.dfl_snapshot)
    cfg = loader.config
    dataset = cfg.get("dataset")
    if dataset not in DATASET_MAP:
        raise ValueError(f"Unsupported dataset for this script: {dataset!r}. Supported: {list(DATASET_MAP.keys())}")

    partition_path = Path(args.dfl_snapshot) / "partition.json"
    partition = PartitionInfo.load(str(partition_path))

    train_samples, val_samples, test_samples, _label_names = _load_train_test(
        dataset,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
    )

    forget_indices = sorted(partition.agent_indices[int(args.target_agent)])
    member_samples = [train_samples[i] for i in forget_indices]
    nonmember_source = str(args.nonmember_source).lower().strip()
    if nonmember_source == "val":
        nonmember_samples = list(val_samples)
    elif nonmember_source == "retain":
        forget_set = set(forget_indices)
        nonmember_samples = [s for i, s in enumerate(train_samples) if i not in forget_set]
    else:
        nonmember_samples = list(test_samples)

    member_samples, nonmember_samples = _select_equal_prefix(member_samples, nonmember_samples, int(args.max_samples))
    print(f"Member samples: {len(member_samples)} (agent {args.target_agent} train)")
    print(f"Non-member samples: {len(nonmember_samples)} ({nonmember_source})")

    # Build model + collator (TinyLlama LoRA wrapper).
    model = LoRAModelWrapper(lora_r=8, lora_alpha=16, device=device)
    model.load_base_model()
    model.init_lora()
    collator = LLMCollator(model.tokenizer, max_length=int(cfg.get("max_length") or 512))

    # Output directory
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = args.tag or f"{dataset}_agent{args.target_agent}_aid{args.eval_agent_id}_{ts}"
    out_dir = ROOT / ARTIFACT_ROOT / "unlearning_audit" / "mia" / tag
    _ensure_dir(out_dir)

    results: Dict[str, Any] = {
        "tag": tag,
        "dataset": dataset,
        "target_agent": int(args.target_agent),
        "eval_agent_id": int(args.eval_agent_id),
        "device": device,
        "dfl_snapshot": str(Path(args.dfl_snapshot).resolve()),
        "dfu_dir": None if args.dfu_dir is None else str(Path(args.dfu_dir).resolve()),
        "retrain_dir": None if args.retrain_dir is None else str(Path(args.retrain_dir).resolve()),
        "n_member": len(member_samples),
        "n_nonmember": len(nonmember_samples),
        "nonmember_source": nonmember_source,
        "detectors": {},
    }

    # ---------------- DFL ----------------
    round_idx = int(args.round) if args.round is not None else int(max(loader.available_rounds))
    dfl_state = loader.load_agent_state(round_idx=round_idx, agent_id=int(args.eval_agent_id), pre_agg=False)
    model.set_lora_state_dict(dfl_state)
    results["detectors"]["dfl"] = _evaluate_one(
        name="dfl",
        model=model,
        collator=collator,
        member_samples=member_samples,
        nonmember_samples=nonmember_samples,
        device=device,
        batch_size=int(args.batch_size),
        k_percent=float(args.k_percent),
        show_progress=bool(args.show_progress),
        out_dir=out_dir,
    )

    # --------------- DFU -----------------
    if args.dfu_dir:
        dfu_dir = Path(args.dfu_dir)
        state_path = _find_dfu_lora_state(dfu_dir, int(args.eval_agent_id))
        if state_path is None:
            # Fallback: try any agent state that exists.
            candidates = sorted(dfu_dir.glob("final/agent_*/lora_state.pt"))
            if not candidates:
                candidates = sorted(dfu_dir.glob("agent_*/lora_state.pt"))
            state_path = candidates[0] if candidates else None
        if state_path is None:
            print(f"[WARN] DFU lora_state.pt not found under {dfu_dir} (skipping dfu).")
        else:
            dfu_state = torch.load(state_path, map_location="cpu")
            model.set_lora_state_dict(dfu_state)
            results["detectors"]["dfu"] = _evaluate_one(
                name="dfu",
                model=model,
                collator=collator,
                member_samples=member_samples,
                nonmember_samples=nonmember_samples,
                device=device,
                batch_size=int(args.batch_size),
                k_percent=float(args.k_percent),
                show_progress=bool(args.show_progress),
                out_dir=out_dir,
            )
            results["dfu_state_path"] = str(state_path)

    # ------------- Retrain --------------
    if args.retrain_dir:
        retrain_dir = Path(args.retrain_dir)
        global_rounds = int(cfg.get("global_rounds") or 10)
        # When target_agent=0, original agent 1 maps to retrain agent 0.
        retrain_aid = int(args.eval_agent_id)
        if int(args.target_agent) == 0 and retrain_aid > 0:
            retrain_aid = retrain_aid - 1
        state_path = _find_retrain_lora_state(retrain_dir, round_idx=global_rounds, agent_id=retrain_aid)
        if state_path is None:
            print(f"[WARN] Retrain lora_state.pt not found under {retrain_dir} (skipping retrain).")
        else:
            retrain_state = torch.load(state_path, map_location="cpu")
            model.set_lora_state_dict(retrain_state)
            results["detectors"]["retrain"] = _evaluate_one(
                name="retrain",
                model=model,
                collator=collator,
                member_samples=member_samples,
                nonmember_samples=nonmember_samples,
                device=device,
                batch_size=int(args.batch_size),
                k_percent=float(args.k_percent),
                show_progress=bool(args.show_progress),
                out_dir=out_dir,
            )
            results["retrain_state_path"] = str(state_path)

    # Write JSON
    out_json = out_dir / "mia_audit.json"
    out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote: {out_json}")

    # Print a short console summary.
    def _print_one(name: str, method: str) -> None:
        det = results.get("detectors", {}).get(name, {}).get("methods", {}).get(method, {})
        res = det.get("result") or {}
        if not res:
            return
        print(
            f"{name}/{method}: auc={res.get('auc', 0):.3f} "
            f"auc_sym={res.get('auc_sym', 0):.3f} adv={res.get('adv', 0):.3f} "
            f"ks={res.get('ks_stat', 0):.3f} p={res.get('ks_pvalue', 1):.2e}"
        )

    print("\n=== Summary ===")
    for name in ["dfl", "dfu", "retrain"]:
        for method in ["loss", "min_k"]:
            _print_one(name, method)


if __name__ == "__main__":
    main()
