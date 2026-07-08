#!/usr/bin/env python3
"""Audit backdoor forgetting on DFL vs DFU vs Retrain checkpoints.

This script complements loss/min-k MIA by using a *behavioral* signal:
  - train-time: poison ONLY the forget client with (trigger -> target label)
  - audit-time: measure ASR on triggered inputs

Expected pattern for a good unlearning method:
  - DFL: high ASR (backdoor learned)
  - Retrain (oracle): low ASR
  - DFU (target): ASR close to retrain, while clean utility stays high
"""

from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_ROOT = Path(os.environ.get("LLMDFL_ARTIFACT_ROOT", "artifacts"))

import sys

sys.path.insert(0, str(ROOT))

from src.data.datasets import NewsGroupsDataset, YahooSubsetDataset  # noqa: E402
from src.data.collator import LLMCollator  # noqa: E402
from src.data.backdoor_wrapper import make_triggered_samples  # noqa: E402
from src.data.partitioner import DirichletPartitioner  # noqa: E402
from src.models.evaluator import Evaluator  # noqa: E402
from src.models.lora_model import LoRAModelWrapper  # noqa: E402
from src.dfu.snapshot_loader import SnapshotLoader  # noqa: E402
from src.utils.gpu_guard import guard_gpu_or_raise  # noqa: E402


DATASET_MAP = {
    "20newsgroups": NewsGroupsDataset,
    "yahoo_subset": YahooSubsetDataset,
}


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


def _find_retrain_state(retrain_dir: Path, *, agent_id: int, round_idx: Optional[int]) -> Optional[Path]:
    round_dirs = sorted([d for d in retrain_dir.iterdir() if d.is_dir() and d.name.startswith("round_")])
    if not round_dirs:
        return None
    if round_idx is None:
        round_dir = round_dirs[-1]
    else:
        round_dir = retrain_dir / f"round_{int(round_idx)}"
        if not round_dir.exists():
            return None
    p = round_dir / f"agent_{int(agent_id)}" / "lora_state.pt"
    return p if p.exists() else None


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _scan_agent_ids(root: Path) -> List[int]:
    ids = set()
    for parent in [root / "final", root]:
        if not parent.exists():
            continue
        for p in parent.glob("agent_*/lora_state.pt"):
            try:
                ids.add(int(p.parent.name.split("_", 1)[1]))
            except Exception:
                pass
    return sorted(ids)


def _normalize_weights(weights: Dict[int, float], agent_ids: List[int]) -> Dict[int, float]:
    filtered: Dict[int, float] = {}
    for aid in agent_ids:
        try:
            w = float(weights.get(int(aid), 0.0))
        except Exception:
            w = 0.0
        if w > 0.0:
            filtered[int(aid)] = w
    total = sum(filtered.values())
    if total <= 0.0:
        if not agent_ids:
            return {}
        uniform = 1.0 / float(len(agent_ids))
        return {int(aid): uniform for aid in agent_ids}
    return {int(aid): float(w / total) for aid, w in filtered.items()}


def _load_broadcast_lora_state(
    *,
    run_dir: Path,
    selected_agent_ids: List[int],
    available_agent_ids: List[int],
    mode: str,
    dfu_cfg: Dict[str, Any],
) -> Tuple[Optional[Dict[str, torch.Tensor]], Dict[int, float], Dict[int, str]]:
    if mode == "participant":
        return None, {}, {}

    source_ids = sorted({int(aid) for aid in (selected_agent_ids or available_agent_ids) if int(aid) in set(available_agent_ids)})
    if not source_ids:
        source_ids = list(available_agent_ids)
    if not source_ids:
        return None, {}, {}

    raw_weights: Dict[int, float] = {}
    if mode == "broadcast_weighted":
        for k, v in (dfu_cfg.get("selection_weights") or {}).items():
            try:
                raw_weights[int(k)] = float(v)
            except Exception:
                continue

    weights = _normalize_weights(raw_weights, source_ids)
    state_paths: Dict[int, str] = {}
    agg_state: Dict[str, torch.Tensor] = {}

    for aid in source_ids:
        state_path = _find_dfu_lora_state(run_dir, int(aid))
        if state_path is None:
            continue
        state_paths[int(aid)] = str(state_path)
        state = torch.load(state_path, map_location="cpu")
        weight = float(weights.get(int(aid), 0.0))
        for key, value in state.items():
            if not torch.is_tensor(value):
                continue
            tensor = value.detach().float()
            if key not in agg_state:
                agg_state[key] = tensor * weight
            else:
                agg_state[key] += tensor * weight

    if not agg_state or not state_paths:
        return None, {}, state_paths

    used_ids = sorted(state_paths.keys())
    used_weights = _normalize_weights(weights, used_ids)
    if set(used_ids) != set(source_ids):
        # Recompute with the actually loaded states only.
        agg_state = {}
        for aid in used_ids:
            state = torch.load(Path(state_paths[int(aid)]), map_location="cpu")
            weight = float(used_weights.get(int(aid), 0.0))
            for key, value in state.items():
                if not torch.is_tensor(value):
                    continue
                tensor = value.detach().float()
                if key not in agg_state:
                    agg_state[key] = tensor * weight
                else:
                    agg_state[key] += tensor * weight
        weights = used_weights
    return agg_state, weights, state_paths


def _retrain_agent_id(original_agent_id: int, target_agent: int) -> Optional[int]:
    """Map original DFL/DFU agent id to retrain id after removing target."""
    if int(original_agent_id) == int(target_agent):
        return None
    if int(original_agent_id) > int(target_agent):
        return int(original_agent_id) - 1
    return int(original_agent_id)


def _mean_std(values: List[float]) -> Tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    if len(values) == 1:
        return float(values[0]), 0.0
    mean = sum(values) / len(values)
    var = sum((x - mean) ** 2 for x in values) / len(values)
    return float(mean), float(var ** 0.5)


def _metric(entry: Dict[str, Any], family: str, key: str) -> Optional[float]:
    try:
        return float(((entry.get(family) or {}).get(key)))
    except Exception:
        return None


def _summarize_agent_results(name: str, agent_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build a main-table-compatible mean/best summary from per-agent audits."""
    summary: Dict[str, Any] = {
        "name": name,
        "n_agents": len(agent_results),
        "agent_ids": [int(x["agent_id"]) for x in agent_results],
        "agents": agent_results,
    }
    families = [
        ("clean", ["accuracy", "precision", "recall", "macro_f1", "valid_ratio"]),
        ("asr", ["asr", "valid_ratio"]),
        ("asr_non_target", ["asr", "valid_ratio"]),
        ("clean_target_rate_non_target", ["asr", "valid_ratio"]),
    ]
    for family, keys in families:
        summary[family] = {}
        for key in keys:
            vals = [_metric(x, family, key) for x in agent_results]
            vals = [v for v in vals if v is not None]
            m, s = _mean_std(vals)
            summary[family][key] = m
            summary[family][f"{key}_std"] = s
            if vals:
                summary[family][f"{key}_best"] = max(vals)
                summary[family][f"{key}_worst"] = min(vals)

    clean_f1 = [(_metric(x, "clean", "macro_f1"), x) for x in agent_results]
    clean_f1 = [(v, x) for v, x in clean_f1 if v is not None]
    if clean_f1:
        best_value, best_entry = max(clean_f1, key=lambda item: item[0])
        summary["best_agent_id"] = int(best_entry["agent_id"])
        summary["clean"]["macro_f1_best_agent"] = int(best_entry["agent_id"])
        summary["clean"]["macro_f1_best"] = float(best_value)
    return summary


def _load_train_test(
    dataset: str,
    *,
    max_train_samples: Optional[int],
    max_test_samples: Optional[int],
) -> Tuple[List, List, List[str]]:
    Dataset = DATASET_MAP[dataset]
    train_ds = Dataset(split="train", max_samples=max_train_samples)
    test_ds = Dataset(split="test", max_samples=max_test_samples)
    label_names = getattr(train_ds, "label_names", [])
    return train_ds.samples, test_ds.samples, label_names


def _load_dfl_train_split(
    dataset: str,
    *,
    max_train_samples: Optional[int],
) -> Tuple[List, List[str]]:
    """Load the exact training split shape used by scripts/train_dfl.py."""
    Dataset = DATASET_MAP[dataset]
    train_ds = Dataset(split="train", max_samples=max_train_samples)
    val_size = int(len(train_ds) * 0.1)
    return train_ds.samples[val_size:], getattr(train_ds, "label_names", [])


def _selected_poison_indices(indices: List[int], *, poison_rate: float, seed: int) -> List[int]:
    rng = random.Random(int(seed))
    selected: List[int] = []
    for idx in indices:
        if rng.random() < float(poison_rate):
            selected.append(int(idx))
    return selected


def _build_audit_subsets(
    *,
    dataset: str,
    cfg: Dict[str, Any],
    sample_source: str,
    target_agent: int,
    max_samples: int,
    fallback_train_samples: List,
    fallback_test_samples: List,
    trigger: str,
    trigger_position: str,
) -> Tuple[List, List, Dict[str, Any]]:
    """Build clean/triggered audit samples under a declared audit data source.

    public_test is the conventional backdoor-generalization test.
    target_train and target_poisoned_train are direct forgetting tests tied to
    the forgotten agent's local data.
    """
    source = str(sample_source or "public_test").strip().lower()
    meta: Dict[str, Any] = {
        "sample_source": source,
        "source_agent_id": None,
        "n_source_pool": None,
        "n_poison_selected": None,
    }

    if source == "public_test":
        clean_subset = list(fallback_test_samples)[:max_samples]
        triggered_subset = make_triggered_samples(
            clean_subset,
            trigger=str(trigger),
            position=str(trigger_position),
        )
        meta["n_source_pool"] = len(fallback_test_samples)
        return clean_subset, triggered_subset, meta

    if source not in {"target_train", "target_poisoned_train"}:
        raise ValueError(
            f"Unsupported --sample_source={sample_source!r}. "
            "Expected one of: public_test, target_train, target_poisoned_train"
        )

    train_max = cfg.get("max_train_samples")
    train_samples, _ = _load_dfl_train_split(
        dataset,
        max_train_samples=None if train_max is None else int(train_max),
    )
    if not train_samples:
        train_samples = list(fallback_train_samples)

    labels = [int(s.label) for s in train_samples]
    num_agents = int(cfg.get("num_agents") or cfg.get("K") or 10)
    alpha = float(cfg.get("alpha") if cfg.get("alpha") is not None else 0.5)
    seed = int(cfg.get("seed") if cfg.get("seed") is not None else 42)
    partition = DirichletPartitioner(num_agents=num_agents, alpha=alpha, seed=seed).partition(labels, dataset)
    if int(target_agent) not in partition.agent_indices:
        raise ValueError(f"target_agent={target_agent} not present in reconstructed partition.")

    source_indices = [int(i) for i in partition.agent_indices[int(target_agent)]]
    meta.update(
        {
            "source_agent_id": int(target_agent),
            "partition_num_agents": num_agents,
            "partition_alpha": alpha,
            "partition_seed": seed,
            "n_source_pool": len(source_indices),
        }
    )

    if source == "target_poisoned_train":
        poison_seed_cfg = cfg.get("backdoor_seed")
        poison_seed = int(poison_seed_cfg) if poison_seed_cfg is not None else seed
        poison_rate = float(cfg.get("backdoor_rate") or 0.0)
        source_indices = _selected_poison_indices(source_indices, poison_rate=poison_rate, seed=poison_seed)
        meta.update(
            {
                "backdoor_poison_rate": poison_rate,
                "backdoor_poison_seed": poison_seed,
                "n_poison_selected": len(source_indices),
            }
        )
        if not source_indices:
            raise ValueError("No poisoned target-agent samples were reconstructed for target_poisoned_train.")

    clean_subset = [train_samples[i] for i in source_indices[:max_samples]]
    triggered_subset = make_triggered_samples(
        clean_subset,
        trigger=str(trigger),
        position=str(trigger_position),
    )
    return clean_subset, triggered_subset, meta


def _eval_one(
    *,
    name: str,
    model: LoRAModelWrapper,
    collator: LLMCollator,
    clean_samples: List,
    triggered_samples: List,
    label_names: List[str],
    target_label: int,
    batch_size: int,
    max_new_tokens: int,
) -> Dict[str, Any]:
    evaluator = Evaluator(model, collator)
    clean_metrics = evaluator.evaluate_classification(
        clean_samples,
        label_names,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
    )
    asr_metrics = evaluator.evaluate_backdoor_asr(
        triggered_samples,
        label_names,
        target_label=int(target_label),
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
    )
    clean_non_target = [s for s in clean_samples if int(getattr(s, "label", -1)) != int(target_label)]
    triggered_non_target = [s for s in triggered_samples if int(getattr(s, "label", -1)) != int(target_label)]
    clean_target_rate = evaluator.evaluate_backdoor_asr(
        clean_non_target,
        label_names,
        target_label=int(target_label),
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
    )
    asr_non_target = evaluator.evaluate_backdoor_asr(
        triggered_non_target,
        label_names,
        target_label=int(target_label),
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
    )
    return {
        "name": name,
        "clean": clean_metrics,
        "asr": asr_metrics,
        "asr_non_target": asr_non_target,
        "clean_target_rate_non_target": clean_target_rate,
        "n_non_target": len(triggered_non_target),
    }


def _eval_current_model_for_agent(
    *,
    name: str,
    agent_id: int,
    model: LoRAModelWrapper,
    collator: LLMCollator,
    clean_samples: List,
    triggered_samples: List,
    label_names: List[str],
    target_label: int,
    batch_size: int,
    max_new_tokens: int,
    state_path: Optional[Path] = None,
) -> Dict[str, Any]:
    result = _eval_one(
        name=name,
        model=model,
        collator=collator,
        clean_samples=clean_samples,
        triggered_samples=triggered_samples,
        label_names=label_names,
        target_label=int(target_label),
        batch_size=int(batch_size),
        max_new_tokens=int(max_new_tokens),
    )
    result["agent_id"] = int(agent_id)
    if state_path is not None:
        result["state_path"] = str(state_path)
    return result


def _clone_agent_result(entry: Dict[str, Any], *, agent_id: int, state_path: Optional[str] = None) -> Dict[str, Any]:
    cloned = json.loads(json.dumps(entry))
    cloned["agent_id"] = int(agent_id)
    if state_path is not None:
        cloned["state_path"] = str(state_path)
    return cloned


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dfl_snapshot", type=str, required=True)
    parser.add_argument("--dfu_dir", type=str, default=None)
    parser.add_argument("--retrain_dir", type=str, default=None)
    parser.add_argument("--target_agent", type=int, default=0)
    parser.add_argument("--eval_agent_id", type=int, default=1)
    parser.add_argument(
        "--eval_scope",
        choices=["single", "selected", "all"],
        default="single",
        help=(
            "single: evaluate --eval_agent_id only; selected: evaluate DFU selected agents "
            "(or all surviving for full strategies); all: evaluate all available surviving agents."
        ),
    )
    parser.add_argument(
        "--dfu_state_mode",
        choices=["participant", "broadcast_uniform", "broadcast_weighted"],
        default="participant",
        help=(
            "participant: evaluate each saved DFU agent state separately; "
            "broadcast_uniform/broadcast_weighted: aggregate the saved DFU states into one deployed adapter "
            "and evaluate that common adapter on the requested surviving-agent scope."
        ),
    )
    parser.add_argument(
        "--dfl_eval_scope",
        choices=["same_as_dfu", "all_surviving"],
        default="same_as_dfu",
        help=(
            "same_as_dfu: evaluate DFL on the same agent ids used for DFU; "
            "all_surviving: evaluate DFL on all agents except the target, giving one fixed DFL reference."
        ),
    )
    parser.add_argument("--round", type=int, default=None, help="DFL round index to load (default: last available).")
    parser.add_argument("--trigger", type=str, required=True, help="Trigger string to inject at eval time.")
    parser.add_argument(
        "--trigger_position",
        type=str,
        default=None,
        choices=["prefix", "suffix"],
        help="Trigger position for audit-time injection (default: checkpoint backdoor_position or prefix).",
    )
    parser.add_argument("--target_label", type=int, required=True, help="Target label index for ASR computation.")
    parser.add_argument("--max_samples", type=int, default=500, help="Max eval samples (clean + triggered).")
    parser.add_argument(
        "--sample_source",
        type=str,
        default="public_test",
        choices=["public_test", "target_train", "target_poisoned_train"],
        help=(
            "public_test: held-out public test samples with trigger; "
            "target_train: forgotten agent's local training samples with trigger; "
            "target_poisoned_train: the exact forgotten-agent samples selected for poisoning during DFL, "
            "re-triggered with original labels for direct forgetting ASR."
        ),
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_test_samples", type=int, default=None)
    parser.add_argument("--gpu", type=int, default=None, help="Logical GPU id within CUDA_VISIBLE_DEVICES (0/1).")
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument(
        "--audit_models",
        type=str,
        default="dfl,dfu,retrain",
        help="Comma-separated subset of models to evaluate: dfl,dfu,retrain",
    )
    args = parser.parse_args()

    visible_physical = guard_gpu_or_raise(gpu=args.gpu)
    if args.gpu is not None:
        device = f"cuda:{args.gpu}"
        torch.cuda.set_device(args.gpu)
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"CUDA_VISIBLE_DEVICES (physical): {visible_physical}")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    loader = SnapshotLoader(args.dfl_snapshot)
    cfg = loader.config
    dataset = cfg.get("dataset")
    if dataset not in DATASET_MAP:
        raise ValueError(f"Unsupported dataset for backdoor audit: {dataset!r}. Supported: {list(DATASET_MAP.keys())}")

    train_samples, test_samples, label_names = _load_train_test(
        dataset,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
    )
    if not label_names:
        raise ValueError("label_names missing; backdoor audit requires classification datasets.")

    max_samples = max(1, int(args.max_samples))
    trigger_position = str(args.trigger_position or cfg.get("backdoor_position") or "prefix")
    clean_subset, triggered_subset, source_meta = _build_audit_subsets(
        dataset=dataset,
        cfg=cfg,
        sample_source=str(args.sample_source),
        target_agent=int(args.target_agent),
        max_samples=max_samples,
        fallback_train_samples=train_samples,
        fallback_test_samples=test_samples,
        trigger=str(args.trigger),
        trigger_position=trigger_position,
    )

    # Model + collator
    model = LoRAModelWrapper(lora_r=int(cfg.get("lora_r") or 8), lora_alpha=int(cfg.get("lora_alpha") or 16), device=device)
    model.load_base_model()
    model.init_lora()

    max_length = int(cfg.get("max_length") or 512)
    collator = LLMCollator(model.tokenizer, max_length=max_length, inference_mode=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = args.tag or f"{dataset}_bd_{ts}"
    out_dir = ROOT / ARTIFACT_ROOT / "unlearning_audit" / "backdoor" / tag
    _ensure_dir(out_dir)

    results: Dict[str, Any] = {
        "tag": tag,
        "dataset": dataset,
        "trigger": str(args.trigger),
        "trigger_position": trigger_position,
        "target_label": int(args.target_label),
        "target_label_name": label_names[int(args.target_label)] if 0 <= int(args.target_label) < len(label_names) else None,
        "sample_source": str(args.sample_source),
        "source_meta": source_meta,
        "device": device,
        "dfl_snapshot": str(Path(args.dfl_snapshot).resolve()),
        "dfu_dir": None if args.dfu_dir is None else str(Path(args.dfu_dir).resolve()),
        "retrain_dir": None if args.retrain_dir is None else str(Path(args.retrain_dir).resolve()),
        "n_clean": len(clean_subset),
        "n_triggered": len(triggered_subset),
        "models": {},
    }

    requested_models = {
        item.strip().lower()
        for item in str(args.audit_models or "").split(",")
        if item.strip()
    }
    valid_models = {"dfl", "dfu", "retrain"}
    invalid_models = sorted(requested_models - valid_models)
    if invalid_models:
        raise ValueError(f"Unsupported --audit_models entries: {invalid_models}; valid={sorted(valid_models)}")
    if not requested_models:
        requested_models = set(valid_models)
    results["audit_models"] = sorted(requested_models)

    round_idx = int(args.round) if args.round is not None else int(max(loader.available_rounds))
    surviving_agent_ids = [i for i in range(int(loader.snapshot.num_agents)) if i != int(args.target_agent)]
    broadcast_mode = str(args.dfu_state_mode) != "participant"
    dfu_cfg: Dict[str, Any] = {}

    dfu_eval_ids: List[int] = []
    if args.dfu_dir and "dfu" in requested_models:
        dfu_cfg = _read_json(Path(args.dfu_dir) / "dfu_config.json") or {}
        selected = [int(x) for x in (dfu_cfg.get("selected_agents") or []) if int(x) != int(args.target_agent)]
        available = _scan_agent_ids(Path(args.dfu_dir))
        if args.eval_scope == "single":
            dfu_eval_ids = [int(args.eval_agent_id)]
        elif args.eval_scope == "selected" and not broadcast_mode:
            dfu_eval_ids = selected or available or [int(args.eval_agent_id)]
        else:
            dfu_eval_ids = surviving_agent_ids if broadcast_mode else (available or surviving_agent_ids)
    elif args.eval_scope == "single":
        dfu_eval_ids = [int(args.eval_agent_id)]
    else:
        dfu_eval_ids = surviving_agent_ids

    if str(args.dfl_eval_scope) == "all_surviving":
        dfl_eval_ids = list(surviving_agent_ids)
    elif broadcast_mode and args.eval_scope != "single":
        # In broadcast/deployed mode, use the full surviving set as the DFL baseline so Base/AS/LS/DSU
        # share the same pre-unlearning reference population.
        dfl_eval_ids = list(surviving_agent_ids)
    else:
        # Use the same original agent ids for DFL so DFL/DFU clean F1 are comparable.
        dfl_eval_ids = [aid for aid in dfu_eval_ids if aid in surviving_agent_ids]
        if not dfl_eval_ids:
            dfl_eval_ids = [int(args.eval_agent_id)]

    results["eval_scope"] = str(args.eval_scope)
    results["eval_agent_ids"] = dfl_eval_ids
    results["dfu_eval_agent_ids"] = list(dfu_eval_ids)
    results["dfu_state_mode"] = str(args.dfu_state_mode)
    results["dfl_eval_scope"] = str(args.dfl_eval_scope)

    # ---------------- DFL ----------------
    if "dfl" in requested_models:
        print(f"[audit] Evaluating DFL on agent ids: {dfl_eval_ids}")
        dfl_agent_results: List[Dict[str, Any]] = []
        for aid in dfl_eval_ids:
            dfl_state = loader.load_agent_state(round_idx=round_idx, agent_id=int(aid), pre_agg=False)
            model.set_lora_state_dict(dfl_state)
            dfl_agent_results.append(
                _eval_current_model_for_agent(
                    name="dfl",
                    agent_id=int(aid),
                    model=model,
                    collator=collator,
                    clean_samples=clean_subset,
                    triggered_samples=triggered_subset,
                    label_names=label_names,
                    target_label=int(args.target_label),
                    batch_size=int(args.batch_size),
                    max_new_tokens=int(args.max_new_tokens),
                )
            )
        results["models"]["dfl"] = _summarize_agent_results("dfl", dfl_agent_results)

    # ---------------- DFU ----------------
    if args.dfu_dir and "dfu" in requested_models:
        print(f"[audit] Evaluating DFU on agent ids: {dfu_eval_ids} (mode={args.dfu_state_mode})")
        dfu_dir = Path(args.dfu_dir)
        dfu_agent_results: List[Dict[str, Any]] = []
        state_paths: Dict[str, str] = {}
        if broadcast_mode:
            selected = [int(x) for x in (dfu_cfg.get("selected_agents") or []) if int(x) != int(args.target_agent)]
            available = _scan_agent_ids(dfu_dir)
            dfu_state, broadcast_weights, broadcast_sources = _load_broadcast_lora_state(
                run_dir=dfu_dir,
                selected_agent_ids=selected,
                available_agent_ids=available,
                mode=str(args.dfu_state_mode),
                dfu_cfg=dfu_cfg,
            )
            if dfu_state is None:
                print(f"[WARN] Could not build broadcast DFU state under {dfu_dir} (mode={args.dfu_state_mode}).")
            else:
                model.set_lora_state_dict(dfu_state)
                template = _eval_current_model_for_agent(
                    name="dfu",
                    agent_id=int(dfu_eval_ids[0]) if dfu_eval_ids else int(args.eval_agent_id),
                    model=model,
                    collator=collator,
                    clean_samples=clean_subset,
                    triggered_samples=triggered_subset,
                    label_names=label_names,
                    target_label=int(args.target_label),
                    batch_size=int(args.batch_size),
                    max_new_tokens=int(args.max_new_tokens),
                )
                state_label = f"[{args.dfu_state_mode}:" + ",".join(f"{aid}:{broadcast_weights.get(aid, 0.0):.6f}" for aid in sorted(broadcast_weights)) + "]"
                for aid in dfu_eval_ids:
                    dfu_agent_results.append(_clone_agent_result(template, agent_id=int(aid), state_path=state_label))
                state_paths = {str(aid): path for aid, path in broadcast_sources.items()}
                results["dfu_broadcast_weights"] = {str(aid): float(w) for aid, w in broadcast_weights.items()}
                results["dfu_broadcast_source_agent_ids"] = sorted(int(aid) for aid in broadcast_weights)
        else:
            for aid in dfu_eval_ids:
                state_path = _find_dfu_lora_state(dfu_dir, int(aid))
                if state_path is None:
                    print(f"[WARN] DFU lora_state.pt not found under {dfu_dir} for agent {aid} (skipping).")
                    continue
                dfu_state = torch.load(state_path, map_location="cpu")
                model.set_lora_state_dict(dfu_state)
                dfu_agent_results.append(
                    _eval_current_model_for_agent(
                        name="dfu",
                        agent_id=int(aid),
                        model=model,
                        collator=collator,
                        clean_samples=clean_subset,
                        triggered_samples=triggered_subset,
                        label_names=label_names,
                        target_label=int(args.target_label),
                        batch_size=int(args.batch_size),
                        max_new_tokens=int(args.max_new_tokens),
                        state_path=state_path,
                    )
                )
                state_paths[str(aid)] = str(state_path)
        if dfu_agent_results:
            results["models"]["dfu"] = _summarize_agent_results("dfu", dfu_agent_results)
            results["dfu_state_paths"] = state_paths
            if str(args.eval_agent_id) in state_paths:
                results["dfu_state_path"] = state_paths[str(args.eval_agent_id)]

    # --------------- Retrain -------------
    if args.retrain_dir and "retrain" in requested_models:
        print(f"[audit] Evaluating Retrain on original agent ids: {dfl_eval_ids}")
        retrain_dir = Path(args.retrain_dir)
        global_rounds = int(cfg.get("global_rounds") or 10)
        retrain_agent_results: List[Dict[str, Any]] = []
        state_paths: Dict[str, str] = {}
        for original_aid in dfl_eval_ids:
            retrain_aid = _retrain_agent_id(int(original_aid), int(args.target_agent))
            if retrain_aid is None:
                continue
            state_path = _find_retrain_state(retrain_dir, agent_id=retrain_aid, round_idx=global_rounds)
            if state_path is None:
                state_path = _find_retrain_state(retrain_dir, agent_id=retrain_aid, round_idx=None)
            if state_path is None:
                print(f"[WARN] Retrain lora_state.pt not found under {retrain_dir} for original agent {original_aid} -> retrain agent {retrain_aid} (skipping).")
                continue
            retrain_state = torch.load(state_path, map_location="cpu")
            model.set_lora_state_dict(retrain_state)
            entry = _eval_current_model_for_agent(
                name="retrain",
                agent_id=int(original_aid),
                model=model,
                collator=collator,
                clean_samples=clean_subset,
                triggered_samples=triggered_subset,
                label_names=label_names,
                target_label=int(args.target_label),
                batch_size=int(args.batch_size),
                max_new_tokens=int(args.max_new_tokens),
                state_path=state_path,
            )
            entry["retrain_agent_id"] = int(retrain_aid)
            retrain_agent_results.append(entry)
            state_paths[str(original_aid)] = str(state_path)
        if retrain_agent_results:
            results["models"]["retrain"] = _summarize_agent_results("retrain", retrain_agent_results)
            results["retrain_state_paths"] = state_paths
            if str(args.eval_agent_id) in state_paths:
                results["retrain_state_path"] = state_paths[str(args.eval_agent_id)]

    out_json = out_dir / "backdoor_audit.json"
    out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote: {out_json}")

    print("\n=== Summary ===")
    for k in ["dfl", "dfu", "retrain"]:
        if k not in results["models"]:
            continue
        entry = results["models"][k]
        clean = entry.get("clean", {})
        asr = entry.get("asr", {})
        asr_nt = entry.get("asr_non_target", {})
        clean_target = entry.get("clean_target_rate_non_target", {})
        print(
            f"{k}: clean_acc={clean.get('accuracy', 0):.3f} macro_f1={clean.get('macro_f1', 0):.3f} "
            f"ASR={asr.get('asr', 0):.3f} (valid={asr.get('valid_ratio', 0):.2f}) "
            f"ASR_non_target={asr_nt.get('asr', 0):.3f} (valid={asr_nt.get('valid_ratio', 0):.2f}) "
            f"clean_target_non_target={clean_target.get('asr', 0):.3f}"
        )


if __name__ == "__main__":
    main()
