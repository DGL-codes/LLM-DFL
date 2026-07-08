"""Unified evaluation helpers for multi-agent LoRA checkpoints.

This module centralizes:
- deterministic eval subset selection (prefix slice)
- per-agent metric computation (classification / generation)
- aggregation across selected-k agents (mean/std) and best-agent selection
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from ..data.base import Sample
from ..data.collator import LLMCollator
from .evaluator import Evaluator
from .lora_model import LoRAModelWrapper


CLASSIFICATION_METRIC_KEYS = ["accuracy", "precision", "recall", "macro_f1", "valid_ratio"]
GENERATION_METRIC_KEYS = ["exact_match", "token_f1"]


def _as_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, (np.floating, np.integer)):
        v = float(v)
    if isinstance(v, (int, float)):
        fv = float(v)
        return None if math.isnan(fv) else fv
    return None


def select_eval_subset(samples: List[Sample], max_eval_samples: Optional[int]) -> List[Sample]:
    """Deterministically select the evaluation subset (prefix slice)."""
    if max_eval_samples is None:
        return samples
    try:
        limit = int(max_eval_samples)
    except (TypeError, ValueError):
        return samples
    if limit <= 0:
        return samples
    return samples[: min(limit, len(samples))]


def _infer_primary_metric(per_agent: Dict[int, Dict[str, Any]]) -> str:
    if any(isinstance(m, dict) and "macro_f1" in m for m in per_agent.values()):
        return "macro_f1"
    if any(isinstance(m, dict) and "token_f1" in m for m in per_agent.values()):
        return "token_f1"
    return "macro_f1"


def _infer_metric_keys(per_agent: Dict[int, Dict[str, Any]], *, primary_metric: str) -> List[str]:
    if primary_metric == "macro_f1":
        if any(isinstance(m, dict) and "macro_f1" in m for m in per_agent.values()):
            return list(CLASSIFICATION_METRIC_KEYS)
    if primary_metric == "token_f1":
        if any(isinstance(m, dict) and "token_f1" in m for m in per_agent.values()):
            return list(GENERATION_METRIC_KEYS)

    # Fallback: collect numeric keys (stable order).
    keys: List[str] = []
    seen = set()
    for m in per_agent.values():
        if not isinstance(m, dict):
            continue
        for k, v in m.items():
            if k in seen:
                continue
            if _as_float(v) is None:
                continue
            seen.add(k)
            keys.append(str(k))
    return keys


def select_best_agent(
    per_agent: Dict[int, Dict[str, Any]],
    *,
    primary_metric: str,
    eval_agent_ids: Optional[Iterable[int]] = None,
) -> Optional[int]:
    """Return the best agent id by `primary_metric` (ties -> smaller agent id)."""
    if not per_agent:
        return None

    candidate_ids = list(eval_agent_ids) if eval_agent_ids is not None else list(per_agent.keys())
    candidate_ids = [int(a) for a in candidate_ids if int(a) in per_agent]
    if not candidate_ids:
        candidate_ids = list(per_agent.keys())

    best_id: Optional[int] = None
    best_score = float("-inf")
    for aid in sorted(candidate_ids):
        score = _as_float(per_agent[int(aid)].get(primary_metric))
        if score is None:
            score = float("-inf")
        if (best_id is None) or (score > best_score):
            best_id = int(aid)
            best_score = float(score)
    return best_id


def summarize_per_agent_metrics(
    per_agent: Dict[int, Dict[str, Any]],
    *,
    primary_metric: Optional[str] = None,
) -> Tuple[Dict[str, float], Dict[str, float], Optional[int], Dict[str, float], List[str]]:
    """Compute mean/std and best-agent metrics for the given per-agent metrics."""
    if not per_agent:
        return {}, {}, None, {}, []

    primary = _infer_primary_metric(per_agent) if primary_metric is None else str(primary_metric)
    metric_keys = _infer_metric_keys(per_agent, primary_metric=primary)

    mean_metrics: Dict[str, float] = {}
    std_metrics: Dict[str, float] = {}
    for key in metric_keys:
        values = [_as_float(m.get(key)) for m in per_agent.values() if isinstance(m, dict)]
        values = [v for v in values if v is not None]
        mean_metrics[key] = float(np.mean(values)) if values else 0.0
        std_metrics[key] = float(np.std(values)) if values else 0.0

    best_agent_id = select_best_agent(per_agent, primary_metric=primary)
    best_metrics: Dict[str, float] = {}
    if best_agent_id is not None:
        best_raw = per_agent.get(int(best_agent_id), {})
        for key in metric_keys:
            v = _as_float(best_raw.get(key)) if isinstance(best_raw, dict) else None
            best_metrics[key] = float(v) if v is not None else 0.0

    return mean_metrics, std_metrics, best_agent_id, best_metrics, metric_keys


def build_final_stats(
    per_agent: Dict[int, Dict[str, Any]],
    *,
    primary_metric: Optional[str] = None,
    eval_agent_ids: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Build a JSON-friendly `final_stats` dict from per-agent metrics."""
    eval_ids = sorted([int(x) for x in (eval_agent_ids or list(per_agent.keys())) if int(x) in per_agent])
    mean_metrics, std_metrics, best_agent_id, best_metrics, metric_keys = summarize_per_agent_metrics(
        {int(k): v for k, v in per_agent.items()},
        primary_metric=primary_metric,
    )

    out: Dict[str, Any] = {}
    for k in metric_keys:
        out[f"{k}_mean"] = float(mean_metrics.get(k, 0.0))
        out[f"{k}_std"] = float(std_metrics.get(k, 0.0))
        out[f"{k}_best"] = float(best_metrics.get(k, 0.0))

    out["best_agent_id"] = None if best_agent_id is None else int(best_agent_id)
    out["best_metric_key"] = _infer_primary_metric(per_agent) if primary_metric is None else str(primary_metric)
    out["eval_agent_ids"] = eval_ids
    out["num_eval_agents"] = len(eval_ids)
    out["per_agent"] = {
        str(int(aid)): {str(k): float(v) for k, v in metrics.items() if _as_float(v) is not None}
        for aid, metrics in sorted(per_agent.items(), key=lambda kv: int(kv[0]))
        if isinstance(metrics, dict)
    }
    return out


def evaluate_lora_states(
    *,
    model: LoRAModelWrapper,
    train_collator: LLMCollator,
    task_type: str,
    label_names: Optional[List[str]],
    test_samples: List[Sample],
    lora_states: Dict[int, Dict],
    eval_agent_ids: List[int],
    max_eval_samples: Optional[int] = None,
    batch_size: int = 16,
    max_new_tokens_classification: int = 16,
    max_new_tokens_generation: int = 64,
) -> Tuple[Dict[int, Dict[str, float]], Dict[str, Any]]:
    """Evaluate a set of LoRA states on a deterministic test subset.

    Returns:
      - per_agent: agent_id -> metrics
      - summary: JSON-friendly dict containing mean/std/best/per_agent fields
    """
    eval_ids = [int(x) for x in eval_agent_ids]
    eval_subset = select_eval_subset(test_samples, max_eval_samples)

    # IMPORTANT: don't reuse the training collator (Evaluator mutates inference_mode).
    eval_collator = LLMCollator(
        tokenizer=train_collator.tokenizer,
        max_length=train_collator.max_length,
        inference_mode=True,
    )
    evaluator = Evaluator(model, eval_collator)

    per_agent: Dict[int, Dict[str, float]] = {}
    for aid in eval_ids:
        if aid not in lora_states:
            raise KeyError(f"Missing LoRA state for agent {aid}")
        model.set_lora_state_dict(lora_states[aid])

        if (task_type or "classification") == "classification":
            if not label_names:
                raise ValueError("label_names is required for classification evaluation")
            metrics = evaluator.evaluate_classification(
                eval_subset,
                label_names,
                batch_size=batch_size,
                max_new_tokens=max_new_tokens_classification,
            )
            per_agent[int(aid)] = {k: float(v) for k, v in metrics.items()}
        else:
            metrics = evaluator.evaluate_generation(
                eval_subset,
                batch_size=batch_size,
                max_new_tokens=max_new_tokens_generation,
            )
            per_agent[int(aid)] = {k: float(v) for k, v in metrics.items()}

    primary = "macro_f1" if (task_type or "classification") == "classification" else "token_f1"
    summary = build_final_stats(per_agent, primary_metric=primary, eval_agent_ids=eval_ids)
    return per_agent, summary

