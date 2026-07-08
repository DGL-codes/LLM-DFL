"""Detector utilities for unlearning auditing.

This module provides lightweight, self-contained evaluation helpers for:
- loss-based membership inference (MIA)
- Min-K% membership inference (Shi et al.)
- distribution separation diagnostics (AUC symmetry + KS statistic)

The goal is to *audit the detector itself* before using it to claim unlearning:
if a detector cannot distinguish members vs non-members on the original DFL
model, it is not a meaningful unlearning verifier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

from ..data.base import Sample
from ..data.collator import LLMCollator
from ..models.lora_model import LoRAModelWrapper


@dataclass(frozen=True)
class MIADetectorResult:
    method: str
    auc: float
    auc_sym: float
    adv: float
    tpr_at_1fpr: float
    tpr_at_5fpr: float
    ks_stat: float
    ks_pvalue: float
    n_member: int
    n_nonmember: int


def _as_floats(xs: List[Any]) -> List[float]:
    out: List[float] = []
    for x in xs:
        if x is None:
            continue
        if isinstance(x, (np.floating, np.integer)):
            x = float(x)
        if isinstance(x, (int, float)):
            xf = float(x)
            if np.isnan(xf):
                continue
            out.append(xf)
    return out


def compute_sample_loss(
    *,
    model: LoRAModelWrapper,
    collator: LLMCollator,
    samples: List[Sample],
    device: str,
    batch_size: int = 8,
    show_progress: bool = False,
    progress_desc: str = "loss",
) -> List[float]:
    """Compute per-sample mean token loss (cross-entropy)."""
    model.model.eval()
    losses: List[float] = []

    num_batches = (len(samples) + batch_size - 1) // batch_size if samples else 0
    batch_iter: Any = range(0, len(samples), batch_size)
    if show_progress:
        batch_iter = tqdm(batch_iter, desc=progress_desc, total=num_batches, dynamic_ncols=True)

    with torch.no_grad():
        for i in batch_iter:
            batch_samples = samples[i : i + batch_size]
            batch = collator(batch_samples)
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model.model(**batch)

            logits = outputs.logits[:, :-1, :].contiguous()
            labels = batch["labels"][:, 1:].contiguous()

            for j in range(len(batch_samples)):
                sample_logits = logits[j]
                sample_labels = labels[j]
                mask = sample_labels != -100
                if mask.sum() == 0:
                    losses.append(float("nan"))
                    continue
                sample_logits = sample_logits[mask]
                sample_labels = sample_labels[mask]
                loss = torch.nn.functional.cross_entropy(sample_logits, sample_labels, reduction="mean")
                losses.append(float(loss.item()))

    return losses


def compute_min_k_logprob(
    *,
    model: LoRAModelWrapper,
    collator: LLMCollator,
    samples: List[Sample],
    device: str,
    k_percent: float = 20.0,
    batch_size: int = 8,
    show_progress: bool = False,
    progress_desc: str = "min-k",
) -> List[float]:
    """Compute Min-K% mean log-probability for membership inference.

    Returns a per-sample scalar: mean(log p(token)) over the K% tokens with the
    lowest log-probabilities (i.e., most surprising tokens).
    """
    model.model.eval()
    scores: List[float] = []

    num_batches = (len(samples) + batch_size - 1) // batch_size if samples else 0
    batch_iter: Any = range(0, len(samples), batch_size)
    if show_progress:
        batch_iter = tqdm(batch_iter, desc=progress_desc, total=num_batches, dynamic_ncols=True)

    with torch.no_grad():
        for i in batch_iter:
            batch_samples = samples[i : i + batch_size]
            batch = collator(batch_samples)
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model.model(**batch)
            logits = outputs.logits[:, :-1, :]
            labels = batch["labels"][:, 1:]

            log_probs = torch.log_softmax(logits, dim=-1)
            for j in range(len(batch_samples)):
                sample_labels = labels[j]
                mask = sample_labels != -100
                if mask.sum() == 0:
                    scores.append(float("nan"))
                    continue
                sample_log_probs = log_probs[j][mask]
                sample_labels_masked = sample_labels[mask]
                token_log_probs = sample_log_probs.gather(1, sample_labels_masked.unsqueeze(1)).squeeze(1)

                k = max(1, int(len(token_log_probs) * float(k_percent) / 100.0))
                min_k_vals, _ = torch.topk(token_log_probs, k, largest=False)
                scores.append(float(min_k_vals.mean().item()))

    return scores


def compute_mia_from_scores(
    *,
    member_scores: List[float],
    nonmember_scores: List[float],
    method: str,
) -> MIADetectorResult:
    """Compute ROC/AUC metrics given membership scores (higher => more member)."""
    mem = _as_floats(member_scores)
    non = _as_floats(nonmember_scores)

    labels = [1] * len(mem) + [0] * len(non)
    scores = mem + non
    if len(set(labels)) < 2 or len(scores) < 2:
        return MIADetectorResult(
            method=str(method),
            auc=0.5,
            auc_sym=0.5,
            adv=0.0,
            tpr_at_1fpr=0.0,
            tpr_at_5fpr=0.0,
            ks_stat=0.0,
            ks_pvalue=1.0,
            n_member=len(mem),
            n_nonmember=len(non),
        )

    auc = float(roc_auc_score(labels, scores))
    auc_sym = float(max(auc, 1.0 - auc))
    adv = float(auc_sym - 0.5)

    fpr, tpr, _ = roc_curve(labels, scores)
    # Guard for short arrays.
    tpr_at_1fpr = float(tpr[min(np.searchsorted(fpr, 0.01), len(tpr) - 1)]) if len(tpr) else 0.0
    tpr_at_5fpr = float(tpr[min(np.searchsorted(fpr, 0.05), len(tpr) - 1)]) if len(tpr) else 0.0

    # KS test for distribution separation (members vs non-members).
    ks = ks_2samp(mem, non, alternative="two-sided", mode="auto")
    ks_stat = float(ks.statistic)
    ks_pvalue = float(ks.pvalue)

    return MIADetectorResult(
        method=str(method),
        auc=auc,
        auc_sym=auc_sym,
        adv=adv,
        tpr_at_1fpr=tpr_at_1fpr,
        tpr_at_5fpr=tpr_at_5fpr,
        ks_stat=ks_stat,
        ks_pvalue=ks_pvalue,
        n_member=len(mem),
        n_nonmember=len(non),
    )


def run_mia_detector(
    *,
    model: LoRAModelWrapper,
    collator: LLMCollator,
    member_samples: List[Sample],
    nonmember_samples: List[Sample],
    device: str,
    method: str = "loss",
    batch_size: int = 8,
    k_percent: float = 20.0,
    show_progress: bool = False,
    progress_prefix: str = "",
) -> Tuple[MIADetectorResult, Dict[str, Any]]:
    """Run a MIA detector and return (result, debug_payload)."""
    method = str(method).lower().strip()

    if method == "loss":
        mem_losses = compute_sample_loss(
            model=model,
            collator=collator,
            samples=member_samples,
            device=device,
            batch_size=batch_size,
            show_progress=show_progress,
            progress_desc=f"{progress_prefix}member loss",
        )
        non_losses = compute_sample_loss(
            model=model,
            collator=collator,
            samples=nonmember_samples,
            device=device,
            batch_size=batch_size,
            show_progress=show_progress,
            progress_desc=f"{progress_prefix}nonmember loss",
        )
        # Lower loss => more member.
        mem_scores = [-x for x in mem_losses]
        non_scores = [-x for x in non_losses]
        result = compute_mia_from_scores(member_scores=mem_scores, nonmember_scores=non_scores, method=method)
        debug = {"member_loss": _as_floats(mem_losses), "nonmember_loss": _as_floats(non_losses)}
        return result, debug

    if method in {"min_k", "mink", "min-k"}:
        mem_mk = compute_min_k_logprob(
            model=model,
            collator=collator,
            samples=member_samples,
            device=device,
            k_percent=k_percent,
            batch_size=batch_size,
            show_progress=show_progress,
            progress_desc=f"{progress_prefix}member min-k",
        )
        non_mk = compute_min_k_logprob(
            model=model,
            collator=collator,
            samples=nonmember_samples,
            device=device,
            k_percent=k_percent,
            batch_size=batch_size,
            show_progress=show_progress,
            progress_desc=f"{progress_prefix}nonmember min-k",
        )
        # Lower min-k logprob => more member (convention); invert to membership score.
        mem_scores = [-x for x in mem_mk]
        non_scores = [-x for x in non_mk]
        result = compute_mia_from_scores(member_scores=mem_scores, nonmember_scores=non_scores, method="min_k")
        debug = {"member_min_k_logprob": _as_floats(mem_mk), "nonmember_min_k_logprob": _as_floats(non_mk)}
        return result, debug

    raise ValueError(f"Unknown detector method: {method!r} (expected: 'loss' or 'min_k')")

