"""Unlearning verification metrics for DFU.

The primary goal is to quantify whether the model still *memorizes* samples that
should be forgotten.

Detector philosophy (LLM membership inference, classification-as-generation):
  - members: samples from the target agent's TRAIN split (the forget set)
  - non-members: samples not used in training (val/test)
  - a good unlearning method should make members indistinguishable from
    non-members (AUC_sym -> 0.5)

Notes
-----
Historically, this repo used a "forget vs retain" loss gap and sometimes reused
retain TRAIN samples as the "non-member" pool, which is not a proper membership
test (both sets are members in the original training). We now support passing an
explicit `nonmember_samples` pool to compute meaningful MIA.
"""
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.stats import ks_2samp

from ..data.base import Sample
from ..data.collator import LLMCollator
from ..models.lora_model import LoRAModelWrapper


@dataclass
class UnlearningMetrics:
    """Metrics for unlearning verification."""
    # MIA metrics
    mia_auc: float  # AUC for membership inference
    mia_tpr_at_1fpr: float  # TPR at 1% FPR
    mia_tpr_at_5fpr: float  # TPR at 5% FPR
    
    # Loss-based metrics
    forget_loss: float  # Average loss on forget set
    retain_loss: float  # Average loss on retain set
    loss_gap: float  # forget_loss - retain_loss (higher = better unlearning)
    
    # Accuracy metrics (for classification)
    forget_accuracy: Optional[float] = None
    retain_accuracy: Optional[float] = None
    accuracy_gap: Optional[float] = None  # retain_acc - forget_acc

    # Extra detector diagnostics (optional; used for auditing)
    mia_auc_sym: Optional[float] = None  # max(auc, 1-auc) to ignore direction
    mia_adv: Optional[float] = None  # mia_auc_sym - 0.5
    mia_ks_stat: Optional[float] = None
    mia_ks_pvalue: Optional[float] = None


class UnlearningVerifier:
    """Verifier for unlearning effectiveness."""
    
    def __init__(
        self,
        model: LoRAModelWrapper,
        collator: LLMCollator,
        device: str = "cuda"
    ):
        self.model = model
        self.collator = collator
        self.device = device
    
    def compute_sample_loss(
        self,
        samples: List[Sample],
        batch_size: int = 8,
        show_progress: bool = False,
        progress_desc: str = "Computing loss"
    ) -> List[float]:
        """Compute per-sample loss for a list of samples.
        
        Args:
            samples: List of samples
            batch_size: Batch size for inference
            show_progress: Whether to show progress bar
            progress_desc: Description for progress bar
            
        Returns:
            List of per-sample losses
        """
        self.model.model.eval()
        losses = []
        
        num_batches = (len(samples) + batch_size - 1) // batch_size
        batch_iter = range(0, len(samples), batch_size)
        
        if show_progress:
            batch_iter = tqdm(batch_iter, desc=progress_desc, total=num_batches, dynamic_ncols=True)
        
        with torch.no_grad():
            for i in batch_iter:
                batch_samples = samples[i:i+batch_size]
                batch = self.collator(batch_samples)
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.model.model(**batch)
                
                # Compute per-sample loss
                logits = outputs.logits[:, :-1, :].contiguous()
                labels = batch["labels"][:, 1:].contiguous()
                
                for j in range(len(batch_samples)):
                    sample_logits = logits[j]
                    sample_labels = labels[j]
                    
                    # Mask out padding
                    mask = sample_labels != -100
                    if mask.sum() == 0:
                        losses.append(float('nan'))
                        continue
                    
                    sample_logits = sample_logits[mask]
                    sample_labels = sample_labels[mask]
                    
                    loss = torch.nn.functional.cross_entropy(
                        sample_logits, sample_labels, reduction='mean'
                    )
                    losses.append(loss.item())
        
        return losses
    
    def compute_min_k_prob(
        self,
        samples: List[Sample],
        k_percent: float = 20.0,
        batch_size: int = 8
    ) -> List[float]:
        """Compute Min-K% probability for membership inference.
        
        Min-K% Prob (Shi et al., 2024): Uses the average log probability
        of the K% tokens with lowest probability as the membership signal.
        
        Args:
            samples: List of samples
            k_percent: Percentage of tokens to consider (default: 20%)
            batch_size: Batch size
            
        Returns:
            List of Min-K% scores (lower = more likely member)
        """
        self.model.model.eval()
        scores = []
        
        with torch.no_grad():
            for i in range(0, len(samples), batch_size):
                batch_samples = samples[i:i+batch_size]
                batch = self.collator(batch_samples)
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.model.model(**batch)
                logits = outputs.logits[:, :-1, :]
                labels = batch["labels"][:, 1:]
                
                # Compute log probabilities
                log_probs = torch.log_softmax(logits, dim=-1)
                
                for j in range(len(batch_samples)):
                    sample_labels = labels[j]
                    mask = sample_labels != -100
                    
                    if mask.sum() == 0:
                        scores.append(float('nan'))
                        continue
                    
                    # Get log probs for actual tokens
                    sample_log_probs = log_probs[j][mask]
                    sample_labels_masked = sample_labels[mask]
                    
                    token_log_probs = sample_log_probs.gather(
                        1, sample_labels_masked.unsqueeze(1)
                    ).squeeze(1)
                    
                    # Get Min-K% tokens
                    k = max(1, int(len(token_log_probs) * k_percent / 100))
                    min_k_probs, _ = torch.topk(token_log_probs, k, largest=False)
                    
                    scores.append(min_k_probs.mean().item())
        
        return scores
    
    def run_mia(
        self,
        member_samples: List[Sample],
        nonmember_samples: List[Sample],
        method: str = "loss",
        batch_size: int = 8,
        show_progress: bool = False,
        precomputed_member_losses: List[float] = None,
        precomputed_nonmember_losses: List[float] = None,
    ) -> Dict[str, float]:
        """Run a simple membership inference attack.

        Conventions:
          - labels: 1 = member, 0 = non-member
          - scores: higher => more likely member
        """
        if method == "loss":
            # 使用预计算的 loss（如果提供），避免重复计算
            if precomputed_member_losses is not None:
                member_losses = precomputed_member_losses
            else:
                member_losses = self.compute_sample_loss(
                    member_samples,
                    batch_size,
                    show_progress=show_progress, progress_desc="MIA forget loss"
                )
            
            if precomputed_nonmember_losses is not None:
                nonmember_losses = precomputed_nonmember_losses
            else:
                nonmember_losses = self.compute_sample_loss(
                    nonmember_samples,
                    batch_size,
                    show_progress=show_progress, progress_desc="MIA retain loss"
                )
            # For loss: lower = more likely member, so we negate
            member_scores = [-s for s in member_losses if not np.isnan(s)]
            nonmember_scores = [-s for s in nonmember_losses if not np.isnan(s)]
        else:  # min_k
            member_scores = self.compute_min_k_prob(member_samples, batch_size=batch_size)
            nonmember_scores = self.compute_min_k_prob(nonmember_samples, batch_size=batch_size)
            member_scores = [s for s in member_scores if not np.isnan(s)]
            nonmember_scores = [s for s in nonmember_scores if not np.isnan(s)]

        labels = [1] * len(member_scores) + [0] * len(nonmember_scores)
        scores = member_scores + nonmember_scores
        
        if len(set(labels)) < 2:
            return {
                "auc": 0.5,
                "auc_sym": 0.5,
                "adv": 0.0,
                "tpr_at_1fpr": 0.0,
                "tpr_at_5fpr": 0.0,
                "ks_stat": 0.0,
                "ks_pvalue": 1.0,
            }
        
        # Compute AUC
        auc = roc_auc_score(labels, scores)
        auc_sym = max(float(auc), float(1.0 - auc))
        adv = float(auc_sym - 0.5)

        # Compute TPR at specific FPR
        fpr, tpr, _ = roc_curve(labels, scores)
        tpr_at_1fpr = tpr[np.searchsorted(fpr, 0.01)]
        tpr_at_5fpr = tpr[np.searchsorted(fpr, 0.05)]

        # KS test for score distribution separation (regardless of label direction)
        ks = ks_2samp(member_scores, nonmember_scores, alternative="two-sided", mode="auto")

        return {
            "auc": float(auc),
            "auc_sym": float(auc_sym),
            "adv": float(adv),
            "tpr_at_1fpr": float(tpr_at_1fpr),
            "tpr_at_5fpr": float(tpr_at_5fpr),
            "ks_stat": float(ks.statistic),
            "ks_pvalue": float(ks.pvalue),
        }

    def verify_unlearning(
        self,
        forget_samples: List[Sample],
        retain_samples: List[Sample],
        *,
        nonmember_samples: Optional[List[Sample]] = None,
        batch_size: int = 8,
        verbose: bool = True,
        max_samples: int = None,
        show_progress: bool = True
    ) -> UnlearningMetrics:
        """Comprehensive unlearning verification.

        Args:
            forget_samples: Samples that should be forgotten (Agent 0's data)
            retain_samples: Samples that should be retained (other agents' data)
            nonmember_samples: Optional non-member pool (val/test). If provided, MIA is computed on
                (forget as members) vs (nonmember as non-members). If omitted, falls back to using
                retain_samples as the non-member pool (not a strict membership test).
            batch_size: Batch size
            verbose: Print results
            max_samples: Maximum number of samples to evaluate (None = all)
            show_progress: Whether to show progress bar

        Returns:
            UnlearningMetrics with all verification results
        """
        # 限制评估样本数量（使用确定性前N条，避免 random.sample 带来的波动）
        if max_samples is not None and max_samples > 0:
            if len(forget_samples) > max_samples:
                forget_samples = forget_samples[:max_samples]
            if len(retain_samples) > max_samples:
                retain_samples = retain_samples[:max_samples]
        
        if verbose:
            print(f"Computing unlearning verification metrics...")
            print(f"  Forget samples: {len(forget_samples)}, Retain samples: {len(retain_samples)}")

        # 1. Compute losses (只计算一次，用于 loss 指标和 MIA)
        forget_losses = self.compute_sample_loss(
            forget_samples, batch_size,
            show_progress=show_progress, progress_desc="Forget loss"
        )
        retain_losses = self.compute_sample_loss(
            retain_samples, batch_size,
            show_progress=show_progress, progress_desc="Retain loss"
        )

        forget_losses_clean = [l for l in forget_losses if not np.isnan(l)]
        retain_losses_clean = [l for l in retain_losses if not np.isnan(l)]

        avg_forget_loss = np.mean(forget_losses_clean) if forget_losses_clean else 0.0
        avg_retain_loss = np.mean(retain_losses_clean) if retain_losses_clean else 0.0
        loss_gap = avg_forget_loss - avg_retain_loss

        # 2. Run MIA (forget members vs non-members)
        mia_nonmember_samples = nonmember_samples if nonmember_samples is not None else retain_samples
        nonmember_losses = None
        if nonmember_samples is not None:
            nonmember_losses = self.compute_sample_loss(
                mia_nonmember_samples,
                batch_size,
                show_progress=show_progress,
                progress_desc="Non-member loss",
            )

        # 复用已计算的 forget loss，避免重复计算
        mia = self.run_mia(
            forget_samples,
            mia_nonmember_samples,
            method="loss",
            batch_size=batch_size,
            show_progress=False,  # 不再显示进度，因为已经计算过了
            precomputed_member_losses=forget_losses,
            precomputed_nonmember_losses=nonmember_losses,
        )

        metrics = UnlearningMetrics(
            mia_auc=float(mia.get("auc", 0.5)),
            mia_auc_sym=float(mia.get("auc_sym", 0.5)),
            mia_adv=float(mia.get("adv", 0.0)),
            mia_tpr_at_1fpr=float(mia.get("tpr_at_1fpr", 0.0)),
            mia_tpr_at_5fpr=float(mia.get("tpr_at_5fpr", 0.0)),
            mia_ks_stat=float(mia.get("ks_stat", 0.0)),
            mia_ks_pvalue=float(mia.get("ks_pvalue", 1.0)),
            forget_loss=avg_forget_loss,
            retain_loss=avg_retain_loss,
            loss_gap=loss_gap
        )

        if verbose:
            print(f"  Forget Loss: {avg_forget_loss:.4f}")
            print(f"  Retain Loss: {avg_retain_loss:.4f}")
            print(f"  Loss Gap (forget - retain): {loss_gap:.4f}")
            print(f"  MIA AUC: {metrics.mia_auc:.4f} (sym={metrics.mia_auc_sym:.4f}, adv={metrics.mia_adv:.4f})")
            print(f"  MIA TPR@1%FPR: {metrics.mia_tpr_at_1fpr:.4f}")
            print(f"  MIA TPR@5%FPR: {metrics.mia_tpr_at_5fpr:.4f}")
            print(f"  MIA KS: {metrics.mia_ks_stat:.4f} (p={metrics.mia_ks_pvalue:.2e})")

            # Interpretation
            if loss_gap > 0.5:
                print("  → Strong unlearning signal: forget set has higher loss")
            elif loss_gap > 0.1:
                print("  → Moderate unlearning signal")
            else:
                print("  → Weak unlearning signal: model may still remember forget set")

            adv = metrics.mia_adv or 0.0
            if adv < 0.05:
                print("  → MIA has near-random separation (auc_sym≈0.5)")
            elif adv < 0.15:
                print("  → MIA has weak separation")
            else:
                print("  → MIA has strong separation")

        return metrics


def compare_dfl_dfu_unlearning(
    dfl_model_state: Dict[str, torch.Tensor],
    dfu_model_state: Dict[str, torch.Tensor],
    model: LoRAModelWrapper,
    collator: LLMCollator,
    forget_samples: List[Sample],
    retain_samples: List[Sample],
    device: str = "cuda",
    batch_size: int = 8
) -> Dict[str, UnlearningMetrics]:
    """Compare unlearning metrics between DFL and DFU models.

    Args:
        dfl_model_state: Original DFL model state
        dfu_model_state: DFU model state after unlearning
        model: LoRA model wrapper
        collator: Data collator
        forget_samples: Samples to forget
        retain_samples: Samples to retain
        device: Device
        batch_size: Batch size

    Returns:
        Dict with "dfl" and "dfu" UnlearningMetrics
    """
    verifier = UnlearningVerifier(model, collator, device)

    results = {}

    # Evaluate DFL model
    print("\n=== DFL Model (before unlearning) ===")
    model.set_lora_state_dict(dfl_model_state)
    results["dfl"] = verifier.verify_unlearning(
        forget_samples,
        retain_samples,
        batch_size=batch_size,
    )

    # Evaluate DFU model
    print("\n=== DFU Model (after unlearning) ===")
    model.set_lora_state_dict(dfu_model_state)
    results["dfu"] = verifier.verify_unlearning(
        forget_samples,
        retain_samples,
        batch_size=batch_size,
    )

    # Summary
    print("\n=== Unlearning Effectiveness Summary ===")
    dfl_gap = results["dfl"].loss_gap
    dfu_gap = results["dfu"].loss_gap
    improvement = dfu_gap - dfl_gap

    print(f"Loss Gap Improvement: {improvement:.4f}")
    print(f"  DFL: {dfl_gap:.4f} → DFU: {dfu_gap:.4f}")

    if improvement > 0.3:
        print("  → Significant unlearning improvement!")
    elif improvement > 0.1:
        print("  → Moderate unlearning improvement")
    else:
        print("  → Limited unlearning improvement")

    return results
