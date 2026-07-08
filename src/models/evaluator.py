"""Evaluator for LLM outputs."""
import warnings
import torch
from typing import List, Dict, Optional
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm
import re

from .lora_model import LoRAModelWrapper
from ..data.base import Sample
from ..data.collator import LLMCollator


class Evaluator:
    """Evaluator for different task types."""

    def __init__(self, model: LoRAModelWrapper, collator: LLMCollator):
        self.model = model
        self.collator = collator
        self.collator.inference_mode = True

    @torch.no_grad()
    def generate_predictions(
        self,
        samples: List[Sample],
        batch_size: int = 8,
        max_new_tokens: int = 16,  # Reduced: max label is ~10 tokens
        show_progress: bool = False
    ) -> List[str]:
        """Generate predictions for samples."""
        self.model.eval()
        predictions = []

        num_batches = (len(samples) + batch_size - 1) // batch_size
        batch_iter = range(0, len(samples), batch_size)
        if show_progress:
            batch_iter = tqdm(batch_iter, total=num_batches, desc="Generating", leave=False)

        for i in batch_iter:
            batch_samples = samples[i:i + batch_size]
            batch = self.collator(batch_samples)

            input_ids = batch["input_ids"].to(self.model.device)
            attention_mask = batch["attention_mask"].to(self.model.device)

            outputs = self.model.generate(
                input_ids, attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1
            )

            # Decode only the new tokens
            for j, output in enumerate(outputs):
                # Use actual input length (sequence length before generation)
                input_len = input_ids.shape[1]
                new_tokens = output[input_len:]
                pred = self.model.tokenizer.decode(new_tokens, skip_special_tokens=True)
                predictions.append(pred.strip())

        return predictions

    @staticmethod
    def _match_label(pred: str, label_names: List[str]) -> int:
        pred_lower = str(pred).lower().strip()
        pred_norm = re.sub(r"[^a-z0-9]+", " ", pred_lower).strip()
        if not pred_norm:
            return -1

        for idx, name in enumerate(label_names):
            name_norm = re.sub(r"[^a-z0-9]+", " ", str(name).lower()).strip()
            if not name_norm:
                continue
            if pred_norm == name_norm:
                return int(idx)
            if re.search(rf"(^|\s){re.escape(name_norm)}($|\s)", pred_norm):
                return int(idx)
        return -1
    
    def evaluate_classification(
        self,
        samples: List[Sample],
        label_names: List[str],
        batch_size: int = 8,
        max_new_tokens: int = 16
    ) -> Dict[str, float]:
        """Evaluate classification task."""
        predictions = self.generate_predictions(samples, batch_size, max_new_tokens)
        
        # Map predictions to labels
        pred_labels = []
        true_labels = [s.label for s in samples]
        
        for pred in predictions:
            pred_labels.append(self._match_label(pred, label_names))
        
        # Filter out unknown predictions for metrics
        valid_mask = [p != -1 for p in pred_labels]
        valid_preds = [p for p, v in zip(pred_labels, valid_mask) if v]
        valid_true = [t for t, v in zip(true_labels, valid_mask) if v]
        
        if not valid_preds:
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "macro_f1": 0.0, "valid_ratio": 0.0}

        # Suppress sklearn warning when num_classes > 50% of samples
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*number of unique classes.*")
            return {
                "accuracy": accuracy_score(valid_true, valid_preds),
                "precision": precision_score(valid_true, valid_preds, average="macro", zero_division=0),
                "recall": recall_score(valid_true, valid_preds, average="macro", zero_division=0),
                "macro_f1": f1_score(valid_true, valid_preds, average="macro", zero_division=0),
                "valid_ratio": len(valid_preds) / len(samples)
            }

    def evaluate_backdoor_asr(
        self,
        samples: List[Sample],
        label_names: List[str],
        *,
        target_label: int,
        batch_size: int = 8,
        max_new_tokens: int = 16,
    ) -> Dict[str, float]:
        """Compute backdoor ASR on *triggered* inputs.

        ASR = P(predicted_label == target_label | trigger present).
        """
        if not samples:
            return {"asr": 0.0, "valid_ratio": 0.0}
        if not label_names:
            raise ValueError("label_names is required for ASR evaluation")

        target = int(target_label)
        preds = self.generate_predictions(samples, batch_size=batch_size, max_new_tokens=max_new_tokens)
        mapped = [self._match_label(p, label_names) for p in preds]
        valid = [p for p in mapped if p != -1]
        if not valid:
            return {"asr": 0.0, "valid_ratio": 0.0}
        asr = sum(1 for p in valid if p == target) / len(valid)
        return {"asr": float(asr), "valid_ratio": float(len(valid) / len(samples))}
    
    def evaluate_generation(
        self,
        samples: List[Sample],
        batch_size: int = 8,
        max_new_tokens: int = 64
    ) -> Dict[str, float]:
        """Evaluate generation task using exact match and token F1."""
        predictions = self.generate_predictions(samples, batch_size, max_new_tokens)
        
        exact_matches = 0
        token_f1_scores = []
        
        for pred, sample in zip(predictions, samples):
            ref = sample.output_text.strip().lower()
            pred_clean = pred.strip().lower()
            
            # Exact match
            if pred_clean == ref:
                exact_matches += 1
            
            # Token F1
            pred_tokens = set(pred_clean.split())
            ref_tokens = set(ref.split())
            
            if not pred_tokens or not ref_tokens:
                token_f1_scores.append(0.0)
                continue
            
            common = pred_tokens & ref_tokens
            precision = len(common) / len(pred_tokens) if pred_tokens else 0
            recall = len(common) / len(ref_tokens) if ref_tokens else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            token_f1_scores.append(f1)
        
        return {
            "exact_match": exact_matches / len(samples),
            "token_f1": sum(token_f1_scores) / len(token_f1_scores)
        }
