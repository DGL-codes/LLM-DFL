"""Backdoor data utilities for unlearning auditing.

This module provides lightweight helpers to:
- inject a trigger string into the input text
- overwrite the output/label to a target class (classification-as-generation)

Typical usage in this repo:
  - poison ONLY the forget client (e.g., agent0) during DFL training
  - later, audit forgetting by measuring ASR (attack success rate) on triggered inputs
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Sequence

from .base import Sample


def apply_trigger(text: str, trigger: str, *, position: str = "prefix") -> str:
    trigger = str(trigger or "").strip()
    if not trigger:
        return text

    pos = str(position or "prefix").lower().strip()
    if pos not in {"prefix", "suffix"}:
        raise ValueError(f"Unknown trigger position: {position!r} (expected 'prefix' or 'suffix')")

    if pos == "suffix":
        return f"{text}\n{trigger}"
    return f"{trigger}\n{text}"


def make_triggered_samples(samples: List[Sample], *, trigger: str, position: str = "prefix") -> List[Sample]:
    """Return a triggered copy of samples (keeps labels/outputs unchanged)."""
    out: List[Sample] = []
    for s in samples:
        out.append(
            Sample(
                instruction=s.instruction,
                input_text=apply_trigger(s.input_text, trigger, position=position),
                output_text=s.output_text,
                label=s.label,
                paraphrased_answer=s.paraphrased_answer,
                perturbed_answers=s.perturbed_answers,
            )
        )
    return out


@dataclass(frozen=True)
class BackdoorPoisonSpec:
    trigger: str
    poison_rate: float
    target_label: int
    position: str = "prefix"
    seed: int = 42


def poison_samples_by_indices(
    samples: List[Sample],
    indices: Sequence[int],
    *,
    label_names: Sequence[str],
    spec: BackdoorPoisonSpec,
) -> int:
    """Poison a subset of `samples` in-place at the provided indices.

    For each selected index (w.p. `poison_rate`), we:
      - inject the trigger into `input_text`
      - overwrite `output_text` and `label` to the target class

    Returns number of poisoned samples.
    """
    if not samples or not indices:
        return 0

    if spec.poison_rate <= 0.0:
        return 0
    if spec.poison_rate > 1.0:
        raise ValueError(f"poison_rate must be in [0,1], got: {spec.poison_rate}")

    if not label_names:
        raise ValueError("label_names is required for backdoor poisoning")

    target_label = int(spec.target_label)
    if target_label < 0 or target_label >= len(label_names):
        raise ValueError(
            f"Invalid target_label={target_label} for {len(label_names)} labels. "
            f"Valid range: [0, {len(label_names) - 1}]"
        )
    target_output = str(label_names[target_label])

    rng = random.Random(int(spec.seed))
    poisoned = 0
    for idx in indices:
        i = int(idx)
        if i < 0 or i >= len(samples):
            continue
        if rng.random() >= float(spec.poison_rate):
            continue

        s = samples[i]
        samples[i] = Sample(
            instruction=s.instruction,
            input_text=apply_trigger(s.input_text, str(spec.trigger), position=spec.position),
            output_text=target_output,
            label=target_label,
            paraphrased_answer=s.paraphrased_answer,
            perturbed_answers=s.perturbed_answers,
        )
        poisoned += 1
    return poisoned

