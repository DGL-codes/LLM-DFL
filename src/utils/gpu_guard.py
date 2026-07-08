"""GPU guardrails for this project.

Default policy is to use physical GPU 2/3 only. We enforce this by requiring
`CUDA_VISIBLE_DEVICES` to be set to a subset of {"2","3"} and treating
`--gpu` arguments as *logical* indices within that visible set.

When needed, the allowed physical set can be overridden explicitly via
`LLMDFL_ALLOWED_PHYSICAL_GPUS`, e.g.:

- ``LLMDFL_ALLOWED_PHYSICAL_GPUS=2,3`` (default behavior)
- ``LLMDFL_ALLOWED_PHYSICAL_GPUS=0,1,2,3`` (full-machine runs)
"""

from __future__ import annotations

import os
from typing import Iterable, List, Optional, Sequence


def _parse_cuda_visible_devices(value: str) -> List[str]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        raise ValueError("Empty CUDA_VISIBLE_DEVICES")
    for p in parts:
        if not p.isdigit():
            raise ValueError(f"Invalid CUDA_VISIBLE_DEVICES entry: {p!r}")
    return parts


def _resolve_allowed_physical(default_allowed: Sequence[int]) -> List[int]:
    env_value = os.environ.get("LLMDFL_ALLOWED_PHYSICAL_GPUS")
    if env_value is None or str(env_value).strip() == "":
        return [int(i) for i in default_allowed]

    parts = _parse_cuda_visible_devices(str(env_value))
    nums = [int(p) for p in parts]
    if len(set(nums)) != len(nums):
        raise RuntimeError(
            "LLMDFL_ALLOWED_PHYSICAL_GPUS contains duplicates: "
            f"{env_value!r}"
        )
    return nums


def enforce_cuda_visible_devices(
    *,
    allowed_physical: Sequence[int] = (2, 3),
    require: bool = True,
) -> List[int]:
    """Ensure CUDA_VISIBLE_DEVICES is set and only contains allowed physical ids.

    Returns the parsed list of *physical* GPU ids made visible.
    """
    allowed_list = _resolve_allowed_physical(allowed_physical)

    value = os.environ.get("CUDA_VISIBLE_DEVICES")
    if value is None or str(value).strip() == "":
        allowed_hint = ",".join(str(i) for i in allowed_list)
        if require:
            raise RuntimeError(
                "This project requires explicit CUDA_VISIBLE_DEVICES.\n"
                f"Allowed physical GPU set: {allowed_hint}.\n"
                "Please set CUDA_VISIBLE_DEVICES to a non-empty subset of the allowed set.\n"
                "and then pass --gpu as a LOGICAL index within that visible set "
                "(e.g. --gpu 0 or --gpu 1)."
            )
        return []

    parts = _parse_cuda_visible_devices(str(value))
    allowed_set = {str(i) for i in allowed_list}
    bad = [p for p in parts if p not in allowed_set]
    if bad:
        allowed_hint = ",".join(str(i) for i in allowed_list)
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES violates allowed physical set.\n"
            f"Got CUDA_VISIBLE_DEVICES={value!r} (invalid entries: {bad}).\n"
            f"Allowed physical GPU set: {allowed_hint}."
        )

    # No duplicates.
    if len(set(parts)) != len(parts):
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES contains duplicates: {value!r}")

    return [int(p) for p in parts]


def enforce_torch_gpu_index(
    gpu: Optional[int],
    *,
    visible_physical: Sequence[int],
) -> None:
    """Validate `--gpu` as a torch-visible index (0..len(visible)-1)."""
    if gpu is None:
        return
    if not isinstance(gpu, int):
        raise TypeError(f"--gpu must be int or None, got: {type(gpu)}")
    if gpu < 0 or gpu >= len(visible_physical):
        visible_str = ",".join(str(i) for i in visible_physical)
        raise RuntimeError(
            f"Invalid --gpu {gpu} for CUDA_VISIBLE_DEVICES={visible_str}.\n"
            "NOTE: --gpu is a *logical* index within CUDA_VISIBLE_DEVICES.\n"
            f"Valid values: {list(range(len(visible_physical)))}"
        )


def enforce_torch_gpu_indices(
    gpus: Iterable[int],
    *,
    visible_physical: Sequence[int],
) -> None:
    for gpu in gpus:
        enforce_torch_gpu_index(int(gpu), visible_physical=visible_physical)


def guard_gpu_or_raise(*, gpu: Optional[int]) -> List[int]:
    """One-shot helper for scripts: enforce env + validate --gpu."""
    visible_physical = enforce_cuda_visible_devices()
    enforce_torch_gpu_index(gpu, visible_physical=visible_physical)
    return visible_physical
