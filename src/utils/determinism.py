"""Small deterministic helpers (seed derivation, sampling).

We avoid Python's built-in `hash()` because it is randomized per process.
"""

from __future__ import annotations

import zlib


def derive_seed(
    base_seed: int,
    *,
    salt: str,
    round_idx: int = 0,
    agent_id: int = 0,
    extra: int = 0,
) -> int:
    """Derive a stable 32-bit seed from a base seed + structured identifiers."""
    base_seed = int(base_seed) & 0xFFFFFFFF
    salt_id = zlib.crc32(str(salt).encode("utf-8")) & 0xFFFFFFFF
    mixed = (
        base_seed * 1000003
        + salt_id * 10007
        + int(round_idx) * 1009
        + int(agent_id) * 37
        + int(extra) * 31
    )
    return int(mixed % (2**32 - 1))

