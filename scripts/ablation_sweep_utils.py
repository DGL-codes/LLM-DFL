#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


RATIO_GRID: List[float] = [round(i / 10, 1) for i in range(1, 11)]
COUNT_GRID: List[int] = list(range(1, 10))


def strategy_dir_agent_count(k: int) -> str:
    return f"strategy_ours_count{int(k)}"


def strategy_dir_lora_ratio(r: float) -> str:
    return f"strategy_full_lora{float(r):.1f}_topratio_ours"


def strategy_dir_both(k: int, r: float) -> str:
    return f"strategy_ours_count{int(k)}_lora{float(r):.1f}_topratio_ours"


def mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        raise ValueError("mean_std: empty values")
    if len(values) == 1:
        return float(values[0]), 0.0
    return statistics.fmean(values), statistics.stdev(values)


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def latest_history_json(seed_root: Path, *, dataset: str, algorithm: str, strategy_dir: str) -> Optional[Path]:
    pattern = (
        f"{dataset}/{algorithm}/{strategy_dir}/"
        f"K*/G*_L*/alpha*/seed*_*/dfu_*/history.json"
    )
    candidates = [p for p in seed_root.glob(pattern) if p.is_file()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def extract_macro_f1(history_path: Path, *, agg: str = "best") -> float:
    """Return macro-F1 in [0,1].

    agg:
      - "best": prefer the best-agent statistic if present
      - "mean": prefer the averaged statistic if present

    Then fall back to the other final-stats value and finally to the last
    entry in `unlearning_metrics` / `avg_metrics`.
    """
    agg = (agg or "best").lower().strip()
    if agg not in {"best", "mean"}:
        raise ValueError(f"Unknown agg: {agg}")

    data = _load_json(history_path)
    final_stats = data.get("final_stats") or {}
    if isinstance(final_stats, dict):
        ordered_keys = (
            ["macro_f1_best", "macro_f1_mean"]
            if agg == "best"
            else ["macro_f1_mean", "macro_f1_best"]
        )
        for key in ordered_keys:
            value = final_stats.get(key)
            if value is not None:
                return float(value)

    metrics = data.get("unlearning_metrics") or data.get("avg_metrics") or []
    if metrics and isinstance(metrics, list) and isinstance(metrics[-1], dict):
        last = metrics[-1]
        if last.get("test_macro_f1_best") is not None:
            return float(last["test_macro_f1_best"])
        if last.get("test_macro_f1") is not None:
            return float(last["test_macro_f1"])
        if last.get("macro_f1") is not None:
            return float(last["macro_f1"])

    raise ValueError(f"Cannot find macro-F1 in {history_path}")


@dataclass(frozen=True)
class SweepPoint:
    k: int
    r: float


_BOTH_RE = re.compile(r"^strategy_ours_count(?P<k>\d+)_lora(?P<r>[0-9.]+)_topratio_ours$")


def parse_both_strategy_dir(name: str) -> Optional[SweepPoint]:
    m = _BOTH_RE.match(name or "")
    if not m:
        return None
    try:
        return SweepPoint(k=int(m.group("k")), r=float(m.group("r")))
    except ValueError:
        return None


def list_both_points(seed_root: Path, *, dataset: str, algorithm: str) -> List[SweepPoint]:
    base = seed_root / dataset / algorithm
    if not base.exists():
        return []
    points: List[SweepPoint] = []
    for p in base.iterdir():
        if not p.is_dir():
            continue
        point = parse_both_strategy_dir(p.name)
        if point is None:
            continue
        points.append(point)
    points.sort(key=lambda x: (x.k, x.r))
    return points


def require_seed_roots(sweep_root: Path, seeds: List[int]) -> Dict[int, Path]:
    out: Dict[int, Path] = {}
    for seed in seeds:
        sr = sweep_root / f"seed{int(seed)}"
        if not sr.exists():
            raise FileNotFoundError(f"Missing seed root: {sr}")
        out[int(seed)] = sr
    return out
