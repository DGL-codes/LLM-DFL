#!/usr/bin/env python3
"""Reproduce the paper's LLM tables/figures from existing local artifacts.

This script is intentionally *read-only* (no training). It:
- parses DFU `history.json` files under known experiment folders
- aggregates Table I/II metrics (Macro-F1, MIA-AUC) for DSU ×/✓
- compares the aggregated results to the numbers extracted from the PDF
- writes CSV + a diff report under `reports/`

Notes
-----
This repo contains multiple experiment folders with partially missing runs
(e.g. broken symlinks). The script is robust to missing files: it reports
what it could find and flags missing seeds/runs in the diff report.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent

DATASETS = ["20newsgroups", "yahoo_subset"]
METHODS = ["d-federaser", "d-fedosd", "d-fedrecovery", "d-oblivionis"]
METHOD_LABEL = {
    "d-federaser": "FedEraser",
    "d-fedosd": "FedOSD",
    "d-fedrecovery": "FedRecovery",
    "d-oblivionis": "Oblivionis",
}

DEFAULT_SEEDS = [42, 43, 44]


PAPER_TABLES = {
    "20newsgroups": {
        "retrain": {"f1": (54.93, 1.25), "mia": (46.90, 1.08)},
        "d-federaser": {"base": {"f1": (40.03, 1.70), "mia": (48.36, 1.26)}, "dsu": {"f1": (44.94, 0.63), "mia": (48.80, 1.98)}},
        "d-fedosd": {"base": {"f1": (47.98, 6.63), "mia": (45.13, 2.88)}, "dsu": {"f1": (58.48, 1.00), "mia": (45.73, 2.89)}},
        "d-fedrecovery": {"base": {"f1": (46.18, 6.39), "mia": (47.34, 1.60)}, "dsu": {"f1": (61.23, 3.61), "mia": (49.47, 3.83)}},
        "d-oblivionis": {"base": {"f1": (46.37, 1.53), "mia": (43.61, 3.17)}, "dsu": {"f1": (62.01, 2.47), "mia": (43.58, 1.49)}},
    },
    "yahoo_subset": {
        "retrain": {"f1": (72.18, 1.52), "mia": (51.93, 1.35)},
        "d-federaser": {"base": {"f1": (67.49, 3.47), "mia": (49.00, 1.15)}, "dsu": {"f1": (69.33, 0.90), "mia": (47.74, 0.65)}},
        "d-fedosd": {"base": {"f1": (60.97, 16.98), "mia": (41.28, 1.80)}, "dsu": {"f1": (75.34, 1.05), "mia": (45.18, 0.54)}},
        "d-fedrecovery": {"base": {"f1": (70.82, 4.44), "mia": (50.51, 2.98)}, "dsu": {"f1": (74.28, 3.17), "mia": (53.11, 2.61)}},
        "d-oblivionis": {"base": {"f1": (67.82, 5.10), "mia": (44.20, 2.21)}, "dsu": {"f1": (73.64, 5.57), "mia": (48.90, 1.39)}},
    },
}


@dataclass(frozen=True)
class MetricAgg:
    mean: float
    std: float
    n: int


@dataclass(frozen=True)
class Cell:
    f1: MetricAgg
    mia: MetricAgg
    seed_values_f1: Dict[int, float]
    seed_values_mia: Dict[int, float]
    run_paths: Dict[int, str]
    dfl_snapshots: Dict[int, str]
    dfu_seeds: Dict[int, int]


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, (np.floating, np.integer)):
        v = float(v)
    if isinstance(v, (int, float)):
        fv = float(v)
        return None if math.isnan(fv) else fv
    return None


def _macro_f1_best(final_stats: Dict[str, Any]) -> Optional[float]:
    v = _safe_float(final_stats.get("macro_f1_best"))
    if v is not None and v != 0.0:
        return v
    per = final_stats.get("per_agent") or {}
    best = None
    for _aid, metrics in per.items():
        if not isinstance(metrics, dict):
            continue
        f1 = _safe_float(metrics.get("macro_f1"))
        if f1 is None:
            continue
        best = f1 if best is None else max(best, f1)
    return best


def _mia_auc(final_stats: Dict[str, Any]) -> Optional[float]:
    return _safe_float(final_stats.get("mia_auc"))


def _agg(values: Dict[int, float]) -> MetricAgg:
    xs = [float(v) for v in values.values() if v is not None]
    if not xs:
        return MetricAgg(mean=float("nan"), std=float("nan"), n=0)
    mean = float(np.mean(xs))
    std = float(np.std(xs, ddof=1)) if len(xs) >= 2 else 0.0
    return MetricAgg(mean=mean, std=std, n=len(xs))


def _fmt_ms(m: MetricAgg) -> str:
    if m.n <= 0 or math.isnan(m.mean):
        return "-"
    return f"{m.mean:.2f}±{m.std:.2f}"


def _find_latest_history(search_root: Path, pattern: str) -> Optional[Path]:
    candidates = list(search_root.glob(pattern))
    candidates = [p for p in candidates if p.is_file() and p.name == "history.json"]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _parse_snapshot_seed(snapshot: str) -> Optional[int]:
    m = re.search(r"seed(\d+)_", str(snapshot))
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _is_seed_aligned(history_path: Path, expected_seed: int) -> bool:
    cfg_path = history_path.parent / "dfu_config.json"
    if not cfg_path.exists():
        return False
    try:
        cfg = _read_json(cfg_path)
    except Exception:
        return False
    try:
        dfu_seed = int(cfg.get("seed"))
    except Exception:
        return False
    snapshot = cfg.get("dfl_snapshot")
    if not isinstance(snapshot, str):
        return False
    snapshot_seed = _parse_snapshot_seed(snapshot)
    if snapshot_seed is None:
        return False
    return (dfu_seed == int(expected_seed)) and (snapshot_seed == int(expected_seed))


def _find_history_for_seed(
    *,
    root: Path,
    dataset: str,
    method: str,
    strategy_glob: str,
    seed: int,
    require_seed_aligned_snapshot: bool,
) -> Optional[Path]:
    # Typical layout: <root>/<dataset>/<method>/<strategy...>/K.../G.../alpha.../<snapshot>/dfu_*/history.json
    base = root / dataset / method
    if not base.exists():
        return None
    candidates = list(base.glob(f"{strategy_glob}/**/history.json"))
    candidates = [p for p in candidates if p.is_file() and p.name == "history.json"]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    aligned = [p for p in candidates if _is_seed_aligned(p, seed)]
    if aligned:
        return aligned[0]
    if require_seed_aligned_snapshot:
        return None
    return candidates[0]


def _load_cell(
    *,
    runs_by_seed: Dict[int, Path],
) -> Cell:
    seed_f1: Dict[int, float] = {}
    seed_mia: Dict[int, float] = {}
    run_paths: Dict[int, str] = {}
    dfl_snapshots: Dict[int, str] = {}
    dfu_seeds: Dict[int, int] = {}
    for seed, history_path in sorted(runs_by_seed.items()):
        try:
            data = _read_json(history_path)
        except Exception:
            continue
        final_stats = data.get("final_stats") or {}
        f1 = _macro_f1_best(final_stats)
        mia = _mia_auc(final_stats)
        if f1 is not None:
            seed_f1[int(seed)] = float(f1) * 100.0
        if mia is not None:
            seed_mia[int(seed)] = float(mia) * 100.0
        run_paths[int(seed)] = str(history_path)

        # Optional: enrich with dfu_config.json (helps diagnose what "seeds" mean).
        try:
            dfu_cfg_path = history_path.parent / "dfu_config.json"
            if dfu_cfg_path.exists():
                cfg = _read_json(dfu_cfg_path)
                snap = cfg.get("dfl_snapshot")
                if isinstance(snap, str):
                    dfl_snapshots[int(seed)] = snap
                if cfg.get("seed") is not None:
                    dfu_seeds[int(seed)] = int(cfg["seed"])
        except Exception:
            pass

    return Cell(
        f1=_agg(seed_f1),
        mia=_agg(seed_mia),
        seed_values_f1=seed_f1,
        seed_values_mia=seed_mia,
        run_paths=run_paths,
        dfl_snapshots=dfl_snapshots,
        dfu_seeds=dfu_seeds,
    )


def _load_mia_override_csv(path: Path) -> Dict[Tuple[str, str, str, int], float]:
    out: Dict[Tuple[str, str, str, int], float] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                dataset = str(row.get("dataset", "")).strip()
                method = str(row.get("method", "")).strip()
                cell = str(row.get("cell", "")).strip()
                seed = int(str(row.get("seed", "")).strip())
                value = float(str(row.get("mia_value", "")).strip())
            except Exception:
                continue
            if not dataset or not method or not cell:
                continue
            out[(dataset, method, cell, seed)] = value
    return out


def _load_run_override_csv(path: Path) -> Dict[Tuple[str, str, str, int], Path]:
    out: Dict[Tuple[str, str, str, int], Path] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                dataset = str(row.get("dataset", "")).strip()
                method = str(row.get("method", "")).strip()
                cell = str(row.get("cell", "")).strip()
                seed = int(str(row.get("seed", "")).strip())
                history_path = str(row.get("history_path", "")).strip()
            except Exception:
                continue
            if not dataset or not method or not cell or not history_path:
                continue
            hp = Path(history_path)
            if not hp.is_absolute():
                hp = ROOT / hp
            out[(dataset, method, cell, seed)] = hp
    return out


def _override_cell_mia(
    *,
    dataset: str,
    method: str,
    cell_key: str,
    seeds: List[int],
    cell: Cell,
    overrides: Dict[Tuple[str, str, str, int], float],
    missing: List[str],
) -> Cell:
    seed_mia = dict(cell.seed_values_mia)
    for seed in seeds:
        key = (dataset, method, cell_key, int(seed))
        if key not in overrides:
            missing.append(f"[mia_override] missing {dataset} {method} {cell_key} seed{seed}")
            continue
        seed_mia[int(seed)] = float(overrides[key])
    return Cell(
        f1=cell.f1,
        mia=_agg(seed_mia),
        seed_values_f1=cell.seed_values_f1,
        seed_values_mia=seed_mia,
        run_paths=cell.run_paths,
        dfl_snapshots=cell.dfl_snapshots,
        dfu_seeds=cell.dfu_seeds,
    )


def _write_selected_runs_csv(
    *,
    path: Path,
    tables: Dict[str, Dict[str, Dict[str, Cell]]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: List[List[str]] = [
        ["dataset", "method", "cell", "seed", "history_path", "dfu_dir", "dfl_snapshot", "dfu_seed"]
    ]
    for dataset in DATASETS:
        for method in METHODS:
            for cell_key in ["base", "dsu"]:
                cell = tables.get(dataset, {}).get(method, {}).get(cell_key)
                if cell is None:
                    continue
                for seed, history_path in sorted(cell.run_paths.items()):
                    hp = Path(history_path)
                    rows.append(
                        [
                            dataset,
                            method,
                            cell_key,
                            str(seed),
                            str(hp),
                            str(hp.parent),
                            str(cell.dfl_snapshots.get(int(seed), "")),
                            "" if int(seed) not in cell.dfu_seeds else str(cell.dfu_seeds[int(seed)]),
                        ]
                    )
    _write_csv(path, rows)


def _write_csv(path: Path, rows: List[List[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def _diff(a: MetricAgg, paper: Tuple[float, float]) -> Tuple[Optional[float], Optional[float]]:
    if a.n <= 0 or math.isnan(a.mean):
        return None, None
    pm, ps = paper
    return a.mean - pm, a.std - ps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, default=",".join(str(s) for s in DEFAULT_SEEDS))
    parser.add_argument("--out_dir", type=str, default="reports")
    parser.add_argument(
        "--require_seed_aligned_snapshot",
        type=str,
        default="true",
        help="If true, only aggregate runs where dfu_config.seed == requested seed and dfl_snapshot seed matches it.",
    )
    parser.add_argument(
        "--diagnose_seed",
        type=int,
        default=42,
        help="Also report the single-seed numbers for this seed (useful to sanity-check means/stds).",
    )
    parser.add_argument(
        "--base_seed42_root",
        type=str,
        default="dfu_sweeps_20news_lora_ratio_4methods_20260101_002412",
        help="Root containing seed42 base runs (strategy_full_lora1.0_*) for both datasets.",
    )
    parser.add_argument(
        "--base_ms_root",
        type=str,
        default="dfu_ms_full_sweeps_boxplots_20260101_064248",
        help="Root containing seed43/44 base runs under seedXX/...",
    )
    parser.add_argument(
        "--base_ms_root_20news_fedrecovery",
        type=str,
        default="dfu_ms_20news_fedrecovery_lora_ratio_20260101_054210",
        help="Optional extra root for 20newsgroups d-fedrecovery multi-seed base runs.",
    )
    parser.add_argument(
        "--dsu_root",
        type=str,
        default="dfu_ablation_as_ls_bestcfg_seed42_20260101_130036",
        help="Root containing DSU(AS+LS) bestcfg runs under seedXX/...",
    )
    parser.add_argument(
        "--seed_aligned_root",
        type=str,
        default="dfu_seed_aligned_llm_strict_424344",
        help="Root containing newly backfilled seed-aligned DFU runs.",
    )
    parser.add_argument(
        "--meta_json",
        type=str,
        default="artifacts/ablation_as_ls_bestcfg_424344.json",
        help="Meta JSON with best k/r per dataset+method.",
    )
    parser.add_argument(
        "--mia_protocol",
        type=str,
        default="history",
        choices=["history", "val", "test", "retain"],
        help="MIA source for table aggregation. 'history' uses existing history.json; others require --mia_override_csv.",
    )
    parser.add_argument(
        "--mia_metric",
        type=str,
        default="auc",
        choices=["auc", "auc_sym"],
        help="Only metadata flag for documentation; effective value comes from --mia_override_csv.",
    )
    parser.add_argument(
        "--mia_override_csv",
        type=str,
        default="",
        help="Optional override CSV (dataset,method,cell,seed,mia_value) produced by scripts/recompute_llm_table_mia.py.",
    )
    parser.add_argument(
        "--selected_runs_csv",
        type=str,
        default="artifacts/llm_selected_runs_424344.csv",
        help="Output CSV of selected base/dsu run paths used by this aggregation.",
    )
    parser.add_argument(
        "--run_override_csv",
        type=str,
        default="",
        help="Optional CSV (dataset,method,cell,seed,history_path) to force selected runs.",
    )
    args = parser.parse_args()
    require_seed_aligned_snapshot = str(args.require_seed_aligned_snapshot).strip().lower() in {"1", "true", "yes", "y"}

    seeds = [int(s.strip()) for s in str(args.seeds).split(",") if s.strip()]
    out_dir = ROOT / args.out_dir

    meta = _read_json(ROOT / args.meta_json)
    best_k: Dict[str, int] = {str(k): int(v) for k, v in (meta.get("best_k") or {}).items()}
    best_r: Dict[str, float] = {str(k): float(v) for k, v in (meta.get("best_r") or {}).items()}

    base_seed42_root = ROOT / args.base_seed42_root
    base_ms_root = ROOT / args.base_ms_root
    base_ms_root_20news_fedrecovery = ROOT / args.base_ms_root_20news_fedrecovery
    dsu_root = ROOT / args.dsu_root
    seed_aligned_root = ROOT / args.seed_aligned_root
    mia_override_csv = (ROOT / args.mia_override_csv) if str(args.mia_override_csv).strip() else None
    run_override_csv = (ROOT / args.run_override_csv) if str(args.run_override_csv).strip() else None
    mia_overrides: Dict[Tuple[str, str, str, int], float] = {}
    run_overrides: Dict[Tuple[str, str, str, int], Path] = {}
    if args.mia_protocol != "history":
        if mia_override_csv is None:
            raise ValueError("--mia_protocol requires --mia_override_csv")
        mia_overrides = _load_mia_override_csv(mia_override_csv)
        if not mia_overrides:
            raise ValueError(f"No valid overrides found in {mia_override_csv}")
    if run_override_csv is not None:
        run_overrides = _load_run_override_csv(run_override_csv)

    tables: Dict[str, Dict[str, Dict[str, Cell]]] = {}
    missing: List[str] = []

    for dataset in DATASETS:
        tables[dataset] = {}
        for method in METHODS:
            key = f"{dataset}::{method}"
            k = best_k.get(key)
            r = best_r.get(key)
            if k is None or r is None:
                missing.append(f"Missing best_k/best_r for {key} in {args.meta_json}")
                continue

            # Collect base (DSU×): strategy_full_lora1.0_* preferred, fallback strategy_full
            base_runs: Dict[int, Path] = {}
            for seed in seeds:
                override_key = (dataset, method, "base", int(seed))
                if override_key in run_overrides:
                    hp = run_overrides[override_key]
                    if not hp.exists():
                        missing.append(f"[run_override] base history missing: {hp}")
                    elif require_seed_aligned_snapshot and not _is_seed_aligned(hp, int(seed)):
                        missing.append(f"[run_override] base not seed-aligned: {hp}")
                    else:
                        base_runs[int(seed)] = hp
                        continue

                root_candidates: List[Path] = []
                if seed_aligned_root.exists():
                    root_candidates.append(seed_aligned_root)
                if seed == 42:
                    root_candidates.append(base_seed42_root)
                else:
                    if dataset == "20newsgroups" and method == "d-fedrecovery" and base_ms_root_20news_fedrecovery.exists():
                        root_candidates.append(base_ms_root_20news_fedrecovery / f"seed{seed}")
                    else:
                        root_candidates.append(base_ms_root / f"seed{seed}")

                h: Optional[Path] = None
                dedup_roots: List[Path] = []
                seen_roots: set[str] = set()
                for rc in root_candidates:
                    krc = str(rc)
                    if krc in seen_roots:
                        continue
                    seen_roots.add(krc)
                    dedup_roots.append(rc)

                for root in dedup_roots:
                    h = _find_history_for_seed(
                        root=root,
                        dataset=dataset,
                        method=method,
                        strategy_glob="strategy_full_lora1.0_*",
                        seed=int(seed),
                        require_seed_aligned_snapshot=require_seed_aligned_snapshot,
                    )
                    if h is not None:
                        break
                    h = _find_history_for_seed(
                        root=root,
                        dataset=dataset,
                        method=method,
                        strategy_glob="strategy_full",
                        seed=int(seed),
                        require_seed_aligned_snapshot=require_seed_aligned_snapshot,
                    )
                    if h is not None:
                        break
                if h is None:
                    missing.append(f"[base] missing history for seed{seed} {dataset} {method}")
                    continue
                base_runs[int(seed)] = h

            # Collect DSU (✓): strategy_ours_count{k}_lora{r}_*
            dsu_runs: Dict[int, Path] = {}
            for seed in seeds:
                override_key = (dataset, method, "dsu", int(seed))
                if override_key in run_overrides:
                    hp = run_overrides[override_key]
                    if not hp.exists():
                        missing.append(f"[run_override] dsu history missing: {hp}")
                    elif require_seed_aligned_snapshot and not _is_seed_aligned(hp, int(seed)):
                        missing.append(f"[run_override] dsu not seed-aligned: {hp}")
                    else:
                        dsu_runs[int(seed)] = hp
                        continue

                root_candidates: List[Path] = []
                if seed_aligned_root.exists():
                    root_candidates.append(seed_aligned_root)
                root_candidates.append(dsu_root / f"seed{seed}")
                # We keep the glob flexible to support both top_ratio and epsilon naming.
                h: Optional[Path] = None
                dedup_roots: List[Path] = []
                seen_roots: set[str] = set()
                for rc in root_candidates:
                    krc = str(rc)
                    if krc in seen_roots:
                        continue
                    seen_roots.add(krc)
                    dedup_roots.append(rc)

                for root in dedup_roots:
                    h = _find_history_for_seed(
                        root=root,
                        dataset=dataset,
                        method=method,
                        strategy_glob=f"strategy_ours_count{k}_lora{r}_*",
                        seed=int(seed),
                        require_seed_aligned_snapshot=require_seed_aligned_snapshot,
                    )
                    if h is not None:
                        break
                if h is None:
                    missing.append(f"[dsu] missing history for seed{seed} {dataset} {method} (k={k}, r={r})")
                    continue
                dsu_runs[int(seed)] = h

            base_cell = _load_cell(runs_by_seed=base_runs)
            dsu_cell = _load_cell(runs_by_seed=dsu_runs)
            if args.mia_protocol != "history":
                base_cell = _override_cell_mia(
                    dataset=dataset,
                    method=method,
                    cell_key="base",
                    seeds=seeds,
                    cell=base_cell,
                    overrides=mia_overrides,
                    missing=missing,
                )
                dsu_cell = _override_cell_mia(
                    dataset=dataset,
                    method=method,
                    cell_key="dsu",
                    seeds=seeds,
                    cell=dsu_cell,
                    overrides=mia_overrides,
                    missing=missing,
                )
            tables[dataset][method] = {"base": base_cell, "dsu": dsu_cell}

    # Save selected runs used by this aggregation.
    _write_selected_runs_csv(path=ROOT / args.selected_runs_csv, tables=tables)

    # Write CSV tables.
    for dataset in DATASETS:
        paper = PAPER_TABLES[dataset]
        rows: List[List[str]] = []
        rows.append(
            [
                "method",
                "dsu",
                "macro_f1(mean±std)",
                "mia_auc(mean±std)",
                "n_seeds",
                "seed_values_macro_f1",
                "seed_values_mia_auc",
            ]
        )

        # Retrain row (paper-only, unless you later add local retrain parsing).
        retrain_f1 = MetricAgg(mean=paper["retrain"]["f1"][0], std=paper["retrain"]["f1"][1], n=len(seeds))
        retrain_mia = MetricAgg(mean=paper["retrain"]["mia"][0], std=paper["retrain"]["mia"][1], n=len(seeds))
        rows.append(["Retrain", "-", _fmt_ms(retrain_f1), _fmt_ms(retrain_mia), str(len(seeds)), "-", "-"])

        for method in METHODS:
            for dsu_key, dsu_label in [("base", "×"), ("dsu", "✓")]:
                cell = tables.get(dataset, {}).get(method, {}).get(dsu_key)
                if cell is None:
                    continue
                rows.append(
                    [
                        METHOD_LABEL[method],
                        dsu_label,
                        _fmt_ms(cell.f1),
                        _fmt_ms(cell.mia),
                        str(cell.f1.n),
                        json.dumps({str(k): float(v) for k, v in cell.seed_values_f1.items()}, ensure_ascii=False),
                        json.dumps({str(k): float(v) for k, v in cell.seed_values_mia.items()}, ensure_ascii=False),
                    ]
                )

        out_csv = out_dir / ("table_I_20news.csv" if dataset == "20newsgroups" else "table_II_yahoo.csv")
        _write_csv(out_csv, rows)

    # Diff report.
    diff_lines: List[str] = []
    diff_lines.append("# Reproduction diff report (LLM part)")
    diff_lines.append("")
    diff_lines.append("This compares locally aggregated results (from existing `history.json`) to the paper tables.")
    diff_lines.append("")
    diff_lines.append("## Local sources")
    diff_lines.append(f"- Base seed42 root: `{args.base_seed42_root}`")
    diff_lines.append(f"- Base multi-seed root: `{args.base_ms_root}`")
    diff_lines.append(f"- DSU bestcfg root: `{args.dsu_root}`")
    diff_lines.append(f"- Seed-aligned backfill root: `{args.seed_aligned_root}`")
    diff_lines.append(f"- DSU meta json: `{args.meta_json}`")
    diff_lines.append(f"- Seeds: {seeds}")
    diff_lines.append(f"- Require seed-aligned snapshot: `{require_seed_aligned_snapshot}`")
    diff_lines.append(f"- MIA protocol: `{args.mia_protocol}`")
    diff_lines.append(f"- MIA metric: `{args.mia_metric}`")
    if mia_override_csv is not None:
        diff_lines.append(f"- MIA override csv: `{mia_override_csv}`")
    if run_override_csv is not None:
        diff_lines.append(f"- Run override csv: `{run_override_csv}`")
    diff_lines.append(f"- Selected runs csv: `{ROOT / args.selected_runs_csv}`")
    diff_lines.append("")

    if missing:
        diff_lines.append("## Missing runs / warnings")
        for m in missing:
            diff_lines.append(f"- {m}")
        diff_lines.append("")

    # Diagnostics: check whether the different seeds actually correspond to different DFL snapshots.
    diff_lines.append("## Diagnostics")
    diff_lines.append("")
    diff_lines.append(
        "- This repo historically used different notions of \"seed\" (DFL training seed vs DFU/unlearning seed). "
        "We attempt to infer the DFL snapshot used per run from sibling `dfu_config.json`."
    )
    diff_lines.append(
        "- If all rows for a method/dataset point to the SAME `dfl_snapshot`, then the reported variance is NOT from "
        "different DFL partitions/training; this often explains mismatches in the paper's reported std."
    )
    diff_lines.append(
        "- Note: MIA numbers in old `history.json` may have been computed on forget-vs-retain (both train) rather than "
        "member-vs-nonmember. Use `scripts/eval_unlearning_detectors.py` for the intended audit."
    )
    diff_lines.append("")

    for dataset in DATASETS:
        diff_lines.append(f"## {dataset}")
        paper = PAPER_TABLES[dataset]
        diff_lines.append("")
        diff_lines.append("| Method | DSU | Local F1 | Paper F1 | Δmean | Δstd | Local MIA | Paper MIA | Δmean | Δstd | n |")
        diff_lines.append("|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for method in METHODS:
            for dsu_key, dsu_label, paper_key in [("base", "×", "base"), ("dsu", "✓", "dsu")]:
                cell = tables.get(dataset, {}).get(method, {}).get(dsu_key)
                if cell is None:
                    continue
                pf1 = paper[method][paper_key]["f1"]
                pmia = paper[method][paper_key]["mia"]

                df1_mean, df1_std = _diff(cell.f1, pf1)
                dmia_mean, dmia_std = _diff(cell.mia, pmia)

                diff_lines.append(
                    "| "
                    + " | ".join(
                        [
                            METHOD_LABEL[method],
                            dsu_label,
                            _fmt_ms(cell.f1),
                            f"{pf1[0]:.2f}±{pf1[1]:.2f}",
                            "-" if df1_mean is None else f"{df1_mean:+.2f}",
                            "-" if df1_std is None else f"{df1_std:+.2f}",
                            _fmt_ms(cell.mia),
                            f"{pmia[0]:.2f}±{pmia[1]:.2f}",
                            "-" if dmia_mean is None else f"{dmia_mean:+.2f}",
                            "-" if dmia_std is None else f"{dmia_std:+.2f}",
                            str(cell.f1.n),
                        ]
                    )
                    + " |"
                )
        diff_lines.append("")

        # Single-seed (diagnose_seed) report for quick sanity-checking.
        diag_seed = int(args.diagnose_seed)
        diff_lines.append(f"### Single-seed view (seed={diag_seed})")
        diff_lines.append("")
        diff_lines.append("| Method | DSU | Local F1 (seed) | Local MIA (seed) | dfl_snapshot | dfu_seed | history.json |")
        diff_lines.append("|---|:---:|---:|---:|---|---:|---|")
        for method in METHODS:
            for dsu_key, dsu_label in [("base", "×"), ("dsu", "✓")]:
                cell = tables.get(dataset, {}).get(method, {}).get(dsu_key)
                if cell is None:
                    continue
                f1v = cell.seed_values_f1.get(diag_seed)
                miav = cell.seed_values_mia.get(diag_seed)
                snap = cell.dfl_snapshots.get(diag_seed, "-")
                dfu_seed = cell.dfu_seeds.get(diag_seed)
                hp = cell.run_paths.get(diag_seed, "-")
                diff_lines.append(
                    "| "
                    + " | ".join(
                        [
                            METHOD_LABEL[method],
                            dsu_label,
                            "-" if f1v is None else f"{f1v:.2f}",
                            "-" if miav is None else f"{miav:.2f}",
                            f"`{snap}`" if snap != "-" else "-",
                            "-" if dfu_seed is None else str(dfu_seed),
                            f"`{hp}`" if hp != "-" else "-",
                        ]
                    )
                    + " |"
                )
        diff_lines.append("")

    out_md = out_dir / "repro_diff_report.md"
    out_md.write_text("\n".join(diff_lines) + "\n", encoding="utf-8")

    # ------------------------------------------------------------
    # Figures (from existing sweep JSONs / cached plots)
    # ------------------------------------------------------------

    def _dataset_short(name: str) -> str:
        return {"20newsgroups": "20news", "yahoo_subset": "yahoo"}.get(name, name)

    def _load_sweep_json(dataset: str, sweep_dir: str, sweep_name: str, method: str) -> Optional[Dict[str, Any]]:
        root = ROOT / "final-figure" / _dataset_short(dataset) / sweep_dir
        if not root.exists():
            return None
        for p in sorted(root.glob(f"*{sweep_name}*4seeds-box.json")):
            try:
                obj = _read_json(p)
            except Exception:
                continue
            if obj.get("dataset") != dataset:
                continue
            if obj.get("algorithm") != method:
                continue
            return obj
        return None

    def _plot_sweep_grid(dataset: str, sweep: str, *, out_path: Path) -> None:
        sweep_dir = "lora-select" if sweep == "lora_ratio" else "agent-select"
        sweep_name = "lora-ratio" if sweep == "lora_ratio" else "agent-count"

        figs: List[Tuple[str, Dict[str, Any]]] = []
        for method in METHODS:
            obj = _load_sweep_json(dataset, sweep_dir, sweep_name, method)
            if obj is None:
                continue
            figs.append((method, obj))

        if not figs:
            return

        out_path.parent.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        axes = axes.reshape(-1)

        for ax, (method, obj) in zip(axes, figs):
            xs = obj.get("xs") or []
            values = obj.get("values") or {}
            data = []
            for x in xs:
                k = str(x)
                vs = values.get(k, [])
                if not isinstance(vs, list):
                    vs = []
                data.append([float(v) for v in vs if _safe_float(v) is not None])

            positions = list(range(len(xs)))
            ax.boxplot(
                data,
                positions=positions,
                widths=0.6,
                patch_artist=True,
                boxprops=dict(facecolor="#5780A4", alpha=0.35, edgecolor="#5780A4"),
                medianprops=dict(color="#1f1f1f", linewidth=1.2),
                whiskerprops=dict(color="#5780A4"),
                capprops=dict(color="#5780A4"),
                flierprops=dict(marker="o", markersize=2, markerfacecolor="#5780A4", alpha=0.35),
            )

            means = [float(np.mean(v)) if v else float("nan") for v in data]
            ax.plot(positions, means, color="#1f1f1f", linewidth=1.2, marker="o", markersize=3)

            ax.set_title(METHOD_LABEL[method])
            ax.set_xticks(positions)
            ax.set_xticklabels([str(x) for x in xs], rotation=0)
            ax.set_ylabel("Macro-F1 (%)")
            ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.4)

        # Hide unused subplots if any.
        for i in range(len(figs), len(axes)):
            axes[i].axis("off")

        fig.suptitle(f"{dataset} sweep: {sweep}", fontsize=14)
        fig.savefig(out_path, dpi=200)
        plt.close(fig)

    # Sweep figures (Fig.4–7 equivalents).
    for dataset in DATASETS:
        _plot_sweep_grid(dataset, "agent_count", out_path=out_dir / f"fig_{dataset}_agent_count_sweep.png")
        _plot_sweep_grid(dataset, "lora_ratio", out_path=out_dir / f"fig_{dataset}_lora_ratio_sweep.png")

    # Ablation bars (Fig.9–10 equivalents): reuse cached plots if present.
    ablation_src = {
        "20newsgroups": ROOT / "figures" / "20news-ablation-bars-bestcfg-4seeds.png",
        "yahoo_subset": ROOT / "figures" / "yahoo-ablation-bars-bestcfg-4seeds.png",
    }
    for dataset, src in ablation_src.items():
        if src.exists():
            shutil.copy2(src, out_dir / f"fig_{dataset}_ablation_as_ls.png")

    print(f"Wrote: {out_md}")
    print(f"Wrote: {out_dir / 'table_I_20news.csv'}")
    print(f"Wrote: {out_dir / 'table_II_yahoo.csv'}")
    print(f"Wrote: {ROOT / args.selected_runs_csv}")


if __name__ == "__main__":
    main()
