#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Optional, Tuple


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_ROOT = Path(os.environ.get("LLMDFL_EXPERIMENT_DIR", "实验结果/运行产物"))


def _default_audit_subroots(name: str) -> str:
    roots = []
    candidate = DEFAULT_RESULTS_ROOT / "artifacts" / "unlearning_audit" / name
    if candidate.exists():
        try:
            roots.append(str(candidate.relative_to(ROOT)))
        except Exception:
            roots.append(str(candidate))
    roots.append(f"artifacts/unlearning_audit/{name}")
    seen = []
    for r in roots:
        if r not in seen:
            seen.append(r)
    return ",".join(seen)


BD_RE = re.compile(
    r"^bd_grid_(?P<dataset>20newsgroups|yahoo_subset)_seed(?P<seed>\d+)_(?P<algo>d-[a-z]+)_(?P<strategy>full_all|full_ours|ours_all|ours_ours)(?P<suffix>.*)$"
)
MIA_RE = re.compile(
    r"^mia_grid_(?P<dataset>20newsgroups|yahoo_subset)_seed(?P<seed>\d+)_(?P<algo>d-[a-z]+)_(?P<strategy>full_all|full_ours|ours_all|ours_ours)(?P<suffix>.*)_nonmemberVAL$"
)
STRATEGY_ORDER = ["full_all", "full_ours", "ours_all", "ours_ours"]


def _load_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _to_float(x: object) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _extract_bd_metrics(data: Dict) -> Dict[str, Optional[float]]:
    models = data.get("models") or {}

    def pick(model: str, key: str, subkey: str) -> Optional[float]:
        node = (models.get(model) or {}).get(key) or {}
        return _to_float(node.get(subkey))

    return {
        "dfl_clean_f1": pick("dfl", "clean", "macro_f1"),
        "dfl_asr": pick("dfl", "asr", "asr"),
        "dfu_clean_f1": pick("dfu", "clean", "macro_f1"),
        "dfu_asr": pick("dfu", "asr", "asr"),
        "retrain_clean_f1": pick("retrain", "clean", "macro_f1"),
        "retrain_asr": pick("retrain", "asr", "asr"),
    }


def _extract_mia_metrics(data: Dict) -> Dict[str, Optional[float]]:
    det = data.get("detectors") or {}

    def pick(model: str, metric: str) -> Optional[float]:
        node = (
            det.get(model, {})
            .get("methods", {})
            .get("loss", {})
            .get("result", {})
        )
        return _to_float(node.get(metric))

    return {
        "mia_dfl_adv": pick("dfl", "adv"),
        "mia_dfu_adv": pick("dfu", "adv"),
        "mia_retrain_adv": pick("retrain", "adv"),
        "mia_dfl_auc_sym": pick("dfl", "auc_sym"),
        "mia_dfu_auc_sym": pick("dfu", "auc_sym"),
        "mia_retrain_auc_sym": pick("retrain", "auc_sym"),
    }


def _fmt_pm(values: List[float], digits: int = 4) -> str:
    if not values:
        return "-"
    if len(values) == 1:
        return f"{values[0]:.{digits}f}"
    return f"{mean(values):.{digits}f}±{pstdev(values):.{digits}f}"


def _fmt_tuple(m: Optional[float], s: Optional[float], digits: int = 4) -> str:
    if m is None:
        return "-"
    if s is None:
        return f"{m:.{digits}f}"
    return f"{m:.{digits}f}±{s:.{digits}f}"


def _mean_std(values: Iterable[float]) -> Tuple[Optional[float], Optional[float]]:
    arr = list(values)
    if not arr:
        return None, None
    if len(arr) == 1:
        return arr[0], 0.0
    return mean(arr), pstdev(arr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_md", default="reports/unlearning_detector_validation.md")
    ap.add_argument("--out_csv", default="reports/unlearning_detector_validation.csv")
    ap.add_argument("--bd_root", default=_default_audit_subroots("backdoor"))
    ap.add_argument("--mia_root", default=_default_audit_subroots("mia"))
    ap.add_argument("--tag_contains", default="", help="Only aggregate audit tags containing this substring.")
    args = ap.parse_args()

    bd_roots = [ROOT / item.strip() for item in str(args.bd_root).split(",") if item.strip()]
    mia_roots = [ROOT / item.strip() for item in str(args.mia_root).split(",") if item.strip()]

    rows_by_key: Dict[Tuple[str, str, str, int], Dict[str, Optional[float]]] = {}

    seen_bd = set()
    for bd_root in bd_roots:
        if not bd_root.exists():
            continue
        for p in sorted(bd_root.glob("*/backdoor_audit.json")):
            rp = str(p.resolve())
            if rp in seen_bd:
                continue
            seen_bd.add(rp)
            data = _load_json(p)
            if not data:
                continue
            tag = str(data.get("tag") or "")
            if args.tag_contains and args.tag_contains not in tag:
                continue
            m = BD_RE.match(tag)
            if not m:
                continue
            dataset = m.group("dataset")
            algo = m.group("algo")
            strategy = m.group("strategy")
            suffix = m.group("suffix") or ""
            seed = int(m.group("seed"))
            key = (dataset, algo, strategy, suffix, seed)
            rows_by_key.setdefault(key, {}).update(_extract_bd_metrics(data))

    seen_mia = set()
    for mia_root in mia_roots:
        if not mia_root.exists():
            continue
        for p in sorted(mia_root.glob("*/mia_audit.json")):
            rp = str(p.resolve())
            if rp in seen_mia:
                continue
            seen_mia.add(rp)
            data = _load_json(p)
            if not data:
                continue
            tag = str(data.get("tag") or "")
            if args.tag_contains and args.tag_contains not in tag:
                continue
            m = MIA_RE.match(tag)
            if not m:
                continue
            dataset = m.group("dataset")
            algo = m.group("algo")
            strategy = m.group("strategy")
            suffix = m.group("suffix") or ""
            seed = int(m.group("seed"))
            key = (dataset, algo, strategy, suffix, seed)
            rows_by_key.setdefault(key, {}).update(_extract_mia_metrics(data))

    # Per-seed rows with full metrics.
    seed_rows: List[Dict[str, object]] = []
    for (dataset, algo, strategy, suffix, seed), metrics in sorted(rows_by_key.items()):
        strategy_display = strategy + suffix
        row: Dict[str, object] = {
            "dataset": dataset,
            "algorithm": algo,
            "strategy": strategy,
            "strategy_suffix": suffix,
            "strategy_display": strategy_display,
            "seed": seed,
        }
        row.update(metrics)
        seed_rows.append(row)

    metric_names = [
        "dfl_clean_f1",
        "dfl_asr",
        "dfu_clean_f1",
        "dfu_asr",
        "retrain_clean_f1",
        "retrain_asr",
        "mia_dfl_adv",
        "mia_dfu_adv",
        "mia_retrain_adv",
        "mia_dfl_auc_sym",
        "mia_dfu_auc_sym",
        "mia_retrain_auc_sym",
    ]

    agg: Dict[Tuple[str, str, str], Dict[str, object]] = {}
    bucket: Dict[Tuple[str, str, str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in seed_rows:
        key = (
            str(row["dataset"]),
            str(row["algorithm"]),
            str(row["strategy"]),
            str(row.get("strategy_suffix") or ""),
        )
        bucket[key].append(row)

    for key, rows in sorted(bucket.items()):
        dataset, algo, strategy, suffix = key
        seeds = sorted(int(r["seed"]) for r in rows)
        out: Dict[str, object] = {
            "dataset": dataset,
            "algorithm": algo,
            "strategy": strategy,
            "strategy_suffix": suffix,
            "strategy_display": strategy + suffix,
            "n": len(rows),
            "seeds": ",".join(str(s) for s in seeds),
        }
        for metric in metric_names:
            vals = [_to_float(r.get(metric)) for r in rows]
            vals = [v for v in vals if v is not None]
            m, s = _mean_std(vals)
            out[f"{metric}_mean"] = m
            out[f"{metric}_std"] = s
        agg[key] = out

    out_csv = ROOT / args.out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    csv_fields = [
        "dataset",
        "algorithm",
        "strategy",
        "strategy_suffix",
        "strategy_display",
        "n",
        "seeds",
    ]
    for metric in metric_names:
        csv_fields.append(f"{metric}_mean")
        csv_fields.append(f"{metric}_std")

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields)
        w.writeheader()
        for key in sorted(agg):
            w.writerow(agg[key])

    out_md = ROOT / args.out_md
    lines: List[str] = []
    lines.append("# Unlearning Detector Validation (20News + Yahoo)")
    lines.append("")
    lines.append("This report uses two complementary signals:")
    lines.append("")
    lines.append("- **Behavioral forgetting**: backdoor ASR drop from DFL to DFU/Retrain.")
    lines.append("- **Membership signal**: MIA (`nonmember=val`) with `auc_sym/adv`.")
    lines.append("")
    lines.append(f"- Aggregated rows: `{len(agg)}`")
    lines.append(f"- Per-seed rows: `{len(seed_rows)}`")
    lines.append(f"- CSV: `{args.out_csv}`")
    lines.append("")

    datasets = ["20newsgroups", "yahoo_subset"]
    algo_order = ["d-federaser", "d-fedosd", "d-fedrecovery", "d-oblivionis"]

    def _sort_key(row: Dict[str, object]) -> Tuple[int, int, str]:
        algo = str(row["algorithm"])
        strategy = str(row["strategy"])
        suffix = str(row.get("strategy_suffix") or "")
        return (
            algo_order.index(algo) if algo in algo_order else len(algo_order),
            STRATEGY_ORDER.index(strategy) if strategy in STRATEGY_ORDER else len(STRATEGY_ORDER),
            suffix,
        )

    for ds in datasets:
        lines.append(f"## {ds}")
        lines.append("")
        lines.append("| Method | Strategy | n | Seeds | DFL ASR | DFU ASR | Retrain ASR | DFU Clean F1 | Retrain Clean F1 | MIA adv (DFL/DFU/Retrain) |")
        lines.append("|---|---|---:|---|---:|---:|---:|---:|---:|---|")
        wrote = 0
        ds_rows = [row for key, row in agg.items() if str(row["dataset"]) == ds]
        for row in sorted(ds_rows, key=_sort_key):
            algo = str(row["algorithm"])
            strategy_display = str(row.get("strategy_display") or row.get("strategy"))
            adv_triplet = (
                _fmt_tuple(_to_float(row.get("mia_dfl_adv_mean")), _to_float(row.get("mia_dfl_adv_std")), 4)
                + " / "
                + _fmt_tuple(_to_float(row.get("mia_dfu_adv_mean")), _to_float(row.get("mia_dfu_adv_std")), 4)
                + " / "
                + _fmt_tuple(_to_float(row.get("mia_retrain_adv_mean")), _to_float(row.get("mia_retrain_adv_std")), 4)
            )
            lines.append(
                "| "
                f"{algo} | {strategy_display} | {int(row['n'])} | {row['seeds']} | "
                f"{_fmt_tuple(_to_float(row.get('dfl_asr_mean')), _to_float(row.get('dfl_asr_std')), 4)} | "
                f"{_fmt_tuple(_to_float(row.get('dfu_asr_mean')), _to_float(row.get('dfu_asr_std')), 4)} | "
                f"{_fmt_tuple(_to_float(row.get('retrain_asr_mean')), _to_float(row.get('retrain_asr_std')), 4)} | "
                f"{_fmt_tuple(_to_float(row.get('dfu_clean_f1_mean')), _to_float(row.get('dfu_clean_f1_std')), 4)} | "
                f"{_fmt_tuple(_to_float(row.get('retrain_clean_f1_mean')), _to_float(row.get('retrain_clean_f1_std')), 4)} | "
                f"{adv_triplet} |"
            )
            wrote += 1
        if wrote == 0:
            lines.append("| - | - | 0 | - | - | - | - | - | - | - |")
        lines.append("")

    lines.append("## Conclusion")
    lines.append("")
    lines.append("- Primary forgetting criterion is **ASR drop** (DFL high → DFU/Retrain low).")
    lines.append("- MIA (`nonmember=val`) is secondary evidence; lower `adv` indicates weaker membership signal.")
    lines.append("- Combined readout should check both forgetting strength and clean utility retention.")
    lines.append("")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_md}")


if __name__ == "__main__":
    main()
