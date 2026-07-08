#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Optional, Tuple


ROOT = Path(__file__).resolve().parent.parent

BD_RE = re.compile(
    r"^bd_grid_(?P<dataset>20newsgroups|yahoo_subset)_seed(?P<seed>\d+)_(?P<algo>d-[a-z]+)_(?P<strategy>full_all|full_ours|ours_all|ours_ours)(?P<suffix>.*)$"
)

STRATEGY_LABELS = {
    "full_all": "Base",
    "ours_all": "AS",
    "full_ours": "LS",
    "ours_ours": "AS+LS",
}
STRATEGY_ORDER = ["full_all", "ours_all", "full_ours", "ours_ours"]
ALGO_ORDER = ["d-federaser", "d-fedosd", "d-fedrecovery", "d-oblivionis"]
DATASET_ORDER = ["20newsgroups", "yahoo_subset"]
RATE_ORDER = ["0p1", "0p2"]


def _extract_bd_rate(suffix: str) -> str:
    m = re.search(r"_bd(?P<rate>\d+p\d+)", suffix or "")
    return m.group("rate") if m else ""


def _rate_sort_key(rate: str) -> Tuple[int, str]:
    if rate in RATE_ORDER:
        return (RATE_ORDER.index(rate), rate)
    return (len(RATE_ORDER), rate)


def _to_float(x: object) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _load_json(path: Path) -> Optional[Dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _model_metric(data: Dict, model: str, family: str, key: str) -> Optional[float]:
    return _to_float((((data.get("models") or {}).get(model) or {}).get(family) or {}).get(key))


def _mean_std(values: Iterable[Optional[float]]) -> Tuple[Optional[float], Optional[float]]:
    arr = [v for v in values if v is not None]
    if not arr:
        return None, None
    if len(arr) == 1:
        return arr[0], 0.0
    return mean(arr), pstdev(arr)


def _fmt(x: Optional[float], digits: int = 4) -> str:
    return "-" if x is None else f"{x:.{digits}f}"


def _fmt_pm(m: Optional[float], s: Optional[float], digits: int = 4) -> str:
    if m is None:
        return "-"
    if s is None:
        return _fmt(m, digits)
    return f"{m:.{digits}f}±{s:.{digits}f}"


def _parse_roots(raw: str) -> List[Path]:
    roots = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        p = Path(item)
        roots.append(p if p.is_absolute() else ROOT / p)
    return roots


def load_seed_rows(
    roots: List[Path],
    f1_tol: float,
    asr_drop_tol: float,
    tag_contains: str = "",
    asr_family: str = "asr",
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    seen = set()
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.glob("*/backdoor_audit.json")):
            data = _load_json(path)
            if not data:
                continue
            tag = str(data.get("tag") or "")
            if tag_contains and tag_contains not in tag:
                continue
            m = BD_RE.match(tag)
            if not m:
                continue
            key = (m.group("dataset"), m.group("algo"), m.group("strategy"), int(m.group("seed")), tag)
            if key in seen:
                continue
            seen.add(key)
            suffix = m.group("suffix") or ""
            bd_rate = _extract_bd_rate(suffix)
            dfl_f1 = _model_metric(data, "dfl", "clean", "macro_f1")
            dfl_asr = _model_metric(data, "dfl", asr_family, "asr")
            dfu_f1 = _model_metric(data, "dfu", "clean", "macro_f1")
            dfu_asr = _model_metric(data, "dfu", asr_family, "asr")
            retrain_f1 = _model_metric(data, "retrain", "clean", "macro_f1")
            retrain_asr = _model_metric(data, "retrain", asr_family, "asr")
            f1_delta = None if dfl_f1 is None or dfu_f1 is None else dfu_f1 - dfl_f1
            asr_delta = None if dfl_asr is None or dfu_asr is None else dfu_asr - dfl_asr
            asr_drop = None if asr_delta is None else -asr_delta
            f1_ok = f1_delta is not None and f1_delta >= -f1_tol
            asr_ok = asr_drop is not None and asr_drop >= asr_drop_tol
            rows.append(
                {
                    "dataset": m.group("dataset"),
                    "algorithm": m.group("algo"),
                    "strategy": m.group("strategy"),
                    "strategy_label": STRATEGY_LABELS[m.group("strategy")],
                    "seed": int(m.group("seed")),
                    "bd_rate": bd_rate,
                    "tag": tag,
                    "path": str(path),
                    "eval_scope": data.get("eval_scope"),
                    "dfu_state_mode": data.get("dfu_state_mode"),
                    "dfl_n_agents": ((data.get("models") or {}).get("dfl") or {}).get("n_agents"),
                    "dfu_n_agents": ((data.get("models") or {}).get("dfu") or {}).get("n_agents"),
                    "retrain_n_agents": ((data.get("models") or {}).get("retrain") or {}).get("n_agents"),
                    "dfl_clean_f1": dfl_f1,
                    "dfl_asr": dfl_asr,
                    "dfu_clean_f1": dfu_f1,
                    "dfu_asr": dfu_asr,
                    "retrain_clean_f1": retrain_f1,
                    "retrain_asr": retrain_asr,
                    "f1_delta_vs_dfl": f1_delta,
                    "asr_delta_vs_dfl": asr_delta,
                    "asr_drop_vs_dfl": asr_drop,
                    "f1_ok": int(f1_ok),
                    "asr_ok": int(asr_ok),
                    "joint_ok": int(f1_ok and asr_ok),
                }
            )
    return sorted(
        rows,
        key=lambda r: (
            str(r["dataset"]),
            str(r["algorithm"]),
            _rate_sort_key(str(r.get("bd_rate") or "")),
            int(r["seed"]),
            STRATEGY_ORDER.index(str(r["strategy"])),
        ),
    )


def select_best_per_cell(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    buckets: Dict[Tuple[str, str, str, int, str], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        buckets[
            (
                str(row["dataset"]),
                str(row["algorithm"]),
                str(row.get("bd_rate") or ""),
                int(row["seed"]),
                str(row["strategy"]),
            )
        ].append(row)

    def score(row: Dict[str, object]) -> Tuple[int, int, int, int, float, float]:
        f1_delta = _to_float(row.get("f1_delta_vs_dfl"))
        asr_drop = _to_float(row.get("asr_drop_vs_dfl"))
        has_asr = int(asr_drop is not None)
        has_f1 = int(f1_delta is not None)
        return (
            int(row.get("joint_ok") or 0),
            has_asr,
            has_f1,
            int(row.get("asr_ok") or 0),
            int(row.get("f1_ok") or 0),
            asr_drop if asr_drop is not None else float("-inf"),
            f1_delta if f1_delta is not None else float("-inf"),
        )

    selected = [max(group, key=score) for group in buckets.values()]
    return sorted(
        selected,
        key=lambda r: (
            str(r["dataset"]),
            str(r["algorithm"]),
            _rate_sort_key(str(r.get("bd_rate") or "")),
            int(r["seed"]),
            STRATEGY_ORDER.index(str(r["strategy"])),
        ),
    )


def write_seed_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    fields = [
        "dataset",
        "algorithm",
        "bd_rate",
        "seed",
        "strategy",
        "strategy_label",
        "eval_scope",
        "dfu_state_mode",
        "dfl_n_agents",
        "dfu_n_agents",
        "retrain_n_agents",
        "dfl_clean_f1",
        "dfl_asr",
        "dfu_clean_f1",
        "dfu_asr",
        "retrain_clean_f1",
        "retrain_asr",
        "f1_delta_vs_dfl",
        "asr_delta_vs_dfl",
        "asr_drop_vs_dfl",
        "f1_ok",
        "asr_ok",
        "joint_ok",
        "tag",
        "path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in fields})


def build_agg(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    buckets: Dict[Tuple[str, str, str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        buckets[
            (
                str(row["dataset"]),
                str(row["algorithm"]),
                str(row.get("bd_rate") or ""),
                str(row["strategy"]),
            )
        ].append(row)

    out: List[Dict[str, object]] = []
    metrics = [
        "dfl_clean_f1",
        "dfl_asr",
        "dfu_clean_f1",
        "dfu_asr",
        "retrain_clean_f1",
        "retrain_asr",
        "f1_delta_vs_dfl",
        "asr_delta_vs_dfl",
        "asr_drop_vs_dfl",
    ]
    for key, group in buckets.items():
        dataset, algo, bd_rate, strategy = key
        row: Dict[str, object] = {
            "dataset": dataset,
            "algorithm": algo,
            "bd_rate": bd_rate,
            "strategy": strategy,
            "strategy_label": STRATEGY_LABELS[strategy],
            "n": len(group),
            "seeds": ",".join(str(int(r["seed"])) for r in sorted(group, key=lambda x: int(x["seed"]))),
            "eval_scope": ",".join(sorted({str(r.get("eval_scope")) for r in group if r.get("eval_scope") is not None})),
            "dfu_state_mode": ",".join(sorted({str(r.get("dfu_state_mode")) for r in group if r.get("dfu_state_mode") is not None})),
            "f1_ok_count": sum(int(r["f1_ok"]) for r in group),
            "asr_ok_count": sum(int(r["asr_ok"]) for r in group),
            "joint_ok_count": sum(int(r["joint_ok"]) for r in group),
        }
        for metric in metrics:
            m, s = _mean_std(_to_float(r.get(metric)) for r in group)
            row[f"{metric}_mean"] = m
            row[f"{metric}_std"] = s
        out.append(row)
    return sorted(
        out,
        key=lambda r: (
            _rate_sort_key(str(r.get("bd_rate") or "")),
            DATASET_ORDER.index(str(r["dataset"])),
            ALGO_ORDER.index(str(r["algorithm"])),
            STRATEGY_ORDER.index(str(r["strategy"])),
        ),
    )


def write_agg_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    fields = [
        "dataset",
        "algorithm",
        "bd_rate",
        "strategy",
        "strategy_label",
        "n",
        "seeds",
        "eval_scope",
        "dfu_state_mode",
        "dfl_clean_f1_mean",
        "dfl_clean_f1_std",
        "dfl_asr_mean",
        "dfl_asr_std",
        "dfu_clean_f1_mean",
        "dfu_clean_f1_std",
        "dfu_asr_mean",
        "dfu_asr_std",
        "retrain_clean_f1_mean",
        "retrain_clean_f1_std",
        "retrain_asr_mean",
        "retrain_asr_std",
        "f1_delta_vs_dfl_mean",
        "f1_delta_vs_dfl_std",
        "asr_drop_vs_dfl_mean",
        "asr_drop_vs_dfl_std",
        "f1_ok_count",
        "asr_ok_count",
        "joint_ok_count",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in fields})


def component_rows(seed_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    by_cell: Dict[Tuple[str, str, str, int], Dict[str, Dict[str, object]]] = defaultdict(dict)
    for row in seed_rows:
        by_cell[
            (
                str(row["dataset"]),
                str(row["algorithm"]),
                str(row.get("bd_rate") or ""),
                int(row["seed"]),
            )
        ][str(row["strategy"])] = row

    out: List[Dict[str, object]] = []
    for (dataset, algo, bd_rate, seed), strategies in sorted(by_cell.items()):
        base = strategies.get("full_all")
        if not base:
            continue
        base_f1 = _to_float(base.get("dfu_clean_f1"))
        base_asr = _to_float(base.get("dfu_asr"))
        for strategy in ["ours_all", "full_ours", "ours_ours"]:
            row = strategies.get(strategy)
            if not row:
                continue
            f1 = _to_float(row.get("dfu_clean_f1"))
            asr = _to_float(row.get("dfu_asr"))
            out.append(
                {
                    "dataset": dataset,
                    "algorithm": algo,
                    "bd_rate": bd_rate,
                    "seed": seed,
                    "component": STRATEGY_LABELS[strategy],
                    "strategy": strategy,
                    "f1_delta_vs_base": None if f1 is None or base_f1 is None else f1 - base_f1,
                    "asr_drop_vs_base": None if asr is None or base_asr is None else base_asr - asr,
                    "joint_ok": int(row["joint_ok"]),
                }
            )
    return out


def build_md(
    path: Path,
    seed_rows: List[Dict[str, object]],
    agg_rows: List[Dict[str, object]],
    component: List[Dict[str, object]],
    f1_tol: float,
    asr_drop_tol: float,
    out_csv: str,
    out_seed_csv: str,
    asr_family: str,
) -> None:
    agg_by_key = {
        (str(r["bd_rate"]), str(r["dataset"]), str(r["algorithm"]), str(r["strategy"])): r
        for r in agg_rows
    }
    comp_bucket: Dict[Tuple[str, str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in component:
        comp_bucket[
            (
                str(row.get("bd_rate") or ""),
                str(row["dataset"]),
                str(row["algorithm"]),
                str(row["component"]),
            )
        ].append(row)

    failures = [r for r in seed_rows if int(r["joint_ok"]) != 1]
    f1_missing = [r for r in seed_rows if _to_float(r.get("f1_delta_vs_dfl")) is None]
    asr_missing = [r for r in seed_rows if _to_float(r.get("asr_drop_vs_dfl")) is None]
    f1_failures = [
        r for r in seed_rows
        if _to_float(r.get("f1_delta_vs_dfl")) is not None and int(r["f1_ok"]) != 1
    ]
    asr_failures = [
        r for r in seed_rows
        if _to_float(r.get("asr_drop_vs_dfl")) is not None and int(r["asr_ok"]) != 1
    ]

    lines: List[str] = []
    lines.append("# Backdoor Effective Audit")
    lines.append("")
    lines.append(f"- ASR metric family: `{asr_family}`.")
    lines.append(f"- Criterion: `DFU ASR <= DFL ASR - {asr_drop_tol:g}` and `DFU clean F1 >= DFL clean F1 - {f1_tol:g}`.")
    lines.append(f"- Aggregate CSV: `{out_csv}`")
    lines.append(f"- Per-seed CSV: `{out_seed_csv}`")
    lines.append(f"- Rows: `{len(seed_rows)}` per-seed, `{len(agg_rows)}` aggregate.")
    lines.append(
        f"- Failures with complete metrics: `{len(f1_failures)}` F1, `{len(asr_failures)}` ASR."
    )
    if f1_missing or asr_missing:
        lines.append(
            f"- Missing metrics still running/incomplete: `{len(f1_missing)}` F1, `{len(asr_missing)}` ASR."
        )
    lines.append(f"- Joint failures including missing metrics: `{len(failures)}`.")
    lines.append("")

    rates_present = sorted({str(r.get("bd_rate") or "") for r in seed_rows}, key=_rate_sort_key)
    for bd_rate in rates_present:
        rate_label = bd_rate.replace("p", ".") if bd_rate else "unknown"
        lines.append(f"## rate={rate_label}")
        lines.append("")
        for dataset in DATASET_ORDER:
            lines.append(f"### {dataset}")
            lines.append("")
            lines.append("| Method | Strategy | n | DFL F1 | DFL ASR | DFU F1 | DFU ASR | ΔF1 vs DFL | ASR drop | pass seeds |")
            lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
            for algo in ALGO_ORDER:
                for strategy in STRATEGY_ORDER:
                    row = agg_by_key.get((bd_rate, dataset, algo, strategy))
                    if not row:
                        lines.append(f"| {algo} | {STRATEGY_LABELS[strategy]} | 0 | - | - | - | - | - | - | 0/0 |")
                        continue
                    n = int(row["n"])
                    lines.append(
                        "| "
                        f"{algo} | {row['strategy_label']} | {n} | "
                        f"{_fmt_pm(_to_float(row.get('dfl_clean_f1_mean')), _to_float(row.get('dfl_clean_f1_std')))} | "
                        f"{_fmt_pm(_to_float(row.get('dfl_asr_mean')), _to_float(row.get('dfl_asr_std')))} | "
                        f"{_fmt_pm(_to_float(row.get('dfu_clean_f1_mean')), _to_float(row.get('dfu_clean_f1_std')))} | "
                        f"{_fmt_pm(_to_float(row.get('dfu_asr_mean')), _to_float(row.get('dfu_asr_std')))} | "
                        f"{_fmt_pm(_to_float(row.get('f1_delta_vs_dfl_mean')), _to_float(row.get('f1_delta_vs_dfl_std')))} | "
                        f"{_fmt_pm(_to_float(row.get('asr_drop_vs_dfl_mean')), _to_float(row.get('asr_drop_vs_dfl_std')))} | "
                        f"{int(row['joint_ok_count'])}/{n} |"
                    )
            lines.append("")

            lines.append("#### Component Deltas vs Base")
            lines.append("")
            lines.append("| Method | Component | ΔF1 vs Base | ASR drop vs Base | pass seeds |")
            lines.append("|---|---|---:|---:|---:|")
            for algo in ALGO_ORDER:
                for label in ["AS", "LS", "AS+LS"]:
                    group = comp_bucket.get((bd_rate, dataset, algo, label), [])
                    f1_m, f1_s = _mean_std(_to_float(r.get("f1_delta_vs_base")) for r in group)
                    asr_m, asr_s = _mean_std(_to_float(r.get("asr_drop_vs_base")) for r in group)
                    passes = sum(int(r["joint_ok"]) for r in group)
                    lines.append(f"| {algo} | {label} | {_fmt_pm(f1_m, f1_s)} | {_fmt_pm(asr_m, asr_s)} | {passes}/{len(group)} |")
            lines.append("")

    lines.append("## Rescue Queue")
    lines.append("")
    if not failures:
        lines.append("- No joint failures under the configured criterion.")
    else:
        lines.append("| Rate | Dataset | Method | Seed | Strategy | DFL F1 | DFU F1 | ΔF1 | DFL ASR | DFU ASR | ASR drop | Flags |")
        lines.append("|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|")
        for row in failures:
            flags = []
            if _to_float(row.get("f1_delta_vs_dfl")) is None:
                flags.append("F1-missing")
            elif int(row["f1_ok"]) != 1:
                flags.append("F1")
            if _to_float(row.get("asr_drop_vs_dfl")) is None:
                flags.append("ASR-missing")
            elif int(row["asr_ok"]) != 1:
                flags.append("ASR")
            lines.append(
                "| "
                f"{(str(row.get('bd_rate') or '').replace('p', '.') if row.get('bd_rate') else '-')} | "
                f"{row['dataset']} | {row['algorithm']} | {row['seed']} | {row['strategy_label']} | "
                f"{_fmt(_to_float(row.get('dfl_clean_f1')))} | {_fmt(_to_float(row.get('dfu_clean_f1')))} | {_fmt(_to_float(row.get('f1_delta_vs_dfl')))} | "
                f"{_fmt(_to_float(row.get('dfl_asr')))} | {_fmt(_to_float(row.get('dfu_asr')))} | {_fmt(_to_float(row.get('asr_drop_vs_dfl')))} | "
                f"{','.join(flags)} |"
            )
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bd_root", default="artifacts/unlearning_audit/backdoor")
    ap.add_argument("--out_md", default="reports/backdoor_effective_audit.md")
    ap.add_argument("--out_csv", default="reports/backdoor_effective_audit.csv")
    ap.add_argument("--out_seed_csv", default="reports/backdoor_effective_audit_seed.csv")
    ap.add_argument("--f1_tolerance", type=float, default=0.01)
    ap.add_argument("--asr_drop_tolerance", type=float, default=0.01)
    ap.add_argument("--tag_contains", default="")
    ap.add_argument(
        "--asr_family",
        default="asr",
        choices=["asr", "asr_non_target"],
        help="Which ASR field to aggregate. `asr_non_target` excludes examples whose true label is already the target label.",
    )
    ap.add_argument(
        "--select_best_per_cell",
        action="store_true",
        help="For duplicate dataset/algorithm/seed/strategy rows, keep the best reproducible tagged result.",
    )
    args = ap.parse_args()

    seed_rows = load_seed_rows(
        _parse_roots(args.bd_root),
        args.f1_tolerance,
        args.asr_drop_tolerance,
        args.tag_contains,
        args.asr_family,
    )
    if args.select_best_per_cell:
        seed_rows = select_best_per_cell(seed_rows)
    agg_rows = build_agg(seed_rows)
    component = component_rows(seed_rows)

    out_csv = ROOT / args.out_csv
    out_seed_csv = ROOT / args.out_seed_csv
    out_md = ROOT / args.out_md
    write_agg_csv(out_csv, agg_rows)
    write_seed_csv(out_seed_csv, seed_rows)
    build_md(
        out_md,
        seed_rows,
        agg_rows,
        component,
        args.f1_tolerance,
        args.asr_drop_tolerance,
        args.out_csv,
        args.out_seed_csv,
        args.asr_family,
    )
    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_seed_csv}")
    print(f"Wrote: {out_md}")


if __name__ == "__main__":
    main()
