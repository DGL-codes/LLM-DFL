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
TAG_RE = re.compile(
    r"^mia_grid_(?P<dataset>20newsgroups|yahoo_subset)_seed(?P<seed>\d+)_"
    r"(?P<algo>d-[a-z]+)_(?P<strategy>full_all|full_ours|ours_all|ours_ours)"
    r"(?P<suffix>.*)_nonmemberVAL$"
)
DATASET_ORDER = ["20newsgroups", "yahoo_subset"]
ALGO_ORDER = ["d-federaser", "d-fedosd", "d-fedrecovery", "d-oblivionis"]
STRATEGY_ORDER = ["full_all", "full_ours", "ours_all", "ours_ours"]


def _load_json(path: Path) -> Optional[Dict]:
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


def _pick(data: Dict, model: str, metric: str) -> Optional[float]:
    node = (
        (data.get("detectors") or {})
        .get(model, {})
        .get("methods", {})
        .get("loss", {})
        .get("result", {})
    )
    return _to_float(node.get(metric))


def _mean_std(values: Iterable[Optional[float]]) -> Tuple[Optional[float], Optional[float]]:
    arr = [float(v) for v in values if v is not None]
    if not arr:
        return None, None
    if len(arr) == 1:
        return arr[0], 0.0
    return mean(arr), pstdev(arr)


def _fmt_pm(mean_value: Optional[float], std_value: Optional[float], digits: int = 4) -> str:
    if mean_value is None:
        return "-"
    if std_value is None:
        return f"{mean_value:.{digits}f}"
    return f"{mean_value:.{digits}f}+/-{std_value:.{digits}f}"


def _strategy_label(strategy: str) -> str:
    return {
        "full_all": "Base",
        "ours_all": "AS",
        "full_ours": "LS",
        "ours_ours": "DSU",
    }.get(strategy, strategy)


def _order_key(row: Dict[str, object]) -> Tuple[int, int, int, str]:
    dataset = str(row.get("dataset"))
    algo = str(row.get("algorithm"))
    strategy = str(row.get("strategy"))
    suffix = str(row.get("tag_suffix") or "")
    return (
        DATASET_ORDER.index(dataset) if dataset in DATASET_ORDER else 99,
        ALGO_ORDER.index(algo) if algo in ALGO_ORDER else 99,
        STRATEGY_ORDER.index(strategy) if strategy in STRATEGY_ORDER else 99,
        suffix,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="artifacts/unlearning_audit/mia")
    ap.add_argument("--tag_contains", default="clean_tdb_nonmemberVAL")
    ap.add_argument("--out_csv", default="reports/clean_tdb_mia_nonmember_val.csv")
    ap.add_argument("--out_md", default="reports/clean_tdb_mia_nonmember_val.md")
    args = ap.parse_args()

    root = ROOT / args.root
    per_seed: List[Dict[str, object]] = []
    for path in sorted(root.glob("*/mia_audit.json")):
        data = _load_json(path)
        if not data:
            continue
        tag = str(data.get("tag") or "")
        if args.tag_contains and args.tag_contains not in tag:
            continue
        m = TAG_RE.match(tag)
        if not m:
            continue
        row: Dict[str, object] = {
            "dataset": m.group("dataset"),
            "algorithm": m.group("algo"),
            "strategy": m.group("strategy"),
            "setting": _strategy_label(m.group("strategy")),
            "tag_suffix": m.group("suffix") or "",
            "seed": int(m.group("seed")),
            "dfl_auc_sym": _pick(data, "dfl", "auc_sym"),
            "dfu_auc_sym": _pick(data, "dfu", "auc_sym"),
            "retrain_auc_sym": _pick(data, "retrain", "auc_sym"),
            "dfl_adv": _pick(data, "dfl", "adv"),
            "dfu_adv": _pick(data, "dfu", "adv"),
            "retrain_adv": _pick(data, "retrain", "adv"),
            "path": str(path),
        }
        per_seed.append(row)

    buckets: Dict[Tuple[str, str, str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in per_seed:
        buckets[
            (
                str(row["dataset"]),
                str(row["algorithm"]),
                str(row["strategy"]),
                str(row["tag_suffix"]),
            )
        ].append(row)

    metric_names = [
        "dfl_auc_sym",
        "dfu_auc_sym",
        "retrain_auc_sym",
        "dfl_adv",
        "dfu_adv",
        "retrain_adv",
    ]
    agg_rows: List[Dict[str, object]] = []
    for (dataset, algo, strategy, suffix), rows in sorted(buckets.items()):
        out: Dict[str, object] = {
            "dataset": dataset,
            "algorithm": algo,
            "strategy": strategy,
            "setting": _strategy_label(strategy),
            "tag_suffix": suffix,
            "n": len(rows),
            "seeds": ",".join(str(r["seed"]) for r in sorted(rows, key=lambda x: int(x["seed"]))),
        }
        for metric in metric_names:
            m, s = _mean_std(r.get(metric) for r in rows)
            out[f"{metric}_mean"] = m
            out[f"{metric}_std"] = s
        agg_rows.append(out)

    agg_rows.sort(key=_order_key)
    out_csv = ROOT / args.out_csv
    out_md = ROOT / args.out_md
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset",
        "algorithm",
        "setting",
        "strategy",
        "tag_suffix",
        "n",
        "seeds",
    ]
    for metric in metric_names:
        fields.extend([f"{metric}_mean", f"{metric}_std"])
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in agg_rows:
            writer.writerow({k: row.get(k) for k in fields})

    lines = [
        "# Clean TDB MIA audit (nonmember=val)",
        "",
        f"- Source root: `{args.root}`",
        f"- Tag filter: `{args.tag_contains}`",
        f"- Per-seed rows: {len(per_seed)}",
        f"- Aggregated cells: {len(agg_rows)}",
        "",
        "| Dataset | Method | Setting | n | Seeds | DFL AUC | DFU AUC | Retrain AUC | DFL Adv | DFU Adv | Retrain Adv |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in agg_rows:
        lines.append(
            "| {dataset} | {algorithm} | {setting} | {n} | {seeds} | {dfl_auc} | {dfu_auc} | {retrain_auc} | {dfl_adv} | {dfu_adv} | {retrain_adv} |".format(
                dataset=row["dataset"],
                algorithm=row["algorithm"],
                setting=row["setting"],
                n=row["n"],
                seeds=row["seeds"],
                dfl_auc=_fmt_pm(row.get("dfl_auc_sym_mean"), row.get("dfl_auc_sym_std")),
                dfu_auc=_fmt_pm(row.get("dfu_auc_sym_mean"), row.get("dfu_auc_sym_std")),
                retrain_auc=_fmt_pm(row.get("retrain_auc_sym_mean"), row.get("retrain_auc_sym_std")),
                dfl_adv=_fmt_pm(row.get("dfl_adv_mean"), row.get("dfl_adv_std")),
                dfu_adv=_fmt_pm(row.get("dfu_adv_mean"), row.get("dfu_adv_std")),
                retrain_adv=_fmt_pm(row.get("retrain_adv_mean"), row.get("retrain_adv_std")),
            )
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
