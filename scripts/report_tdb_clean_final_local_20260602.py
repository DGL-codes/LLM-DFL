#!/usr/bin/env python3
"""Build the final local-ring TDB clean F1/MIA table.

This is a reporting-only helper. It combines:
- Base from the completed clean full sweep.
- AS/LS best settings from the k=1..9 and r=0.1..1.0 local sweeps.
- DSU best settings from the final local-ring joint k x r sweep.

No training, selection, or metric optimization happens here.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DATASET_ORDER = ["20newsgroups", "yahoo_subset"]
METHOD_ORDER = ["d-federaser", "d-fedosd", "d-fedrecovery", "d-oblivionis"]
METHOD_LABEL = {
    "d-federaser": "FedEraser",
    "d-fedosd": "FedOSD",
    "d-fedrecovery": "FedRecovery",
    "d-oblivionis": "D-Oblivionis",
}
DATASET_LABEL = {"20newsgroups": "20News", "yahoo_subset": "Yahoo"}


def fnum(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def pct(value: Any) -> str:
    x = fnum(value)
    return "-" if x is None else f"{100.0 * x:.2f}"


def pct_std(mean_value: Any, std_value: Any) -> str:
    x = fnum(mean_value)
    s = fnum(std_value)
    if x is None:
        return "-"
    return f"{100.0 * x:.2f}±{100.0 * (s or 0.0):.2f}"


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def key(dataset: str, method: str) -> tuple[int, int]:
    return (
        DATASET_ORDER.index(dataset) if dataset in DATASET_ORDER else 99,
        METHOD_ORDER.index(method) if method in METHOD_ORDER else 99,
    )


def setting_param(row: dict[str, str]) -> str:
    k = str(row.get("k") or "").strip()
    r = str(row.get("r") or "").strip()
    parts: list[str] = []
    if k:
        parts.append(f"k={k}")
    if r:
        parts.append(f"r={r}")
    return ",".join(parts) or "-"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base_csv",
        default="reports/tdb_final_full_sweep_f1_mia_by_setting_seed424344_20260526.csv",
    )
    ap.add_argument(
        "--as_ls_csv",
        default="reports/tdb_as_ls_k1to9_r0p1to1_seed424344_20260527_best.csv",
    )
    ap.add_argument(
        "--dsu_csv",
        default="reports/tdb_dsu_joint_k1to9_r0p1to1_seed424344_20260530_final_local_best.csv",
    )
    ap.add_argument("--out_csv", default="reports/tdb_clean_final_local_f1_mia_20260602.csv")
    ap.add_argument("--out_md", default="reports/tdb_clean_final_local_f1_mia_20260602.md")
    args = ap.parse_args()

    base_rows = read_rows(ROOT / args.base_csv)
    as_ls_rows = read_rows(ROOT / args.as_ls_csv)
    dsu_rows = read_rows(ROOT / args.dsu_csv)

    combined: dict[tuple[str, str], dict[str, Any]] = {}
    for row in base_rows:
        if (row.get("setting") or "").lower() != "base":
            continue
        k0 = (row["dataset"], row["method"])
        combined.setdefault(k0, {"dataset": row["dataset"], "method": row["method"]})
        combined[k0]["base_param"] = "-"
        combined[k0]["base_f1"] = row.get("macro_f1_mean")
        combined[k0]["base_f1_std"] = row.get("macro_f1_std")
        combined[k0]["base_mia"] = row.get("mia_auc_mean")
        combined[k0]["base_mia_std"] = row.get("mia_auc_std")

    for row in as_ls_rows:
        setting = (row.get("setting") or "").lower()
        if setting not in {"as", "ls"}:
            continue
        k0 = (row["dataset"], row["algorithm"])
        combined.setdefault(k0, {"dataset": row["dataset"], "method": row["algorithm"]})
        combined[k0][f"{setting}_param"] = setting_param(row)
        combined[k0][f"{setting}_f1"] = row.get("macro_f1_best_mean")
        combined[k0][f"{setting}_f1_std"] = row.get("macro_f1_best_std")
        combined[k0][f"{setting}_mia"] = row.get("mia_auc_mean")
        combined[k0][f"{setting}_mia_std"] = row.get("mia_auc_std")

    for row in dsu_rows:
        k0 = (row["dataset"], row["algorithm"])
        combined.setdefault(k0, {"dataset": row["dataset"], "method": row["algorithm"]})
        combined[k0]["dsu_param"] = setting_param(row)
        combined[k0]["dsu_f1"] = row.get("macro_f1_best_mean")
        combined[k0]["dsu_f1_std"] = row.get("macro_f1_best_std")
        combined[k0]["dsu_mia"] = row.get("mia_auc_mean")
        combined[k0]["dsu_mia_std"] = row.get("mia_auc_std")

    rows = [combined[k] for k in sorted(combined, key=lambda x: key(*x))]

    out_csv = ROOT / args.out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset",
        "method",
        "base_param",
        "base_f1",
        "base_f1_std",
        "base_mia",
        "base_mia_std",
        "as_param",
        "as_f1",
        "as_f1_std",
        "as_mia",
        "as_mia_std",
        "ls_param",
        "ls_f1",
        "ls_f1_std",
        "ls_mia",
        "ls_mia_std",
        "dsu_param",
        "dsu_f1",
        "dsu_f1_std",
        "dsu_mia",
        "dsu_mia_std",
        "dsu_minus_base_f1",
        "dsu_minus_as_f1",
        "dsu_minus_ls_f1",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            dsu = fnum(row.get("dsu_f1"))
            for setting in ["base", "as", "ls"]:
                other = fnum(row.get(f"{setting}_f1"))
                out[f"dsu_minus_{setting}_f1"] = None if dsu is None or other is None else dsu - other
            writer.writerow({field: out.get(field) for field in fields})

    lines: list[str] = []
    lines.append("# Final Local-Ring TDB Clean F1/MIA Table")
    lines.append("")
    lines.append("本表只汇总不带后门投毒的 clean 主实验。DSU 使用最终 `tdb_aggregation_scope=local` 的 joint k×r 搜索结果；不使用 global 聚合诊断结果。")
    lines.append("")
    lines.append(f"- CSV: `{out_csv.relative_to(ROOT)}`")
    lines.append("")
    lines.append("| Dataset | Method | Base F1/MIA | AS best F1/MIA | LS best F1/MIA | DSU joint F1/MIA | DSU-Base | DSU-AS | DSU-LS |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        dsu = fnum(row.get("dsu_f1"))
        base = fnum(row.get("base_f1"))
        as_f1 = fnum(row.get("as_f1"))
        ls_f1 = fnum(row.get("ls_f1"))

        def delta(other: float | None) -> str:
            return "-" if dsu is None or other is None else f"{100.0 * (dsu - other):.2f}"

        lines.append(
            "| "
            f"{DATASET_LABEL.get(row['dataset'], row['dataset'])} | "
            f"{METHOD_LABEL.get(row['method'], row['method'])} | "
            f"{pct_std(row.get('base_f1'), row.get('base_f1_std'))}/{pct_std(row.get('base_mia'), row.get('base_mia_std'))} | "
            f"{row.get('as_param', '-')} {pct_std(row.get('as_f1'), row.get('as_f1_std'))}/{pct_std(row.get('as_mia'), row.get('as_mia_std'))} | "
            f"{row.get('ls_param', '-')} {pct_std(row.get('ls_f1'), row.get('ls_f1_std'))}/{pct_std(row.get('ls_mia'), row.get('ls_mia_std'))} | "
            f"{row.get('dsu_param', '-')} {pct_std(row.get('dsu_f1'), row.get('dsu_f1_std'))}/{pct_std(row.get('dsu_mia'), row.get('dsu_mia_std'))} | "
            f"{delta(base)} | {delta(as_f1)} | {delta(ls_f1)} |"
        )

    out_md = ROOT / args.out_md
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_md)
    print(out_csv)


if __name__ == "__main__":
    main()
