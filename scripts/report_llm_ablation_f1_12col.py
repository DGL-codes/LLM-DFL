#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parent.parent

DATASETS = ["20newsgroups", "yahoo_subset"]
METHODS = ["d-federaser", "d-fedosd", "d-fedrecovery", "d-oblivionis"]
METHOD_LABEL = {
    "d-federaser": "FedEraser",
    "d-fedosd": "FedOSD",
    "d-fedrecovery": "FedRecovery",
    "d-oblivionis": "Oblivionis",
}

COMBOS = [
    ("base", "base"),
    ("as", "as"),
    ("ls", "ls"),
    ("both", "dsu"),
]

# Paper-side F1 references used for the 12-column ablation table.
# Provenance:
# - base / as / ls: PDF page 13-14 sweep text (full participation, best-k, best-r)
# - dsu: PDF Table I / II and page 12 ablation text
PAPER_F1: Dict[str, Dict[str, Dict[str, float]]] = {
    "20newsgroups": {
        "d-federaser": {"base": 43.37, "as": 45.29, "ls": 43.87, "both": 44.94},
        "d-fedosd": {"base": 48.55, "as": 46.24, "ls": 56.45, "both": 58.48},
        "d-fedrecovery": {"base": 47.81, "as": 49.38, "ls": 61.36, "both": 61.23},
        "d-oblivionis": {"base": 45.31, "as": 48.27, "ls": 58.72, "both": 62.01},
    },
    "yahoo_subset": {
        "d-federaser": {"base": 69.03, "as": 69.39, "ls": 67.11, "both": 69.33},
        "d-fedosd": {"base": 64.98, "as": 68.33, "ls": 73.97, "both": 75.34},
        "d-fedrecovery": {"base": 72.14, "as": 71.75, "ls": 74.89, "both": 74.28},
        "d-oblivionis": {"base": 69.93, "as": 70.37, "ls": 74.09, "both": 73.64},
    },
}


def _read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_num(x: float) -> str:
    return f"{x:.2f}"


def _fmt_delta(x: float) -> str:
    return f"{x:+.2f}"


def _write_csv(path: Path, header: List[str], rows: List[List[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def _local_value(data: Dict, dataset: str, method: str, combo: str) -> float:
    return float(data["results"][dataset][method][combo]["mean"])


def build_dataset_rows(data: Dict, dataset: str) -> tuple[List[str], List[List[str]]]:
    header = ["method"]
    for combo_key, combo_label in COMBOS:
        header.extend(
            [
                f"{combo_label}_paper_f1",
                f"{combo_label}_repro_f1",
                f"{combo_label}_delta_f1",
            ]
        )

    rows: List[List[str]] = []
    for method in METHODS:
        row = [METHOD_LABEL[method]]
        for combo_key, _combo_label in COMBOS:
            paper_v = float(PAPER_F1[dataset][method][combo_key])
            local_v = _local_value(data, dataset, method, combo_key)
            delta = local_v - paper_v
            row.extend([_fmt_num(paper_v), _fmt_num(local_v), _fmt_delta(delta)])
        rows.append(row)
    return header, rows


def build_markdown(data: Dict) -> str:
    lines: List[str] = []
    lines.append("# LLM Ablation F1 12-Column Report")
    lines.append("")
    lines.append("- Repro source: `figures/ablation_bestcfg_asls_4seeds.json`")
    lines.append("- Repro seeds: `42,43,44,45`")
    lines.append("- Paper source:")
    lines.append("  - `base/as/ls`: PDF page 13-14 sweep text")
    lines.append("  - `dsu`: PDF Table I / II and page 12 ablation text")
    lines.append("- Note: the paper text uses different sections as sources; base/as/ls and dsu are not all drawn from the same table.")
    lines.append("")

    for dataset in DATASETS:
        header, rows = build_dataset_rows(data, dataset)
        lines.append(f"## {dataset}")
        lines.append("")
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")
        for row in rows:
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Build 12-column F1 ablation comparison tables.")
    ap.add_argument(
        "--repro_json",
        type=str,
        default="figures/ablation_bestcfg_asls_4seeds.json",
        help="Local ablation repro JSON with base/as/ls/both mean values.",
    )
    ap.add_argument(
        "--out_md",
        type=str,
        default="reports/llm_ablation_f1_12col.md",
    )
    ap.add_argument(
        "--out_csv_prefix",
        type=str,
        default="reports/llm_ablation_f1_12col",
        help="Per-dataset CSV prefix, outputs *_20newsgroups.csv and *_yahoo_subset.csv.",
    )
    args = ap.parse_args()

    repro_json = ROOT / args.repro_json
    data = _read_json(repro_json)

    for dataset in DATASETS:
        header, rows = build_dataset_rows(data, dataset)
        out_csv = ROOT / f"{args.out_csv_prefix}_{dataset}.csv"
        _write_csv(out_csv, header, rows)
        print(f"[OK] wrote {out_csv}")

    out_md = ROOT / args.out_md
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(build_markdown(data), encoding="utf-8")
    print(f"[OK] wrote {out_md}")


if __name__ == "__main__":
    main()
