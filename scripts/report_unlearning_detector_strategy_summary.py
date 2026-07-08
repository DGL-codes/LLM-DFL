#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


ROOT = Path(__file__).resolve().parent.parent


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _to_float(text: str) -> float | None:
    text = str(text or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except Exception:
        return None


def _parse_rate(strategy_display: str) -> str:
    if "_bd0p1_" in strategy_display:
        return "0.1"
    if "_bd0p2_" in strategy_display:
        return "0.2"
    return "unknown"


def _parse_setting(strategy_display: str) -> str:
    if strategy_display.startswith("full_all"):
        return "Base"
    if strategy_display.startswith("ours_all"):
        return "AS"
    if strategy_display.startswith("full_ours"):
        return "LS"
    if strategy_display.startswith("ours_ours"):
        return "DSU"
    return strategy_display


def _pct(x: float | None) -> str:
    if x is None:
        return "-"
    return f"{100.0 * x:.2f}"


def _fmt_triplet(row: Dict[str, str] | None) -> str:
    if not row:
        return "-"
    asr = _pct(_to_float(row.get("dfu_asr_mean", "")))
    f1 = _pct(_to_float(row.get("dfu_clean_f1_mean", "")))
    mia = _pct(_to_float(row.get("mia_dfu_adv_mean", "")))
    return f"ASR {asr} / F1 {f1} / MIA {mia}"


def _status(base_row: Dict[str, str] | None, dsu_row: Dict[str, str] | None) -> str:
    if not base_row or not dsu_row:
        return "INCOMPLETE"
    base_asr = _to_float(base_row.get("dfu_asr_mean", ""))
    dsu_asr = _to_float(dsu_row.get("dfu_asr_mean", ""))
    base_f1 = _to_float(base_row.get("dfu_clean_f1_mean", ""))
    dsu_f1 = _to_float(dsu_row.get("dfu_clean_f1_mean", ""))
    if None in (base_asr, dsu_asr, base_f1, dsu_f1):
        return "INCOMPLETE"
    if dsu_asr <= base_asr and dsu_f1 >= base_f1:
        return "DSU better/equal vs Base"
    if dsu_asr <= base_asr and dsu_f1 < base_f1:
        return "Lower ASR, lower F1"
    if dsu_asr > base_asr and dsu_f1 >= base_f1:
        return "Higher ASR, higher F1"
    return "Worse ASR and F1"


def build_report(rows: List[Dict[str, str]]) -> str:
    by_cell: Dict[Tuple[str, str, str], Dict[str, Dict[str, str]]] = defaultdict(dict)
    for row in rows:
        dataset = str(row["dataset"])
        method = str(row["algorithm"])
        strategy_display = str(row["strategy_display"])
        rate = _parse_rate(strategy_display)
        setting = _parse_setting(strategy_display)
        by_cell[(rate, dataset, method)][setting] = row

    lines: List[str] = []
    lines.append("# Unlearning Detector Strategy Summary")
    lines.append("")
    lines.append("- Cell format: `ASR / F1 / MIA adv`, all in percent.")
    lines.append("- `Base=full_all`, `AS=ours_all`, `LS=full_ours`, `DSU=ours_ours`.")
    lines.append("- ASR is the primary forgetting signal; MIA is secondary evidence.")
    lines.append("")
    rates = sorted({key[0] for key in by_cell.keys()}, key=lambda x: (x == "unknown", x))
    for rate in rates:
        lines.append(f"## rate={rate}")
        lines.append("")
        lines.append("| Dataset | Method | Base | AS | LS | DSU | Status |")
        lines.append("|---|---|---|---|---|---|---|")
        for dataset in ["20newsgroups", "yahoo_subset"]:
            methods = sorted(method for r, d, method in by_cell.keys() if r == rate and d == dataset)
            for method in methods:
                cell = by_cell[(rate, dataset, method)]
                base_row = cell.get("Base")
                dsu_row = cell.get("DSU")
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            dataset,
                            method,
                            _fmt_triplet(cell.get("Base")),
                            _fmt_triplet(cell.get("AS")),
                            _fmt_triplet(cell.get("LS")),
                            _fmt_triplet(cell.get("DSU")),
                            _status(base_row, dsu_row),
                        ]
                    )
                    + " |"
                )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in_csv",
        default="reports/unlearning_detector_validation_tdbfull_single_allrates_20260530.csv",
    )
    ap.add_argument(
        "--out_md",
        default="reports/unlearning_detector_strategy_summary_tdbfull_single_allrates_20260530.md",
    )
    args = ap.parse_args()

    rows = _read_rows(ROOT / args.in_csv)
    out_path = ROOT / args.out_md
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_report(rows), encoding="utf-8")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
