#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple


ROOT = Path(__file__).resolve().parent.parent
DATASET_ORDER = ["20newsgroups", "yahoo_subset"]
ALGO_ORDER = ["d-federaser", "d-fedosd", "d-fedrecovery", "d-oblivionis"]
STRATEGY_ORDER = ["full_all", "ours_all", "full_ours", "ours_ours"]
STRATEGY_LABELS = {
    "full_all": "Base",
    "ours_all": "AS",
    "full_ours": "LS",
    "ours_ours": "DSU",
}


def _to_float(x: object) -> Optional[float]:
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None


def _fmt_pct(x: Optional[float]) -> str:
    if x is None:
        return "-"
    return f"{x * 100:.2f}"


def _rate_label(rate: str) -> str:
    return rate.replace("p", ".") if rate else "unknown"


def _status(base: Dict[str, str], dsu: Dict[str, str], eps: float = 1e-9) -> str:
    base_dfl_asr = _to_float(base.get("dfl_asr_mean"))
    base_dfu_asr = _to_float(base.get("dfu_asr_mean"))
    dsu_dfl_asr = _to_float(dsu.get("dfl_asr_mean"))
    dsu_dfu_asr = _to_float(dsu.get("dfu_asr_mean"))
    if dsu_dfl_asr is None or dsu_dfu_asr is None:
        return "N/A"
    if dsu_dfu_asr > dsu_dfl_asr + eps:
        return "FAIL(DSU raises ASR)"
    if base_dfu_asr is None or base_dfl_asr is None:
        return "PASS"
    same_baseline = abs(base_dfl_asr - dsu_dfl_asr) <= eps
    if dsu_dfu_asr > base_dfu_asr + eps:
        return "FAIL(DSU>Base)" if same_baseline else "WARN(DSU>Base, baseline differs)"
    return "PASS"


def _cell(row: Optional[Dict[str, str]]) -> str:
    if not row:
        return "-"
    return f"{_fmt_pct(_to_float(row.get('dfl_asr_mean')))}->{_fmt_pct(_to_float(row.get('dfu_asr_mean')))} / F1 {_fmt_pct(_to_float(row.get('dfu_clean_f1_mean')))}"


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def build_md(rows: List[Dict[str, str]]) -> str:
    by_key: Dict[Tuple[str, str, str, str], Dict[str, str]] = {}
    for row in rows:
        by_key[(row["bd_rate"], row["dataset"], row["algorithm"], row["strategy"])] = row

    rates = sorted({row["bd_rate"] for row in rows}, key=lambda x: (_rate_label(x), x))

    lines: List[str] = []
    lines.append("# Backdoor Strategy Matrix")
    lines.append("")
    lines.append("- Cell format: `DFL_ASR -> DFU_ASR / DFU clean F1` in percent.")
    lines.append("- `FAIL(DSU raises ASR)` means DSU DFU ASR is higher than its own DFL baseline.")
    lines.append("- `FAIL(DSU>Base)` means Base and DSU share the same DFL baseline and DSU DFU ASR is higher than Base DFU ASR.")
    lines.append("- `WARN(DSU>Base, baseline differs)` means DSU is higher than Base, but Base and DSU were audited on different DFL baselines.")
    lines.append("")

    for rate in rates:
        lines.append(f"## rate={_rate_label(rate)}")
        lines.append("")
        lines.append("| Dataset | Method | n | Base | AS | LS | DSU | Status |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---|")
        for dataset in DATASET_ORDER:
            for algo in ALGO_ORDER:
                base = by_key.get((rate, dataset, algo, "full_all"))
                as_row = by_key.get((rate, dataset, algo, "ours_all"))
                ls_row = by_key.get((rate, dataset, algo, "full_ours"))
                dsu = by_key.get((rate, dataset, algo, "ours_ours"))
                n = int(dsu.get("n") if dsu and dsu.get("n") else (base.get("n") if base and base.get("n") else 0))
                status = _status(base, dsu) if base and dsu else "N/A"
                lines.append(
                    f"| {dataset} | {algo} | {n} | {_cell(base)} | {_cell(as_row)} | {_cell(ls_row)} | {_cell(dsu)} | {status} |"
                )
        lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in_csv",
        default="reports/backdoor_effective_audit_tdbfull_single_per_rate_nontarget_20260530.csv",
    )
    ap.add_argument(
        "--out_md",
        default="reports/backdoor_strategy_matrix_tdbfull_single_per_rate_nontarget_20260530.md",
    )
    args = ap.parse_args()

    in_csv = Path(args.in_csv)
    if not in_csv.is_absolute():
        in_csv = ROOT / in_csv
    out_md = Path(args.out_md)
    if not out_md.is_absolute():
        out_md = ROOT / out_md

    rows = load_rows(in_csv)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(build_md(rows), encoding="utf-8")
    print(f"Wrote: {out_md}")


if __name__ == "__main__":
    main()
