#!/usr/bin/env python3
"""Audit whether clean LLM DSU/AS/LS results are F1-effective.

Input is the JSON produced by scripts/extract_ablation_from_sweeps.py.  The
report is intentionally F1-focused; MIA protocol drift is documented elsewhere.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


ROOT = Path(__file__).resolve().parent.parent

DATASETS = ["20newsgroups", "yahoo_subset"]
METHODS = ["d-federaser", "d-fedosd", "d-fedrecovery", "d-oblivionis"]
METHOD_LABEL = {
    "d-federaser": "FedEraser",
    "d-fedosd": "FedOSD",
    "d-fedrecovery": "FedRecovery",
    "d-oblivionis": "Oblivionis",
}

PAPER_F1 = {
    "20newsgroups": {
        "d-federaser": {"base": 40.03, "dsu": 44.94},
        "d-fedosd": {"base": 47.98, "dsu": 58.48},
        "d-fedrecovery": {"base": 46.18, "dsu": 61.23},
        "d-oblivionis": {"base": 46.37, "dsu": 62.01},
    },
    "yahoo_subset": {
        "d-federaser": {"base": 67.49, "dsu": 69.33},
        "d-fedosd": {"base": 60.97, "dsu": 75.34},
        "d-fedrecovery": {"base": 70.82, "dsu": 74.28},
        "d-oblivionis": {"base": 67.82, "dsu": 73.64},
    },
}


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(x: Optional[float]) -> str:
    return "-" if x is None else f"{float(x):.2f}"


def _delta(x: Optional[float]) -> str:
    return "-" if x is None else f"{float(x):+.2f}"


def _combo(data: Dict[str, Any], dataset: str, method: str, combo: str) -> Dict[str, Any]:
    return ((data.get("results") or {}).get(dataset) or {}).get(method, {}).get(combo, {}) or {}


def _mean(data: Dict[str, Any], dataset: str, method: str, combo: str) -> Optional[float]:
    v = _combo(data, dataset, method, combo).get("mean")
    return None if v is None else float(v)


def _std(data: Dict[str, Any], dataset: str, method: str, combo: str) -> Optional[float]:
    v = _combo(data, dataset, method, combo).get("std")
    return None if v is None else float(v)


def _values(data: Dict[str, Any], dataset: str, method: str, combo: str) -> Dict[str, float]:
    values = _combo(data, dataset, method, combo).get("values") or {}
    out: Dict[str, float] = {}
    if isinstance(values, dict):
        for k, v in values.items():
            try:
                out[str(k)] = float(v)
            except (TypeError, ValueError):
                pass
    return out


def _chosen(data: Dict[str, Any], dataset: str, method: str, combo: str) -> str:
    chosen = _combo(data, dataset, method, combo).get("chosen") or {}
    if not isinstance(chosen, dict) or not chosen:
        return ""
    parts = []
    if "agent_count" in chosen:
        try:
            parts.append(f"k={int(round(float(chosen['agent_count'])))}")
        except Exception:
            parts.append(f"k={chosen['agent_count']}")
    if "lora_ratio" in chosen:
        try:
            parts.append(f"r={float(chosen['lora_ratio']):g}")
        except Exception:
            parts.append(f"r={chosen['lora_ratio']}")
    return ",".join(parts)


def _mean_std_text(m: Optional[float], s: Optional[float]) -> str:
    if m is None:
        return "-"
    if s is None:
        return f"{m:.2f}"
    return f"{m:.2f}±{s:.2f}"


def _status(gain: Optional[float], *, min_gain: float, no_regress_tol: float) -> str:
    if gain is None:
        return "missing"
    if gain >= min_gain:
        return "improve"
    if gain >= -abs(no_regress_tol):
        return "no_regress"
    return "regress"


def _seed_gain_flags(
    base_values: Dict[str, float],
    combo_values: Dict[str, float],
    *,
    no_regress_tol: float,
) -> str:
    flags: List[str] = []
    for seed in sorted(set(base_values) & set(combo_values), key=lambda x: int(x)):
        gain = combo_values[seed] - base_values[seed]
        if gain < -abs(no_regress_tol):
            flags.append(f"seed{seed}:{gain:+.2f}")
    return ",".join(flags)


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset",
        "method",
        "base_f1",
        "as_f1",
        "ls_f1",
        "dsu_f1",
        "as_gain_vs_base",
        "ls_gain_vs_base",
        "dsu_gain_vs_base",
        "dsu_gain_vs_best_component",
        "paper_base_f1",
        "paper_dsu_f1",
        "paper_dsu_delta",
        "chosen_as",
        "chosen_ls",
        "chosen_dsu",
        "as_status",
        "ls_status",
        "dsu_status",
        "seed_regressions",
        "needs_rescue",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _build_rows(data: Dict[str, Any], *, min_gain: float, no_regress_tol: float) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for dataset in DATASETS:
        for method in METHODS:
            base = _mean(data, dataset, method, "base")
            as_v = _mean(data, dataset, method, "as")
            ls_v = _mean(data, dataset, method, "ls")
            dsu = _mean(data, dataset, method, "both")
            best_component = max([v for v in [base, as_v, ls_v] if v is not None], default=None)
            as_gain = None if base is None or as_v is None else as_v - base
            ls_gain = None if base is None or ls_v is None else ls_v - base
            dsu_gain = None if base is None or dsu is None else dsu - base
            dsu_best_gain = None if best_component is None or dsu is None else dsu - best_component

            as_status = _status(as_gain, min_gain=min_gain, no_regress_tol=no_regress_tol)
            ls_status = _status(ls_gain, min_gain=min_gain, no_regress_tol=no_regress_tol)
            dsu_status = _status(dsu_gain, min_gain=min_gain, no_regress_tol=no_regress_tol)

            base_values = _values(data, dataset, method, "base")
            regressions = []
            for combo in ["as", "ls", "both"]:
                flags = _seed_gain_flags(
                    base_values,
                    _values(data, dataset, method, combo),
                    no_regress_tol=no_regress_tol,
                )
                if flags:
                    regressions.append(f"{combo}({flags})")

            paper = PAPER_F1.get(dataset, {}).get(method, {})
            paper_dsu = paper.get("dsu")
            paper_delta = None if dsu is None or paper_dsu is None else dsu - float(paper_dsu)

            needs_rescue = (
                dsu_status == "regress"
                or as_status == "regress"
                or ls_status == "regress"
                or (dsu_best_gain is not None and dsu_best_gain < -abs(no_regress_tol))
                or dsu is None
            )

            rows.append(
                {
                    "dataset": dataset,
                    "method": METHOD_LABEL[method],
                    "base_f1": _mean_std_text(base, _std(data, dataset, method, "base")),
                    "as_f1": _mean_std_text(as_v, _std(data, dataset, method, "as")),
                    "ls_f1": _mean_std_text(ls_v, _std(data, dataset, method, "ls")),
                    "dsu_f1": _mean_std_text(dsu, _std(data, dataset, method, "both")),
                    "as_gain_vs_base": _delta(as_gain),
                    "ls_gain_vs_base": _delta(ls_gain),
                    "dsu_gain_vs_base": _delta(dsu_gain),
                    "dsu_gain_vs_best_component": _delta(dsu_best_gain),
                    "paper_base_f1": _fmt(paper.get("base")),
                    "paper_dsu_f1": _fmt(paper_dsu),
                    "paper_dsu_delta": _delta(paper_delta),
                    "chosen_as": _chosen(data, dataset, method, "as"),
                    "chosen_ls": _chosen(data, dataset, method, "ls"),
                    "chosen_dsu": _chosen(data, dataset, method, "both"),
                    "as_status": as_status,
                    "ls_status": ls_status,
                    "dsu_status": dsu_status,
                    "seed_regressions": "; ".join(regressions),
                    "needs_rescue": "yes" if needs_rescue else "no",
                }
            )
    return rows


def _markdown(rows: List[Dict[str, Any]], *, in_json: Path, min_gain: float, no_regress_tol: float) -> str:
    lines: List[str] = []
    lines.append("# LLM Effective F1 Audit")
    lines.append("")
    lines.append(f"- Source: `{in_json}`")
    lines.append("- Scope: clean LLM F1 only; MIA drift is intentionally not used for pass/fail here.")
    lines.append(f"- `improve`: gain >= `{min_gain:.2f}` F1 points.")
    lines.append(f"- `no_regress`: gain is within `{no_regress_tol:.2f}` F1 points below Base.")
    lines.append("- `needs_rescue=yes`: AS/LS/DSU regressed, DSU is below the best component, or AS+LS is missing.")
    lines.append("")
    lines.append(
        "| Dataset | Method | Base | AS | LS | AS+LS | AS gain | LS gain | DSU gain | DSU vs best component | "
        "Paper DSU | Δ vs paper DSU | chosen AS | chosen LS | chosen DSU | statuses | rescue |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|:---:|")
    for row in rows:
        statuses = f"AS={row['as_status']}; LS={row['ls_status']}; DSU={row['dsu_status']}"
        lines.append(
            f"| {row['dataset']} | {row['method']} | {row['base_f1']} | {row['as_f1']} | {row['ls_f1']} | "
            f"{row['dsu_f1']} | {row['as_gain_vs_base']} | {row['ls_gain_vs_base']} | "
            f"{row['dsu_gain_vs_base']} | {row['dsu_gain_vs_best_component']} | {row['paper_dsu_f1']} | "
            f"{row['paper_dsu_delta']} | {row['chosen_as']} | {row['chosen_ls']} | {row['chosen_dsu']} | "
            f"{statuses} | {row['needs_rescue']} |"
        )
    lines.append("")
    bad = [r for r in rows if r["needs_rescue"] == "yes"]
    if bad:
        lines.append("## Rescue Queue")
        lines.append("")
        for row in bad:
            lines.append(
                f"- {row['dataset']} / {row['method']}: {row['seed_regressions'] or 'aggregate regression/missing result'}"
            )
        lines.append("")
    else:
        lines.append("No F1 rescue cells were detected under the configured thresholds.")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation_json", type=str, required=True)
    ap.add_argument("--out_md", type=str, required=True)
    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--min_gain", type=float, default=0.5)
    ap.add_argument(
        "--no_regress_tol",
        type=float,
        default=1.0,
        help="Tolerance in F1 points for treating small component drops as no-regression.",
    )
    args = ap.parse_args()

    in_json = ROOT / args.ablation_json
    data = _load_json(in_json)
    rows = _build_rows(data, min_gain=float(args.min_gain), no_regress_tol=float(args.no_regress_tol))

    out_csv = ROOT / args.out_csv
    _write_csv(out_csv, rows)
    print(f"[OK] wrote {out_csv}")

    out_md = ROOT / args.out_md
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(
        _markdown(rows, in_json=in_json, min_gain=float(args.min_gain), no_regress_tol=float(args.no_regress_tol)),
        encoding="utf-8",
    )
    print(f"[OK] wrote {out_md}")


if __name__ == "__main__":
    main()
