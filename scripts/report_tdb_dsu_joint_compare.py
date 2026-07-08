#!/usr/bin/env python3
"""Compare expanded AS/LS best rows with DSU joint-grid best rows."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _float(row: Optional[dict[str, Any]], key: str) -> Optional[float]:
    if not row:
        return None
    value = row.get(key)
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _fmt(value: Optional[float], digits: int = 4) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _fmt_pm(mean: Optional[float], std: Optional[float]) -> str:
    if mean is None:
        return "-"
    if std is None:
        return f"{mean:.4f}"
    return f"{mean:.4f}+/-{std:.4f}"


def _base_map(rows: Iterable[dict[str, str]]) -> dict[Tuple[str, str], dict[str, str]]:
    out: dict[Tuple[str, str], dict[str, str]] = {}
    for row in rows:
        dataset = row.get("dataset")
        method = row.get("method") or row.get("algorithm")
        if dataset and method:
            out[(dataset, method)] = row
    return out


def _best_map(rows: Iterable[dict[str, str]], setting: str) -> dict[Tuple[str, str], dict[str, str]]:
    out: dict[Tuple[str, str], dict[str, str]] = {}
    for row in rows:
        if str(row.get("setting")) != setting:
            continue
        dataset = row.get("dataset")
        method = row.get("algorithm") or row.get("method")
        if dataset and method:
            out[(dataset, method)] = row
    return out


def _status(
    base: Optional[float],
    as_f1: Optional[float],
    ls_f1: Optional[float],
    dsu_f1: Optional[float],
    dsu_n: Optional[int],
    expected_n: int,
) -> str:
    if dsu_f1 is None:
        return "MISSING_DSU"
    if dsu_n is None or dsu_n < expected_n:
        return "PARTIAL"
    if base is not None and dsu_f1 <= base:
        return "FAIL_DSU_LE_BASE"
    if as_f1 is not None and dsu_f1 <= as_f1:
        return "FAIL_DSU_LE_AS"
    if ls_f1 is not None and dsu_f1 <= ls_f1:
        return "FAIL_DSU_LE_LS"
    return "PASS"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base_csv", required=True)
    ap.add_argument("--as_ls_best_csv", required=True)
    ap.add_argument("--dsu_best_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_md", required=True)
    ap.add_argument("--expected_seeds", type=int, default=3)
    args = ap.parse_args()

    base_rows = _base_map(_read_rows(Path(args.base_csv)))
    as_rows = _best_map(_read_rows(Path(args.as_ls_best_csv)), "AS")
    ls_rows = _best_map(_read_rows(Path(args.as_ls_best_csv)), "LS")
    dsu_rows = _best_map(_read_rows(Path(args.dsu_best_csv)), "DSU")

    keys = sorted(set(base_rows) | set(as_rows) | set(ls_rows) | set(dsu_rows))
    out_rows: list[dict[str, Any]] = []
    for dataset, method in keys:
        base = base_rows.get((dataset, method))
        as_row = as_rows.get((dataset, method))
        ls_row = ls_rows.get((dataset, method))
        dsu_row = dsu_rows.get((dataset, method))

        base_f1 = _float(base, "base_f1") or _float(base, "macro_f1_best_mean")
        as_f1 = _float(as_row, "macro_f1_best_mean")
        ls_f1 = _float(ls_row, "macro_f1_best_mean")
        dsu_f1 = _float(dsu_row, "macro_f1_best_mean")
        dsu_n = None
        if dsu_row and dsu_row.get("n"):
            try:
                dsu_n = int(float(dsu_row["n"]))
            except Exception:
                dsu_n = None

        out_rows.append(
            {
                "dataset": dataset,
                "method": method,
                "base_f1": base_f1,
                "as_k": as_row.get("k") if as_row else "",
                "as_f1": as_f1,
                "as_std": _float(as_row, "macro_f1_best_std"),
                "ls_r": ls_row.get("r") if ls_row else "",
                "ls_f1": ls_f1,
                "ls_std": _float(ls_row, "macro_f1_best_std"),
                "dsu_k": dsu_row.get("k") if dsu_row else "",
                "dsu_r": dsu_row.get("r") if dsu_row else "",
                "dsu_f1": dsu_f1,
                "dsu_std": _float(dsu_row, "macro_f1_best_std"),
                "dsu_n": dsu_n,
                "as_minus_base": None if as_f1 is None or base_f1 is None else as_f1 - base_f1,
                "ls_minus_base": None if ls_f1 is None or base_f1 is None else ls_f1 - base_f1,
                "dsu_minus_base": None if dsu_f1 is None or base_f1 is None else dsu_f1 - base_f1,
                "dsu_minus_as": None if dsu_f1 is None or as_f1 is None else dsu_f1 - as_f1,
                "dsu_minus_ls": None if dsu_f1 is None or ls_f1 is None else dsu_f1 - ls_f1,
                "status": _status(base_f1, as_f1, ls_f1, dsu_f1, dsu_n, args.expected_seeds),
            }
        )

    fields = [
        "dataset",
        "method",
        "base_f1",
        "as_k",
        "as_f1",
        "as_std",
        "ls_r",
        "ls_f1",
        "ls_std",
        "dsu_k",
        "dsu_r",
        "dsu_f1",
        "dsu_std",
        "dsu_n",
        "as_minus_base",
        "ls_minus_base",
        "dsu_minus_base",
        "dsu_minus_as",
        "dsu_minus_ls",
        "status",
    ]
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in out_rows:
            writer.writerow({key: row.get(key) for key in fields})

    pass_count = sum(1 for row in out_rows if row["status"] == "PASS")
    dsu_base_count = sum(
        1
        for row in out_rows
        if row["dsu_minus_base"] is not None and float(row["dsu_minus_base"]) > 0
    )
    lines = [
        "# TDB DSU Joint Grid Comparison",
        "",
        f"- Cells: {len(out_rows)}",
        f"- DSU > Base: {dsu_base_count}/{len(out_rows)}",
        f"- DSU > AS and DSU > LS with complete seeds: {pass_count}/{len(out_rows)}",
        "",
        "| Dataset | Method | Base | AS best | LS best | DSU joint best | DSU-Base | DSU-AS | DSU-LS | Status |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in out_rows:
        as_text = f"k={row['as_k']} {_fmt_pm(row['as_f1'], row['as_std'])}"
        ls_text = f"r={row['ls_r']} {_fmt_pm(row['ls_f1'], row['ls_std'])}"
        dsu_text = f"k={row['dsu_k']},r={row['dsu_r']} {_fmt_pm(row['dsu_f1'], row['dsu_std'])} (n={row['dsu_n']})"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["dataset"]),
                    str(row["method"]),
                    _fmt(row["base_f1"]),
                    as_text,
                    ls_text,
                    dsu_text,
                    _fmt(row["dsu_minus_base"]),
                    _fmt(row["dsu_minus_as"]),
                    _fmt(row["dsu_minus_ls"]),
                    str(row["status"]),
                ]
            )
            + " |"
        )
    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_md}")


if __name__ == "__main__":
    main()
