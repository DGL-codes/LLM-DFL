#!/usr/bin/env python3
"""Build a compact DFL/Base/AS/LS/DSU backdoor forgetting table."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Optional


ROOT = Path(__file__).resolve().parent.parent
TAG_RE = re.compile(
    r"^bd_grid_(?P<dataset>20newsgroups|yahoo_subset)_seed(?P<seed>\d+)_"
    r"(?P<method>d-[a-z]+)_(?P<strategy>full_all|ours_all|full_ours|ours_ours)"
    r"(?P<suffix>.*)$"
)

STRATEGY_TO_COL = {
    "full_all": "base",
    "ours_all": "as",
    "full_ours": "ls",
    "ours_ours": "dsu",
}
METHOD_ORDER = ["d-federaser", "d-fedosd", "d-fedrecovery", "d-oblivionis"]
DATASET_ORDER = ["20newsgroups", "yahoo_subset"]
STRATEGY_ORDER = ["base", "as", "ls", "dsu"]


def rate_from_suffix(suffix: str) -> str:
    m = re.search(r"_bd(?P<rate>\d+p\d+)", suffix or "")
    return m.group("rate") if m else ""


def parse_roots(raw: str) -> list[Path]:
    roots: list[Path] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        p = Path(item)
        roots.append(p if p.is_absolute() else ROOT / p)
    return roots


def fget(data: dict[str, Any], *keys: str) -> Optional[float]:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    try:
        return float(cur)
    except Exception:
        return None


def fmt(x: Optional[float], digits: int = 2) -> str:
    if x is None:
        return "-"
    return f"{100.0 * x:.{digits}f}"


def fmt_delta(x: Optional[float], digits: int = 2) -> str:
    if x is None:
        return "-"
    return f"{100.0 * x:+.{digits}f}"


def mean_std(values: list[Optional[float]]) -> tuple[Optional[float], Optional[float]]:
    arr = [v for v in values if v is not None]
    if not arr:
        return None, None
    if len(arr) == 1:
        return arr[0], 0.0
    return mean(arr), pstdev(arr)


def fmt_pm(values: list[Optional[float]]) -> str:
    m, s = mean_std(values)
    if m is None:
        return "-"
    return f"{100.0*m:.2f}±{100.0*(s or 0.0):.2f}"


def load_rows(roots: list[Path], tag_contains: str, asr_family: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.glob("*/backdoor_audit.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            tag = str(data.get("tag") or path.parent.name)
            if tag_contains and tag_contains not in tag:
                continue
            if tag in seen:
                continue
            m = TAG_RE.match(tag)
            if not m:
                continue
            seen.add(tag)
            dfu = (data.get("models") or {}).get("dfu") or {}
            dfl = (data.get("models") or {}).get("dfl") or {}
            dfu_dir = Path(str(data.get("dfu_dir") or ""))
            cfg: dict[str, Any] = {}
            if dfu_dir.exists() and (dfu_dir / "dfu_config.json").exists():
                try:
                    cfg = json.loads((dfu_dir / "dfu_config.json").read_text(encoding="utf-8"))
                except Exception:
                    cfg = {}
            rows.append(
                {
                    "dataset": m.group("dataset"),
                    "seed": int(m.group("seed")),
                    "method": m.group("method"),
                    "rate": rate_from_suffix(m.group("suffix")),
                    "strategy": STRATEGY_TO_COL[m.group("strategy")],
                    "tag": tag,
                    "path": str(path),
                    "eval_scope": data.get("eval_scope"),
                    "dfu_state_mode": data.get("dfu_state_mode"),
                    "dfl_f1": fget(dfl, "clean", "macro_f1"),
                    "dfl_asr": fget(dfl, asr_family, "asr"),
                    "dfu_f1": fget(dfu, "clean", "macro_f1"),
                    "dfu_asr": fget(dfu, asr_family, "asr"),
                    "dfu_clean_target_rate": fget(dfu, "clean_target_rate_non_target", "asr"),
                    "dfu_asr_lift": None,
                    "dfu_n_agents": dfu.get("n_agents"),
                    "selected_agents": " ".join(str(x) for x in (cfg.get("selected_agents") or [])),
                    "tdb_aggregation_scope": cfg.get("tdb_aggregation_scope"),
                    "k": cfg.get("selection_count"),
                    "r": cfg.get("param_selection_ratio"),
                }
            )
    for row in rows:
        if row["dfu_asr"] is not None and row["dfu_clean_target_rate"] is not None:
            row["dfu_asr_lift"] = row["dfu_asr"] - row["dfu_clean_target_rate"]
    return rows


def build_cells(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (row["rate"], row["dataset"], row["method"], row["seed"])
        buckets[key][row["strategy"]] = row

    out: list[dict[str, Any]] = []
    for (rate, dataset, method, seed), by_strategy in buckets.items():
        base = by_strategy.get("base")
        dsu = by_strategy.get("dsu")
        ref = base or next(iter(by_strategy.values()))
        cell: dict[str, Any] = {
            "rate": rate,
            "dataset": dataset,
            "method": method,
            "seed": seed,
            "dfl_f1": ref.get("dfl_f1"),
            "dfl_asr": ref.get("dfl_asr"),
        }
        for strategy in STRATEGY_ORDER:
            row = by_strategy.get(strategy, {})
            cell[f"{strategy}_f1"] = row.get("dfu_f1")
            cell[f"{strategy}_asr"] = row.get("dfu_asr")
            cell[f"{strategy}_asr_lift"] = row.get("dfu_asr_lift")
            cell[f"{strategy}_n_agents"] = row.get("dfu_n_agents")
            cell[f"{strategy}_k"] = row.get("k")
            cell[f"{strategy}_r"] = row.get("r")
            cell[f"{strategy}_agg"] = row.get("tdb_aggregation_scope")
        cell["dsu_asr_drop_vs_dfl"] = None
        cell["dsu_asr_delta_vs_base"] = None
        if dsu and cell.get("dfl_asr") is not None and dsu.get("dfu_asr") is not None:
            cell["dsu_asr_drop_vs_dfl"] = cell["dfl_asr"] - dsu["dfu_asr"]
        if dsu and base and dsu.get("dfu_asr") is not None and base.get("dfu_asr") is not None:
            cell["dsu_asr_delta_vs_base"] = dsu["dfu_asr"] - base["dfu_asr"]
        out.append(cell)
    return sorted(
        out,
        key=lambda r: (
            r["rate"],
            DATASET_ORDER.index(r["dataset"]) if r["dataset"] in DATASET_ORDER else 99,
            METHOD_ORDER.index(r["method"]) if r["method"] in METHOD_ORDER else 99,
            r["seed"],
        ),
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "rate",
        "dataset",
        "method",
        "seed",
        "dfl_f1",
        "dfl_asr",
        "base_f1",
        "base_asr",
        "as_f1",
        "as_asr",
        "ls_f1",
        "ls_asr",
        "dsu_f1",
        "dsu_asr",
        "dsu_asr_drop_vs_dfl",
        "dsu_asr_delta_vs_base",
        "base_asr_lift",
        "as_asr_lift",
        "ls_asr_lift",
        "dsu_asr_lift",
        "base_n_agents",
        "as_n_agents",
        "ls_n_agents",
        "dsu_n_agents",
        "as_k",
        "ls_r",
        "dsu_k",
        "dsu_r",
        "as_agg",
        "dsu_agg",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in fields})


def write_md(path: Path, cells: list[dict[str, Any]], tag_contains: str, csv_path: str, asr_family: str) -> None:
    lines: list[str] = []
    lines.append("# Backdoor Direct Forgetting Table")
    lines.append("")
    lines.append(f"- Tag filter: `{tag_contains}`")
    lines.append(f"- ASR field: `{asr_family}`. Lower is better.")
    lines.append(f"- Values are percentages. `DSU-Base ASR` < 0 means DSU has lower ASR than Base.")
    lines.append(f"- CSV: `{csv_path}`")
    lines.append("")

    for rate in sorted({str(x["rate"]) for x in cells}):
        lines.append(f"## rate={rate.replace('p', '.')}")
        lines.append("")
        lines.append("| Dataset | Method | Seed | DFL F1/ASR | Base F1/ASR | AS F1/ASR | LS F1/ASR | DSU F1/ASR | DSU drop vs DFL | DSU-Base ASR |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in [x for x in cells if str(x["rate"]) == rate]:
            lines.append(
                "| "
                f"{row['dataset']} | {row['method']} | {row['seed']} | "
                f"{fmt(row.get('dfl_f1'))}/{fmt(row.get('dfl_asr'))} | "
                f"{fmt(row.get('base_f1'))}/{fmt(row.get('base_asr'))} | "
                f"{fmt(row.get('as_f1'))}/{fmt(row.get('as_asr'))} | "
                f"{fmt(row.get('ls_f1'))}/{fmt(row.get('ls_asr'))} | "
                f"{fmt(row.get('dsu_f1'))}/{fmt(row.get('dsu_asr'))} | "
                f"{fmt_delta(row.get('dsu_asr_drop_vs_dfl'))} | "
                f"{fmt_delta(row.get('dsu_asr_delta_vs_base'))} |"
            )
        lines.append("")

    lines.append("## Mean by Dataset and Method")
    lines.append("")
    lines.append("| Rate | Dataset | Method | DFL F1/ASR | Base F1/ASR | AS F1/ASR | LS F1/ASR | DSU F1/ASR | DSU drop vs DFL | DSU-Base ASR |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in cells:
        groups[(row["rate"], row["dataset"], row["method"])].append(row)
    for key in sorted(groups, key=lambda k: (k[0], DATASET_ORDER.index(k[1]), METHOD_ORDER.index(k[2]))):
        group = groups[key]
        rate, dataset, method = key
        lines.append(
            "| "
            f"{rate.replace('p', '.')} | {dataset} | {method} | "
            f"{fmt_pm([x.get('dfl_f1') for x in group])}/{fmt_pm([x.get('dfl_asr') for x in group])} | "
            f"{fmt_pm([x.get('base_f1') for x in group])}/{fmt_pm([x.get('base_asr') for x in group])} | "
            f"{fmt_pm([x.get('as_f1') for x in group])}/{fmt_pm([x.get('as_asr') for x in group])} | "
            f"{fmt_pm([x.get('ls_f1') for x in group])}/{fmt_pm([x.get('ls_asr') for x in group])} | "
            f"{fmt_pm([x.get('dsu_f1') for x in group])}/{fmt_pm([x.get('dsu_asr') for x in group])} | "
            f"{fmt_pm([x.get('dsu_asr_drop_vs_dfl') for x in group])} | "
            f"{fmt_pm([x.get('dsu_asr_delta_vs_base') for x in group])} |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bd_roots",
        default="实验结果/运行产物/artifacts/unlearning_audit/backdoor,artifacts/unlearning_audit/backdoor",
    )
    parser.add_argument("--tag_contains", required=True)
    parser.add_argument("--asr_family", default="asr_non_target", choices=["asr", "asr_non_target"])
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_md", required=True)
    args = parser.parse_args()

    rows = load_rows(parse_roots(args.bd_roots), args.tag_contains, args.asr_family)
    cells = build_cells(rows)
    out_csv = ROOT / args.out_csv
    out_md = ROOT / args.out_md
    write_csv(out_csv, cells)
    write_md(out_md, cells, args.tag_contains, args.out_csv, args.asr_family)
    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_md}")
    print(f"Rows: raw={len(rows)} cells={len(cells)}")


if __name__ == "__main__":
    main()
