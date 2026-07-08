#!/usr/bin/env python3
"""Aggregate TDB-AS k sweep and LS r sweep results from run_tdb_fair_grid.py."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Optional, Tuple


ROOT = Path(__file__).resolve().parent.parent
DATASETS = ["20newsgroups", "yahoo_subset"]
ALGOS = ["d-federaser", "d-fedosd", "d-fedrecovery", "d-oblivionis"]
SETTINGS = ["AS", "LS", "DSU", "Base"]


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
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


def _fmt(x: Optional[float], digits: int = 4) -> str:
    return "-" if x is None else f"{x:.{digits}f}"


def _fmt_pm(m: Optional[float], s: Optional[float], digits: int = 4) -> str:
    if m is None:
        return "-"
    if s is None:
        return f"{m:.{digits}f}"
    return f"{m:.{digits}f}±{s:.{digits}f}"


def _mean_std(values: Iterable[Optional[float]]) -> Tuple[Optional[float], Optional[float]]:
    arr = [float(v) for v in values if v is not None]
    if not arr:
        return None, None
    if len(arr) == 1:
        return arr[0], 0.0
    return mean(arr), pstdev(arr)


def _path_meta(root: Path, run_dir: Path) -> Tuple[Optional[str], Optional[str], Optional[int], Optional[str]]:
    rel = run_dir.relative_to(root)
    parts = rel.parts
    dataset = next((p for p in parts if p in DATASETS), None)
    algo = next((p for p in parts if p in ALGOS), None)
    seed = None
    strategy = None
    for p in parts:
        if p.startswith("seed"):
            try:
                seed = int(p.split("_", 1)[0].replace("seed", ""))
            except Exception:
                pass
        if p.startswith("strategy_"):
            strategy = p
    return dataset, algo, seed, strategy


def _setting_from_cfg(cfg: Dict[str, Any]) -> str:
    strategy = str(cfg.get("selection_strategy") or "")
    has_ls = bool(cfg.get("enable_param_selection"))
    if strategy == "tdb" and has_ls:
        return "DSU"
    if strategy == "tdb":
        return "AS"
    if has_ls:
        return "LS"
    return "Base"


def _extract_row(root: Path, history_path: Path) -> Optional[Dict[str, Any]]:
    run_dir = history_path.parent
    cfg_path = run_dir / "dfu_config.json"
    hist = _load_json(history_path)
    cfg = _load_json(cfg_path)
    if not hist or not cfg:
        return None

    dataset, algo, seed, strategy_dir = _path_meta(root, run_dir)
    final = hist.get("final_stats") or {}
    setting = _setting_from_cfg(cfg)
    selection_diag = cfg.get("selection_diagnostics") or {}
    solver = selection_diag.get("solver") or {}

    row: Dict[str, Any] = {
        "dataset": dataset,
        "algorithm": algo or cfg.get("dfu_algorithm"),
        "seed": seed if seed is not None else cfg.get("seed"),
        "setting": setting,
        "k": cfg.get("selection_count") if setting in {"AS", "DSU"} else None,
        "r": cfg.get("param_selection_ratio") if setting in {"LS", "DSU"} else None,
        "strategy_dir": strategy_dir,
        "selected_agents": " ".join(str(x) for x in (cfg.get("selected_agents") or [])),
        "selected_agents_count": cfg.get("selected_agents_count") or len(cfg.get("selected_agents") or []),
        "macro_f1_mean": _to_float(final.get("macro_f1_mean")),
        "macro_f1_std": _to_float(final.get("macro_f1_std")),
        "macro_f1_best": _to_float(final.get("macro_f1_best")),
        "best_agent_id": final.get("best_agent_id"),
        "mia_auc": _to_float(final.get("mia_auc")),
        "mia_auc_std": _to_float(final.get("mia_auc_std")),
        "trajectory_l1": _to_float(selection_diag.get("trajectory_l1")),
        "trajectory_l2": _to_float(selection_diag.get("trajectory_l2")),
        "label_l1": _to_float(selection_diag.get("label_l1")),
        "target_exposure": _to_float(selection_diag.get("target_exposure")),
        "solver_success": solver.get("solver_success"),
        "solve_time_sec": _to_float(solver.get("solve_time_sec")),
        "run_dir": str(run_dir),
        "history_path": str(history_path),
    }
    return row


def load_rows(root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for history_path in sorted(root.rglob("history.json")):
        row = _extract_row(root, history_path)
        if row:
            rows.append(row)

    # Keep the newest run if a cell was repeated.
    buckets: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row["dataset"], row["algorithm"], row["seed"], row["setting"], row.get("k"), row.get("r"))
        buckets[key].append(row)

    deduped = []
    for group in buckets.values():
        deduped.append(max(group, key=lambda r: Path(str(r["history_path"])).stat().st_mtime))
    return sorted(
        deduped,
        key=lambda r: (
            DATASETS.index(str(r["dataset"])) if r["dataset"] in DATASETS else 99,
            ALGOS.index(str(r["algorithm"])) if r["algorithm"] in ALGOS else 99,
            int(r["seed"] or 0),
            SETTINGS.index(str(r["setting"])) if r["setting"] in SETTINGS else 99,
            float(r["k"] or -1),
            float(r["r"] or -1),
        ),
    )


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields = [
        "dataset",
        "algorithm",
        "seed",
        "setting",
        "k",
        "r",
        "selected_agents",
        "selected_agents_count",
        "macro_f1_mean",
        "macro_f1_std",
        "macro_f1_best",
        "best_agent_id",
        "mia_auc",
        "mia_auc_std",
        "trajectory_l1",
        "trajectory_l2",
        "label_l1",
        "target_exposure",
        "solver_success",
        "solve_time_sec",
        "run_dir",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in fields})


def aggregate(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row["dataset"], row["algorithm"], row["setting"], row.get("k"), row.get("r"))
        buckets[key].append(row)

    out: List[Dict[str, Any]] = []
    for (dataset, algo, setting, k, r), group in buckets.items():
        f1_best_m, f1_best_s = _mean_std(row.get("macro_f1_best") for row in group)
        f1_mean_m, f1_mean_s = _mean_std(row.get("macro_f1_mean") for row in group)
        mia_m, mia_s = _mean_std(row.get("mia_auc") for row in group)
        traj_m, traj_s = _mean_std(row.get("trajectory_l1") for row in group)
        label_m, label_s = _mean_std(row.get("label_l1") for row in group)
        exposure_m, exposure_s = _mean_std(row.get("target_exposure") for row in group)
        solve_m, solve_s = _mean_std(row.get("solve_time_sec") for row in group)
        out.append(
            {
                "dataset": dataset,
                "algorithm": algo,
                "setting": setting,
                "k": k,
                "r": r,
                "n": len(group),
                "seeds": ",".join(str(int(row["seed"])) for row in sorted(group, key=lambda x: int(x["seed"]))),
                "macro_f1_best_mean": f1_best_m,
                "macro_f1_best_std": f1_best_s,
                "macro_f1_mean_mean": f1_mean_m,
                "macro_f1_mean_std": f1_mean_s,
                "mia_auc_mean": mia_m,
                "mia_auc_std": mia_s,
                "trajectory_l1_mean": traj_m,
                "trajectory_l1_std": traj_s,
                "label_l1_mean": label_m,
                "label_l1_std": label_s,
                "target_exposure_mean": exposure_m,
                "target_exposure_std": exposure_s,
                "solve_time_sec_mean": solve_m,
                "solve_time_sec_std": solve_s,
            }
        )
    return sorted(
        out,
        key=lambda r: (
            DATASETS.index(str(r["dataset"])) if r["dataset"] in DATASETS else 99,
            ALGOS.index(str(r["algorithm"])) if r["algorithm"] in ALGOS else 99,
            SETTINGS.index(str(r["setting"])) if r["setting"] in SETTINGS else 99,
            float(r["k"] or -1),
            float(r["r"] or -1),
        ),
    )


def write_agg_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields = [
        "dataset",
        "algorithm",
        "setting",
        "k",
        "r",
        "n",
        "seeds",
        "macro_f1_best_mean",
        "macro_f1_best_std",
        "macro_f1_mean_mean",
        "macro_f1_mean_std",
        "mia_auc_mean",
        "mia_auc_std",
        "trajectory_l1_mean",
        "trajectory_l1_std",
        "label_l1_mean",
        "label_l1_std",
        "target_exposure_mean",
        "target_exposure_std",
        "solve_time_sec_mean",
        "solve_time_sec_std",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in fields})


def best_rows(agg_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in agg_rows:
        if row["setting"] not in {"AS", "LS", "DSU", "Base"}:
            continue
        buckets[(str(row["dataset"]), str(row["algorithm"]), str(row["setting"]))].append(row)
    out = []
    for group in buckets.values():
        out.append(max(group, key=lambda r: float(r.get("macro_f1_best_mean") or -1)))
    return sorted(
        out,
        key=lambda r: (
            DATASETS.index(str(r["dataset"])) if r["dataset"] in DATASETS else 99,
            ALGOS.index(str(r["algorithm"])) if r["algorithm"] in ALGOS else 99,
            SETTINGS.index(str(r["setting"])) if r["setting"] in SETTINGS else 99,
        ),
    )


def maybe_plot(agg_rows: List[Dict[str, Any]], out_prefix: Path) -> List[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    written: List[Path] = []
    for setting, x_key, name in [("AS", "k", "as_k"), ("LS", "r", "ls_r")]:
        rows = [r for r in agg_rows if r["setting"] == setting]
        if not rows:
            continue
        fig, axes = plt.subplots(2, 4, figsize=(18, 7), sharey=False)
        for d_i, dataset in enumerate(DATASETS):
            for a_i, algo in enumerate(ALGOS):
                ax = axes[d_i][a_i]
                group = [r for r in rows if r["dataset"] == dataset and r["algorithm"] == algo]
                group = sorted(group, key=lambda r: float(r[x_key] or 0))
                if group:
                    xs = [float(r[x_key]) for r in group]
                    ys = [float(r["macro_f1_best_mean"] or 0) * 100.0 for r in group]
                    es = [float(r["macro_f1_best_std"] or 0) * 100.0 for r in group]
                    ax.errorbar(xs, ys, yerr=es, marker="o", linewidth=1.8, capsize=3)
                ax.set_title(f"{dataset} / {algo}")
                ax.set_xlabel("k" if setting == "AS" else "r")
                ax.set_ylabel("F1 (%)")
                ax.grid(True, alpha=0.25)
        fig.tight_layout()
        out_path = out_prefix.with_name(f"{out_prefix.name}_{name}_f1.png")
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        written.append(out_path)
    return written


def write_md(path: Path, rows: List[Dict[str, Any]], agg_rows: List[Dict[str, Any]], best: List[Dict[str, Any]], plots: List[Path]) -> None:
    best_by_key = {(r["dataset"], r["algorithm"], r["setting"]): r for r in best}
    lines: List[str] = []
    lines.append("# TDB-AS / LS Hyperparameter Sweep")
    lines.append("")
    lines.append(f"- Per-run rows: `{len(rows)}`")
    lines.append(f"- Aggregate rows: `{len(agg_rows)}`")
    lines.append("- AS sweep: `k=1..9`.")
    lines.append("- LS sweep: `r=0.1..1.0`.")
    lines.append("- Metric reported here: `macro_f1_best` on the public test set, averaged across seeds.")
    if plots:
        lines.append("")
        for p in plots:
            try:
                rel = p.relative_to(ROOT)
            except Exception:
                rel = p
            lines.append(f"![{p.stem}](/home/xzq/private/llm-dfl-0525/{rel})")
    lines.append("")

    for dataset in DATASETS:
        lines.append(f"## {dataset}")
        lines.append("")
        lines.append("| Method | Best AS k | AS F1 | Best LS r | LS F1 |")
        lines.append("|---|---:|---:|---:|---:|")
        for algo in ALGOS:
            as_row = best_by_key.get((dataset, algo, "AS"))
            ls_row = best_by_key.get((dataset, algo, "LS"))
            lines.append(
                "| "
                f"{algo} | "
                f"{as_row.get('k') if as_row else '-'} | "
                f"{_fmt_pm(_to_float(as_row.get('macro_f1_best_mean')) if as_row else None, _to_float(as_row.get('macro_f1_best_std')) if as_row else None)} | "
                f"{ls_row.get('r') if ls_row else '-'} | "
                f"{_fmt_pm(_to_float(ls_row.get('macro_f1_best_mean')) if ls_row else None, _to_float(ls_row.get('macro_f1_best_std')) if ls_row else None)} |"
            )
        lines.append("")

        lines.append("### AS k Curve")
        lines.append("")
        lines.append("| Method | k | F1 | MIA AUC | traj L1 | label L1 | target exposure |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for algo in ALGOS:
            for row in [r for r in agg_rows if r["dataset"] == dataset and r["algorithm"] == algo and r["setting"] == "AS"]:
                lines.append(
                    "| "
                    f"{algo} | {row.get('k')} | "
                    f"{_fmt_pm(_to_float(row.get('macro_f1_best_mean')), _to_float(row.get('macro_f1_best_std')))} | "
                    f"{_fmt_pm(_to_float(row.get('mia_auc_mean')), _to_float(row.get('mia_auc_std')))} | "
                    f"{_fmt_pm(_to_float(row.get('trajectory_l1_mean')), _to_float(row.get('trajectory_l1_std')))} | "
                    f"{_fmt_pm(_to_float(row.get('label_l1_mean')), _to_float(row.get('label_l1_std')))} | "
                    f"{_fmt_pm(_to_float(row.get('target_exposure_mean')), _to_float(row.get('target_exposure_std')))} |"
                )
        lines.append("")

        lines.append("### LS r Curve")
        lines.append("")
        lines.append("| Method | r | F1 | MIA AUC |")
        lines.append("|---|---:|---:|---:|")
        for algo in ALGOS:
            for row in [r for r in agg_rows if r["dataset"] == dataset and r["algorithm"] == algo and r["setting"] == "LS"]:
                lines.append(
                    "| "
                    f"{algo} | {row.get('r')} | "
                    f"{_fmt_pm(_to_float(row.get('macro_f1_best_mean')), _to_float(row.get('macro_f1_best_std')))} | "
                    f"{_fmt_pm(_to_float(row.get('mia_auc_mean')), _to_float(row.get('mia_auc_std')))} |"
                )
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="artifacts/tdb_as_ls_k1to9_r0p1to1_seed424344_20260526")
    ap.add_argument("--out_prefix", default="reports/tdb_as_ls_k1to9_r0p1to1_seed424344_20260526")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_absolute():
        root = ROOT / root
    out_prefix = Path(args.out_prefix)
    if not out_prefix.is_absolute():
        out_prefix = ROOT / out_prefix

    rows = load_rows(root)
    agg_rows = aggregate(rows)
    best = best_rows(agg_rows)
    all_csv = out_prefix.with_name(out_prefix.name + "_all_rows.csv")
    agg_csv = out_prefix.with_name(out_prefix.name + "_aggregate.csv")
    best_csv = out_prefix.with_name(out_prefix.name + "_best.csv")
    md_path = out_prefix.with_suffix(".md")

    write_csv(all_csv, rows)
    write_agg_csv(agg_csv, agg_rows)
    write_agg_csv(best_csv, best)
    plots = maybe_plot(agg_rows, out_prefix)
    write_md(md_path, rows, agg_rows, best, plots)

    print(f"Wrote: {all_csv}")
    print(f"Wrote: {agg_csv}")
    print(f"Wrote: {best_csv}")
    for p in plots:
        print(f"Wrote: {p}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
