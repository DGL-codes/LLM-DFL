#!/usr/bin/env python3
"""Run DSU with the per-cell best AS k and LS r from an AS/LS sweep report."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from run_tdb_fair_grid import (  # noqa: E402
    SNAPSHOTS,
    Task,
    _log_has_completed,
    algo_args,
    csv_list,
    fmt_ratio,
    run_parallel,
)


def read_best_as_ls(path: Path) -> Dict[Tuple[str, str], Tuple[int, str]]:
    best: Dict[Tuple[str, str], Dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            dataset = str(row.get("dataset") or "")
            algo = str(row.get("algorithm") or "")
            setting = str(row.get("setting") or "")
            if not dataset or not algo or setting not in {"AS", "LS"}:
                continue
            best.setdefault((dataset, algo), {})[setting] = row

    out: Dict[Tuple[str, str], Tuple[int, str]] = {}
    missing: List[str] = []
    for key, rows in sorted(best.items()):
        as_row = rows.get("AS")
        ls_row = rows.get("LS")
        if not as_row or not ls_row:
            missing.append(f"{key[0]}/{key[1]}")
            continue
        k = int(float(as_row.get("k") or 0))
        r = fmt_ratio(ls_row.get("r") or "0")
        if k <= 0 or float(r) <= 0:
            missing.append(f"{key[0]}/{key[1]} invalid k/r: k={k}, r={r}")
            continue
        out[key] = (k, r)

    if missing:
        raise SystemExit("Missing best AS/LS rows:\n" + "\n".join(missing))
    return out


def build_tasks(args: argparse.Namespace, best: Dict[Tuple[str, str], Tuple[int, str]]) -> List[Task]:
    out_dir = Path(args.out_dir)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    sens_root = Path(args.sens_cache_root)
    if not sens_root.is_absolute():
        sens_root = ROOT / sens_root

    tasks: List[Task] = []
    datasets = csv_list(args.datasets)
    algorithms = csv_list(args.algorithms)
    seeds = [int(x) for x in csv_list(args.seeds)]

    for dataset in datasets:
        for seed in seeds:
            snapshot = SNAPSHOTS.get((dataset, seed))
            if snapshot is None:
                raise ValueError(f"No known snapshot for {dataset} seed {seed}")
            if not (ROOT / snapshot).exists():
                raise FileNotFoundError(ROOT / snapshot)

            for algo in algorithms:
                k, r = best[(dataset, algo)]
                sens_cache = sens_root / f"{dataset}_{Path(snapshot).name}_agent0.json"
                common = [
                    sys.executable,
                    str(ROOT / "scripts" / "run_dfu.py"),
                    "--dfl_snapshot", snapshot,
                    "--dfu_algorithm", algo,
                    "--output_dir", str(out_dir),
                    "--target_agent", "0",
                    "--seed", str(seed),
                    "--batch_size", str(args.batch_size),
                    "--grad_accum_steps", str(args.grad_accum_steps),
                    "--lr", str(args.lr),
                    "--eval_every", "0",
                    "--max_eval_samples", str(args.max_eval_samples),
                    "--gpu", "0",
                    "--no_save_lora_states",
                ]
                common += algo_args(algo, args.profile)
                if algo == "d-federaser":
                    if args.federaser_calibration_steps is not None:
                        common += ["--calibration_steps", str(args.federaser_calibration_steps)]
                    if args.federaser_calibration_interval is not None:
                        common += ["--calibration_interval", str(args.federaser_calibration_interval)]

                cmd = common + [
                    "--selection_strategy", "tdb",
                    "--selection_count", str(k),
                    "--tdb_sketch_dim", str(args.tdb_sketch_dim),
                    "--tdb_max_intervals", str(args.tdb_max_intervals),
                    "--tdb_time_limit", str(args.tdb_time_limit),
                    "--tdb_alpha_u", str(args.tdb_alpha_u),
                    "--tdb_alpha_p", str(args.tdb_alpha_p),
                    "--tdb_alpha_q", str(args.tdb_alpha_q),
                    "--tdb_tau_q", str(args.tdb_tau_q),
                    "--tdb_aggregation_scope", args.tdb_aggregation_scope,
                    "--enable_param_selection",
                    "--param_selection_mode", args.param_selection_mode,
                    "--param_selection_ratio", r,
                    "--param_sensitivity_cache", str(sens_cache),
                    "--param_sensitivity_alpha", str(args.param_sensitivity_alpha),
                ]
                if args.param_relative_sensitivity:
                    cmd += ["--param_relative_sensitivity"]

                name = f"{dataset}_{algo}_seed{seed}_dsu_best_k{k}_r{r}"
                task = Task(name=name, cmd=cmd, log_path=logs_dir / f"{name}.log")
                if args.skip_completed_logs and _log_has_completed(task.log_path):
                    print(f"[skip-completed] {name} log={task.log_path}", flush=True)
                else:
                    tasks.append(task)
    return tasks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--best_csv", default="reports/tdb_as_ls_k1to9_r0p1to1_seed424344_20260526_best.csv")
    ap.add_argument("--datasets", default="20newsgroups,yahoo_subset")
    ap.add_argument("--algorithms", default="d-federaser,d-fedosd,d-fedrecovery,d-oblivionis")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--physical_gpus", default="0,1")
    ap.add_argument("--out_dir", default="artifacts/tdb_dsu_from_as_ls_best_20260527")
    ap.add_argument("--sens_cache_root", default="artifacts/tdb_as_ls_k1to9_r0p1to1_seed424344_20260526/sens_cache")
    ap.add_argument("--profile", choices=["smoke", "paper"], default="paper")
    ap.add_argument("--max_eval_samples", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum_steps", type=int, default=2)
    ap.add_argument("--lr", default="1e-3")
    ap.add_argument("--federaser_calibration_steps", type=int, default=5)
    ap.add_argument("--federaser_calibration_interval", type=int, default=2)
    ap.add_argument("--tdb_sketch_dim", type=int, default=16)
    ap.add_argument("--tdb_max_intervals", type=int, default=2)
    ap.add_argument("--tdb_time_limit", type=float, default=20.0)
    ap.add_argument("--tdb_alpha_u", type=float, default=1.0)
    ap.add_argument("--tdb_alpha_p", type=float, default=1.0)
    ap.add_argument("--tdb_alpha_q", type=float, default=0.1)
    ap.add_argument("--tdb_tau_q", type=float, default=0.0)
    ap.add_argument("--tdb_aggregation_scope", choices=["local", "global"], default="local")
    ap.add_argument("--param_selection_mode", choices=["epsilon", "top_ratio"], default="top_ratio")
    ap.add_argument("--param_relative_sensitivity", action="store_true")
    ap.add_argument("--param_sensitivity_alpha", type=float, default=1.0)
    ap.add_argument("--skip_completed_logs", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    best_csv = Path(args.best_csv)
    if not best_csv.is_absolute():
        best_csv = ROOT / best_csv
    best = read_best_as_ls(best_csv)
    tasks = build_tasks(args, best)
    gpus = csv_list(args.physical_gpus)
    print(f"Built {len(tasks)} DSU-best tasks; gpus={gpus}; out_dir={args.out_dir}", flush=True)
    run_parallel(tasks, gpus)
    print(f"DSU-best grid completed: {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
