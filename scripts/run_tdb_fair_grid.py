#!/usr/bin/env python3
"""Parallel fair-grid runner for TDB-AS / LS / DSU smoke and sweeps.

Fairness rule:
  Base / AS / LS / DSU share the same DFU hyperparameters. Only AS k and LS r
  vary. This script only changes:
    - selection_strategy / selection_count for AS
    - param_selection_ratio for LS
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.dfu.lora_param_selection import (  # noqa: E402
    compute_module_sensitivities,
    compute_module_sensitivities_relative,
)
from src.dfu.snapshot_loader import SnapshotLoader  # noqa: E402


SNAPSHOTS: Dict[tuple[str, int], str] = {
    ("20newsgroups", 42): "checkpoints/20newsgroups/K10/G10_L5/alpha0.5/seed42_20251220_074624",
    ("20newsgroups", 43): "checkpoints/20newsgroups/K10/G10_L5/alpha0.5/seed43_20260209_075703",
    ("20newsgroups", 44): "checkpoints/20newsgroups/K10/G10_L5/alpha0.5/seed44_20260209_082328",
    ("yahoo_subset", 42): "checkpoints/yahoo_subset/K10/G10_L5/alpha0.5/seed42_20251223_081958",
    ("yahoo_subset", 43): "checkpoints/yahoo_subset/K10/G10_L5/alpha0.5/seed43_20260209_081351",
    ("yahoo_subset", 44): "checkpoints/yahoo_subset/K10/G10_L5/alpha0.5/seed44_20260209_083957",
}


def csv_list(value: str) -> List[str]:
    return [x.strip() for x in str(value).split(",") if x.strip()]


def fmt_ratio(value: str | float) -> str:
    s = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return s if s else "0"


def algo_args(algo: str, profile: str) -> List[str]:
    """Shared method-specific DFU args. Same for Base/AS/LS/DSU."""
    if profile == "smoke":
        if algo == "d-federaser":
            return ["--calibration_steps", "1", "--calibration_interval", "10"]
        if algo == "d-fedosd":
            return [
                "--unlearn_rounds", "1",
                "--unlearn_lr", "1e-3",
                "--recovery_rounds", "1",
                "--recovery_local_steps", "1",
                "--recovery_lr", "1e-3",
                "--retain_grad_samples", "2",
            ]
        if algo == "d-fedrecovery":
            return [
                "--correction_weight", "5.0",
                "--noise_std", "0.0",
                "--recovery_rounds", "1",
                "--recovery_local_steps", "1",
                "--recovery_lr", "1e-3",
            ]
        if algo == "d-oblivionis":
            return [
                "--unlearn_rounds", "1",
                "--unlearn_lr", "5e-4",
                "--unlearn_local_steps", "1",
                "--propagation_rounds", "1",
                "--propagation_local_steps", "1",
                "--propagation_lr", "1e-3",
            ]
    elif profile == "paper":
        if algo == "d-federaser":
            return ["--calibration_steps", "5", "--calibration_interval", "2"]
        if algo == "d-fedosd":
            return [
                "--unlearn_rounds", "3",
                "--unlearn_lr", "1e-3",
                "--recovery_rounds", "2",
                "--recovery_local_steps", "5",
                "--recovery_lr", "1e-3",
                "--retain_grad_samples", "50",
            ]
        if algo == "d-fedrecovery":
            return [
                "--correction_weight", "5.0",
                "--noise_std", "0.0",
                "--recovery_rounds", "3",
                "--recovery_local_steps", "5",
                "--recovery_lr", "1e-3",
            ]
        if algo == "d-oblivionis":
            return [
                "--unlearn_rounds", "1",
                "--unlearn_lr", "5e-4",
                "--propagation_rounds", "3",
                "--propagation_lr", "1e-3",
            ]
    raise ValueError(f"Unsupported algorithm/profile: {algo}/{profile}")


@dataclass
class Task:
    name: str
    cmd: List[str]
    log_path: Path


@dataclass
class Running:
    proc: subprocess.Popen
    gpu: str
    task: Task
    log_file: object


def _log_has_completed(log_path: Path) -> bool:
    if not log_path.exists():
        return False
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False
    return "DFU completed" in text or "D-FedOSD completed" in text or "D-FedRecovery 完成" in text


def build_tasks(args: argparse.Namespace) -> List[Task]:
    tasks: List[Task] = []
    datasets = csv_list(args.datasets)
    algorithms = csv_list(args.algorithms)
    seeds = [int(x) for x in csv_list(args.seeds)]
    settings = csv_list(args.settings)
    ks = [int(x) for x in csv_list(args.ks)]
    rs = [fmt_ratio(x) for x in csv_list(args.rs)]

    out_dir = Path(args.out_dir)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    for dataset in datasets:
        for seed in seeds:
            snapshot = SNAPSHOTS.get((dataset, seed))
            if snapshot is None:
                raise ValueError(f"No known snapshot for {dataset} seed {seed}")
            if not (ROOT / snapshot).exists():
                raise FileNotFoundError(ROOT / snapshot)

            for algo in algorithms:
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

                sens_cache = (
                    out_dir / "sens_cache" / f"{dataset}_{Path(snapshot).name}_agent0.json"
                )

                if "base" in settings:
                    cmd = common + ["--selection_strategy", "full"]
                    if args.base_uses_lora_ratio_one:
                        cmd += [
                            "--enable_param_selection",
                            "--param_selection_mode", args.param_selection_mode,
                            "--param_selection_ratio", "1.0",
                            "--param_sensitivity_cache", str(sens_cache),
                        ]
                        if args.param_relative_sensitivity:
                            cmd += ["--param_relative_sensitivity"]
                        cmd += ["--param_sensitivity_alpha", str(args.param_sensitivity_alpha)]
                    name = f"{dataset}_{algo}_seed{seed}_base"
                    task = Task(name, cmd, logs_dir / f"{name}.log")
                    if args.skip_completed_logs and _log_has_completed(task.log_path):
                        print(f"[skip-completed] {name} log={task.log_path}", flush=True)
                    else:
                        tasks.append(task)

                if "as" in settings:
                    for k in ks:
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
                        ]
                        name = f"{dataset}_{algo}_seed{seed}_as_k{k}"
                        task = Task(name, cmd, logs_dir / f"{name}.log")
                        if args.skip_completed_logs and _log_has_completed(task.log_path):
                            print(f"[skip-completed] {name} log={task.log_path}", flush=True)
                        else:
                            tasks.append(task)

                if "ls" in settings:
                    for r in rs:
                        cmd = common + [
                            "--selection_strategy", "full",
                            "--enable_param_selection",
                            "--param_selection_mode", args.param_selection_mode,
                            "--param_selection_ratio", r,
                            "--param_sensitivity_cache", str(sens_cache),
                        ]
                        if args.param_relative_sensitivity:
                            cmd += ["--param_relative_sensitivity"]
                        cmd += ["--param_sensitivity_alpha", str(args.param_sensitivity_alpha)]
                        name = f"{dataset}_{algo}_seed{seed}_ls_r{r}"
                        task = Task(name, cmd, logs_dir / f"{name}.log")
                        if args.skip_completed_logs and _log_has_completed(task.log_path):
                            print(f"[skip-completed] {name} log={task.log_path}", flush=True)
                        else:
                            tasks.append(task)

                if "dsu" in settings:
                    for k in ks:
                        for r in rs:
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
                            ]
                            if args.param_relative_sensitivity:
                                cmd += ["--param_relative_sensitivity"]
                            cmd += ["--param_sensitivity_alpha", str(args.param_sensitivity_alpha)]
                            name = f"{dataset}_{algo}_seed{seed}_dsu_k{k}_r{r}"
                            task = Task(name, cmd, logs_dir / f"{name}.log")
                            if args.skip_completed_logs and _log_has_completed(task.log_path):
                                print(f"[skip-completed] {name} log={task.log_path}", flush=True)
                            else:
                                tasks.append(task)

    return tasks


def precompute_sensitivity_caches(args: argparse.Namespace) -> None:
    settings = set(csv_list(args.settings))
    needs_cache = "ls" in settings or "dsu" in settings or (
        "base" in settings and args.base_uses_lora_ratio_one
    )
    if not needs_cache:
        return

    out_dir = Path(args.out_dir)
    cache_dir = out_dir / "sens_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    datasets = csv_list(args.datasets)
    seeds = [int(x) for x in csv_list(args.seeds)]
    for dataset in datasets:
        for seed in seeds:
            snapshot = SNAPSHOTS.get((dataset, seed))
            if snapshot is None:
                raise ValueError(f"No known snapshot for {dataset} seed {seed}")
            cache_path = cache_dir / f"{dataset}_{Path(snapshot).name}_agent0.json"
            if cache_path.exists():
                print(f"[sens-cache] exists {cache_path}", flush=True)
                continue

            print(f"[sens-cache] computing {cache_path}", flush=True)
            loader = SnapshotLoader(str(ROOT / snapshot))
            if args.param_relative_sensitivity:
                sensitivities = compute_module_sensitivities_relative(
                    loader,
                    target_agent=0,
                    alpha=args.param_sensitivity_alpha,
                    verbose=True,
                )
            else:
                sensitivities = compute_module_sensitivities(
                    loader,
                    target_agent=0,
                    verbose=True,
                )

            tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
            with tmp_path.open("w", encoding="utf-8") as f:
                json.dump(sensitivities, f, ensure_ascii=False, indent=2)
            tmp_path.replace(cache_path)
            print(f"[sens-cache] saved {cache_path}", flush=True)


def run_parallel(tasks: List[Task], gpus: List[str]) -> None:
    pending = list(tasks)
    running: List[Running] = []
    failures: List[str] = []

    def free_gpus() -> List[str]:
        busy = {r.gpu for r in running}
        return [g for g in gpus if g not in busy]

    while pending or running:
        for r in list(running):
            code = r.proc.poll()
            if code is None:
                continue
            r.log_file.close()
            running.remove(r)
            status = "done" if code == 0 else f"failed({code})"
            print(f"[{status}] gpu={r.gpu} {r.task.name} log={r.task.log_path}", flush=True)
            if code != 0:
                failures.append(f"{r.task.name}: {r.task.log_path}")

        while pending and free_gpus():
            gpu = free_gpus()[0]
            task = pending.pop(0)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            env["LLMDFL_ALLOWED_PHYSICAL_GPUS"] = gpu
            env.setdefault("LLMDFL_LOCAL_FILES_ONLY", "1")
            env.setdefault("TOKENIZERS_PARALLELISM", "false")
            env.setdefault("PYTHONUNBUFFERED", "1")
            task.log_path.parent.mkdir(parents=True, exist_ok=True)
            f = task.log_path.open("w", encoding="utf-8")
            f.write(" ".join(task.cmd) + "\n\n")
            f.flush()
            proc = subprocess.Popen(
                task.cmd,
                cwd=str(ROOT),
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
            )
            running.append(Running(proc=proc, gpu=gpu, task=task, log_file=f))
            print(f"[launch] gpu={gpu} {task.name} log={task.log_path}", flush=True)

        time.sleep(2)

    if failures:
        raise SystemExit("Failed tasks:\n" + "\n".join(failures))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", default="20newsgroups,yahoo_subset")
    ap.add_argument("--algorithms", default="d-federaser")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--settings", default="base,as,ls,dsu")
    ap.add_argument("--ks", default="1,2,3,4,5,6,7,8,9")
    ap.add_argument("--rs", default="0.25,0.5,0.75,1.0")
    ap.add_argument("--physical_gpus", default="0,1,2,3")
    ap.add_argument("--out_dir", default="artifacts/tdb_fair_grid_20260525")
    ap.add_argument("--profile", choices=["smoke", "paper"], default="smoke")
    ap.add_argument("--max_eval_samples", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum_steps", type=int, default=1)
    ap.add_argument("--lr", default="1e-4")
    ap.add_argument("--federaser_calibration_steps", type=int, default=None)
    ap.add_argument("--federaser_calibration_interval", type=int, default=None)
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
    ap.add_argument(
        "--precompute_sensitivity_cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute each LoRA sensitivity cache once before launching parallel jobs.",
    )
    ap.add_argument(
        "--base_uses_lora_ratio_one",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run Base through LS code path with r=1.0, equivalent to all LoRA modules.",
    )
    ap.add_argument(
        "--skip_completed_logs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip tasks whose log already contains a DFU completion marker.",
    )
    args = ap.parse_args()

    gpus = csv_list(args.physical_gpus)
    if args.precompute_sensitivity_cache:
        precompute_sensitivity_caches(args)
    tasks = build_tasks(args)
    print(f"Built {len(tasks)} tasks; gpus={gpus}; out_dir={args.out_dir}", flush=True)
    run_parallel(tasks, gpus)
    print(f"Fair grid completed: {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
