#!/usr/bin/env python3
"""Run AS+LS (agent+LoRA) combined DFU experiments for ablation.

This script is meant to be run AFTER the sweep jobs finish, to fill the only
missing combo: selection_strategy=ours (agent selection) + enable_param_selection
(LoRA module selection).

It supports:
  - mode=best_best: run one (k_best, r_best) per dataset+method
  - mode=sweep_k: fix r=r_best and sweep k=1..9 (seed42-only suggested)

It writes results under:
  <output_root>/seed{seed}/<dataset>/<method>/strategy_ours_count{K}_lora{R}_topratio_ours/...

Selection of (k,r):
  - If --config_json is provided, uses the user-provided mapping directly.
  - Otherwise, uses --sweep_root (+ --sweep_seeds) to pick best-k/r from existing sweeps.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.ablation_sweep_utils import (
    COUNT_GRID,
    RATIO_GRID,
    extract_macro_f1,
    latest_history_json,
    mean_std,
    require_seed_roots,
    strategy_dir_agent_count,
    strategy_dir_both,
    strategy_dir_lora_ratio,
)
from src.dfu.snapshot_loader import SnapshotLoader
from src.dfu.lora_param_selection import compute_module_sensitivities


DATASETS = ["20newsgroups", "yahoo_subset"]
ALGORITHMS = ["d-federaser", "d-fedosd", "d-fedrecovery", "d-oblivionis"]


@dataclass(frozen=True)
class Job:
    dataset: str
    algorithm: str
    selection_count: int
    param_ratio: float
    cmd: List[str]
    log_path: Path


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_sensitivity_cache(dfl_snapshot: Path, target_agent: int, cache_path: Path) -> None:
    if cache_path.exists():
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    loader = SnapshotLoader(str(dfl_snapshot))
    sens = compute_module_sensitivities(loader, target_agent=target_agent, verbose=True)
    cache_path.write_text(json.dumps(sens, ensure_ascii=False, indent=2), encoding="utf-8")


def _algo_hparams(*, algorithm: str, local_steps: int) -> List[str]:
    algo = str(algorithm)
    ls = int(local_steps)
    if algo == "d-federaser":
        return [
            "--calibration_steps",
            "3",
            "--calibration_interval",
            "2",
            "--lr",
            "1e-3",
            "--batch_size",
            "4",
            "--grad_accum_steps",
            "2",
        ]
    if algo == "d-oblivionis":
        return [
            "--unlearn_rounds",
            "1",
            "--unlearn_local_steps",
            str(ls),
            "--unlearn_lr",
            "5e-4",
            "--propagation_rounds",
            "3",
            "--propagation_local_steps",
            str(ls),
            "--propagation_lr",
            "1e-3",
            "--batch_size",
            "4",
            "--grad_accum_steps",
            "2",
        ]
    if algo == "d-fedosd":
        return [
            "--unlearn_rounds",
            "3",
            "--unlearn_lr",
            "1e-3",
            "--recovery_rounds",
            "2",
            "--recovery_local_steps",
            str(ls),
            "--recovery_lr",
            "1e-3",
            "--batch_size",
            "4",
            "--grad_accum_steps",
            "2",
        ]
    if algo == "d-fedrecovery":
        return [
            "--correction_weight",
            "5.0",
            "--noise_std",
            "0.0",
            "--recovery_rounds",
            "3",
            "--recovery_local_steps",
            str(ls),
            "--recovery_lr",
            "1e-3",
            "--batch_size",
            "4",
            "--grad_accum_steps",
            "2",
        ]
    return []


def _collect_seed_series(seed_root: Path, *, dataset: str, algo: str, sweep: str) -> Dict[float, float]:
    xs = COUNT_GRID if sweep == "agent_count" else RATIO_GRID
    out: Dict[float, float] = {}
    for x in xs:
        if sweep == "agent_count":
            strat = strategy_dir_agent_count(int(x))
        else:
            strat = strategy_dir_lora_ratio(float(x))
        hp = latest_history_json(seed_root, dataset=dataset, algorithm=algo, strategy_dir=strat)
        if hp is None:
            continue
        out[float(x)] = float(extract_macro_f1(hp))
    return out


def _pick_best_x(series_by_seed: Dict[int, Dict[float, float]]) -> float:
    per_x: Dict[float, List[float]] = {}
    for _, series in series_by_seed.items():
        for x, v in series.items():
            per_x.setdefault(float(x), []).append(float(v))
    best_x: Optional[float] = None
    best_mean: Optional[float] = None
    for x in sorted(per_x.keys()):
        m, _ = mean_std(per_x[x])
        if best_mean is None or m > best_mean + 1e-12:
            best_mean = m
            best_x = x
    if best_x is None:
        raise ValueError("Cannot pick best x: no sweep points found")
    return float(best_x)


def _run_jobs(jobs: List[Job], gpus: List[int], *, seed: int, dry_run: bool) -> None:
    pending = jobs[:]
    running: Dict[subprocess.Popen, Tuple[Job, int]] = {}
    free_gpus = list(gpus)

    def launch(job: Job, gpu_id: int) -> subprocess.Popen:
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = str(seed)
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        env.setdefault("LLMDFL_LOCAL_FILES_ONLY", "1")
        env.setdefault("HF_HUB_OFFLINE", "1")
        env.setdefault("TRANSFORMERS_OFFLINE", "1")
        # Physical GPU restriction: only allow GPU2/3.
        env.setdefault("CUDA_VISIBLE_DEVICES", "2,3")

        cmd = job.cmd + ["--gpu", str(gpu_id)]
        job.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(job.log_path, "w", encoding="utf-8") as f:
            f.write(" ".join(cmd) + "\n\n")
        log_f = open(job.log_path, "a", encoding="utf-8")
        return subprocess.Popen(cmd, cwd=str(ROOT), stdout=log_f, stderr=subprocess.STDOUT, env=env)

    while pending or running:
        while pending and free_gpus:
            job = pending.pop(0)
            gpu_id = free_gpus.pop(0)
            print(
                f"[LAUNCH][gpu={gpu_id}] {job.dataset} {job.algorithm} "
                f"k={job.selection_count} r={job.param_ratio} -> {job.log_path}",
                flush=True,
            )
            if dry_run:
                free_gpus.append(gpu_id)
                continue
            proc = launch(job, gpu_id)
            running[proc] = (job, gpu_id)
            time.sleep(1.0)

        finished: List[subprocess.Popen] = []
        for proc, (job, gpu_id) in list(running.items()):
            ret = proc.poll()
            if ret is None:
                continue
            finished.append(proc)
            if ret != 0:
                raise RuntimeError(f"Job failed (exit={ret}): {job.log_path}")
            free_gpus.append(gpu_id)
            print(
                f"[DONE] {job.dataset} {job.algorithm} k={job.selection_count} r={job.param_ratio}",
                flush=True,
            )

        for proc in finished:
            running.pop(proc, None)

        if running:
            time.sleep(10.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_json", type=str, default="", help="Optional JSON mapping dataset->algo->{k,ratio}. Overrides sweep picking.")
    parser.add_argument("--sweep_root", type=str, default="", help="Existing sweep root used to pick best k/r (when --config_json is not set).")
    parser.add_argument("--sweep_seeds", type=str, default="42,43,44", help="Seeds under sweep_root to use for best-k/r.")
    parser.add_argument("--output_root", type=str, default="", help="Output root. Default: dfu_ablation_as_ls_<timestamp>.")
    parser.add_argument("--seed", type=int, default=42, help="Unlearning RNG seed for the new AS+LS runs.")
    parser.add_argument("--target_agent", type=int, default=0)
    parser.add_argument("--max_eval_samples", type=int, default=100)
    parser.add_argument("--datasets", type=str, default=",".join(DATASETS))
    parser.add_argument("--algorithms", type=str, default=",".join(ALGORITHMS))
    parser.add_argument(
        "--gpus",
        type=str,
        default="0,1",
        help="Logical GPU ids within CUDA_VISIBLE_DEVICES (default: 0,1 for physical 2,3).",
    )
    parser.add_argument("--mode", type=str, default="best_best", choices=["best_best", "sweep_k"])
    parser.add_argument("--k_values", type=str, default="1-9", help="Only used for mode=sweep_k (e.g., 1-9 or 1,3,5).")
    parser.add_argument(
        "--fixed_ratio",
        type=float,
        default=None,
        help="Override the picked LoRA ratio for all selected dataset/method blocks. Useful for targeted AS+LS k rescues.",
    )
    parser.add_argument("--skip_existing", action="store_true", help="Skip AS+LS points already present under output_root/seed{seed}.")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--save_lora_states",
        action="store_true",
        help="Save final LoRA state .pt files. Default is off for disk-safe full-paper reruns.",
    )
    parser.add_argument(
        "--snapshot_20news",
        type=str,
        default="checkpoints/20newsgroups/K10/G10_L5/alpha0.5/seed42_20251220_074624",
        help="DFL snapshot path for 20newsgroups.",
    )
    parser.add_argument(
        "--snapshot_yahoo",
        type=str,
        default="checkpoints/yahoo_subset/K10/G10_L5/alpha0.5/seed42_20251223_081958",
        help="DFL snapshot path for yahoo_subset.",
    )
    args = parser.parse_args()

    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    algos = [x.strip() for x in args.algorithms.split(",") if x.strip()]
    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    if not gpus:
        raise SystemExit("--gpus is empty")

    if not args.output_root:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_root = f"dfu_ablation_as_ls_{ts}"
    output_root = Path(args.output_root)

    # Snapshot paths.
    snapshot_20 = Path(args.snapshot_20news)
    snapshot_yahoo = Path(args.snapshot_yahoo)
    if not snapshot_20.is_absolute():
        snapshot_20 = ROOT / snapshot_20
    if not snapshot_yahoo.is_absolute():
        snapshot_yahoo = ROOT / snapshot_yahoo

    if not (snapshot_20 / "config.json").exists():
        raise FileNotFoundError(f"20newsgroups snapshot not found/invalid: {snapshot_20}")
    if not (snapshot_yahoo / "config.json").exists():
        raise FileNotFoundError(f"yahoo_subset snapshot not found/invalid: {snapshot_yahoo}")
    dataset_snaps: Dict[str, Path] = {"20newsgroups": snapshot_20, "yahoo_subset": snapshot_yahoo}

    caches_dir = ROOT / "cache" / "sensitivities"
    cache_20 = caches_dir / f"20newsgroups_{snapshot_20.name}_agent{args.target_agent}_abs.json"
    cache_yahoo = caches_dir / f"yahoo_subset_{snapshot_yahoo.name}_agent{args.target_agent}_abs.json"
    _ensure_sensitivity_cache(snapshot_20, args.target_agent, cache_20)
    _ensure_sensitivity_cache(snapshot_yahoo, args.target_agent, cache_yahoo)
    dataset_cache: Dict[str, Path] = {"20newsgroups": cache_20, "yahoo_subset": cache_yahoo}

    # Parse k_values.
    ks: List[int]
    if args.k_values.strip() == "1-9":
        ks = list(range(1, 10))
    elif "-" in args.k_values:
        a, b = args.k_values.split("-", 1)
        ks = list(range(int(a), int(b) + 1))
    else:
        ks = [int(x.strip()) for x in args.k_values.split(",") if x.strip()]

    # Pick best k and r (either from config_json or from sweep_root).
    best_k: Dict[Tuple[str, str], int] = {}
    best_r: Dict[Tuple[str, str], float] = {}

    sweep_root = Path(args.sweep_root) if args.sweep_root else None
    sweep_seeds = [int(x) for x in args.sweep_seeds.split(",") if x.strip()]

    cfg_json_path = Path(args.config_json) if args.config_json else None
    cfg_map: Dict = {}
    if cfg_json_path:
        if not cfg_json_path.exists():
            raise FileNotFoundError(f"--config_json not found: {cfg_json_path}")
        cfg_map = _load_json(cfg_json_path)
        for dataset in datasets:
            if dataset not in cfg_map:
                raise KeyError(f"--config_json missing dataset: {dataset}")
            for algo in algos:
                if algo not in cfg_map[dataset]:
                    raise KeyError(f"--config_json missing algo: {dataset}/{algo}")
                item = cfg_map[dataset][algo]
                best_k[(dataset, algo)] = int(item["k"])
                best_r[(dataset, algo)] = float(item["ratio"])
    else:
        if sweep_root is None:
            raise SystemExit("--sweep_root is required when --config_json is not provided")
        sweep_seed_roots = require_seed_roots(sweep_root, sweep_seeds)
        for dataset in datasets:
            for algo in algos:
                agent_series = {seed: _collect_seed_series(sr, dataset=dataset, algo=algo, sweep="agent_count") for seed, sr in sweep_seed_roots.items()}
                lora_series = {seed: _collect_seed_series(sr, dataset=dataset, algo=algo, sweep="lora_ratio") for seed, sr in sweep_seed_roots.items()}
                best_k[(dataset, algo)] = int(round(_pick_best_x(agent_series)))
                best_r[(dataset, algo)] = float(_pick_best_x(lora_series))

    if args.fixed_ratio is not None:
        fixed_ratio = float(args.fixed_ratio)
        for dataset in datasets:
            for algo in algos:
                best_r[(dataset, algo)] = fixed_ratio

    # Build jobs.
    jobs: List[Job] = []
    for dataset in datasets:
        snap = dataset_snaps[dataset]
        cfg = _load_json(snap / "config.json")
        local_steps = int(cfg.get("local_steps") or 0)
        sens_cache = dataset_cache[dataset]

        for algo in algos:
            r = best_r[(dataset, algo)]
            if args.mode == "best_best":
                k_list = [best_k[(dataset, algo)]]
            else:
                k_list = ks

            for k in k_list:
                out_dir = output_root / f"seed{args.seed}"
                strategy_dir = strategy_dir_both(int(k), float(r))
                if args.skip_existing:
                    existing = latest_history_json(out_dir, dataset=dataset, algorithm=algo, strategy_dir=strategy_dir)
                    if existing is not None:
                        print(f"[SKIP] {dataset} {algo} k={int(k)} r={float(r):.1f}: {existing}", flush=True)
                        continue
                log_path = output_root / "logs_ablation" / f"seed{args.seed}" / dataset / algo / f"k{k}_r{r:.1f}.log"
                cmd = [
                    sys.executable,
                    str(ROOT / "scripts" / "run_dfu.py"),
                    "--dfl_snapshot",
                    str(snap),
                    "--dfu_algorithm",
                    algo,
                    "--output_dir",
                    str(out_dir),
                    "--target_agent",
                    str(args.target_agent),
                    "--seed",
                    str(args.seed),
                    "--max_eval_samples",
                    str(args.max_eval_samples),
                    "--eval_every",
                    "0",
                ]
                if not args.save_lora_states:
                    cmd += ["--no_save_lora_states"]
                cmd += _algo_hparams(algorithm=algo, local_steps=local_steps)
                cmd += [
                    "--selection_strategy",
                    "ours",
                    "--selection_count",
                    str(int(k)),
                    "--enable_param_selection",
                    "--param_selection_mode",
                    "top_ratio",
                    "--param_selection_ratio",
                    str(float(r)),
                    "--param_epsilon_W",
                    "0.0",
                    "--param_sensitivity_cache",
                    str(sens_cache),
                ]
                jobs.append(
                    Job(
                        dataset=dataset,
                        algorithm=algo,
                        selection_count=int(k),
                        param_ratio=float(r),
                        cmd=cmd,
                        log_path=log_path,
                    )
                )

    output_root.mkdir(parents=True, exist_ok=True)
    meta_path = output_root / "metadata" / f"ablation_as_ls_meta_seed{args.seed}.json"
    meta = {
        "config_json": (str(cfg_json_path) if cfg_json_path else ""),
        "sweep_root": (str(sweep_root) if sweep_root else ""),
        "sweep_seeds": (sweep_seeds if sweep_root else []),
        "output_root": str(output_root),
        "seed": args.seed,
        "datasets": datasets,
        "algorithms": algos,
        "mode": args.mode,
        "best_k": {f"{ds}::{algo}": k for (ds, algo), k in best_k.items()},
        "best_r": {f"{ds}::{algo}": r for (ds, algo), r in best_r.items()},
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[OK] wrote {meta_path}", flush=True)

    _run_jobs(jobs, gpus=gpus, seed=args.seed, dry_run=bool(args.dry_run))
    print(f"[OK] finished. output_root={output_root}", flush=True)


if __name__ == "__main__":
    main()
