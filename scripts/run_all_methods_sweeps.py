#!/usr/bin/env python3
"""Run DFU sweeps for 20newsgroups + yahoo_subset across all 4 algorithms.

This script is a superset runner to make it easy to generate:
- Agent-count sweeps (ours top-k agents, full params): k=1..9
- LoRA-ratio sweeps (full agents, top-ratio LoRA modules): r=0.1..1.0

It is safe to rerun with --skip_existing: it will reuse existing runs that already
have a completed history.json and only launch missing points.
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
from typing import Dict, List, Optional, Tuple, Union

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.dfu.snapshot_loader import SnapshotLoader
from src.dfu.lora_param_selection import compute_module_sensitivities


DATASETS = ["20newsgroups", "yahoo_subset"]
ALGORITHMS = ["d-federaser", "d-oblivionis", "d-fedosd", "d-fedrecovery"]


@dataclass(frozen=True)
class Job:
    dataset: str
    algorithm: str
    sweep: str  # "agent_count" or "lora_ratio"
    x: Union[int, float]
    cmd: List[str]
    output_prefix: Path
    log_path: Path


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_dfu_dir(prefix: Path) -> Path:
    candidates = [p for p in prefix.glob("dfu_*") if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No dfu_* directories found under: {prefix}")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _extract_macro_f1(history_path: Path) -> float:
    data = _load_json(history_path)
    final_stats = data.get("final_stats") or {}
    if "macro_f1_best" in final_stats and final_stats["macro_f1_best"] is not None:
        return float(final_stats["macro_f1_best"])
    if "macro_f1_mean" in final_stats and final_stats["macro_f1_mean"] is not None:
        return float(final_stats["macro_f1_mean"])

    unlearning_metrics = data.get("unlearning_metrics") or data.get("avg_metrics") or []
    if unlearning_metrics:
        last = unlearning_metrics[-1]
        if isinstance(last, dict):
            if "test_macro_f1_best" in last and last["test_macro_f1_best"] is not None:
                return float(last["test_macro_f1_best"])
            if "test_macro_f1" in last and last["test_macro_f1"] is not None:
                return float(last["test_macro_f1"])
            if "macro_f1" in last:
                return float(last["macro_f1"])

    raise ValueError(f"Cannot find macro-F1 in {history_path}")


def _strategy_dir(
    *,
    selection_strategy: str,
    selection_ratio: Optional[float],
    selection_count: Optional[int],
    enable_param_selection: bool,
    param_selection_ratio: float,
    param_epsilon_W: float,
    param_random_selection: bool,
    param_selection_mode: str,
) -> str:
    if selection_strategy == "full":
        strategy = "strategy_full"
    elif selection_strategy == "random":
        if selection_count is not None:
            strategy = f"strategy_random_count{selection_count}"
        else:
            strategy = f"strategy_random_ratio{selection_ratio}"
    elif selection_strategy == "ours":
        if selection_count is not None:
            strategy = f"strategy_ours_count{selection_count}"
        else:
            strategy = f"strategy_ours_ratio{selection_ratio}"
    else:
        strategy = f"strategy_{selection_strategy}"

    if enable_param_selection:
        lora_strategy = "random" if param_random_selection else "ours"
        mode = (param_selection_mode or "epsilon").lower()
        if mode == "top_ratio":
            strategy += f"_lora{param_selection_ratio}_topratio_{lora_strategy}"
        else:
            strategy += f"_lora{param_selection_ratio}_eps{param_epsilon_W}_{lora_strategy}"

    return strategy


def _output_prefix(*, output_root: Path, dfl_snapshot: Path, dfu_algorithm: str, strategy_dir: str) -> Path:
    cfg = _load_json(dfl_snapshot / "config.json")
    dataset = cfg.get("dataset", "unknown")
    num_agents = int(cfg["num_agents"])
    global_rounds = int(cfg.get("global_rounds", 0))
    local_steps = int(cfg.get("local_steps", 0))
    alpha = cfg.get("alpha", 0.5)
    return (
        output_root
        / dataset
        / dfu_algorithm
        / strategy_dir
        / f"K{num_agents}"
        / f"G{global_rounds}_L{local_steps}"
        / f"alpha{alpha}"
        / dfl_snapshot.name
    )


def _ensure_sensitivity_cache(dfl_snapshot: Path, target_agent: int, cache_path: Path) -> None:
    if cache_path.exists():
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    loader = SnapshotLoader(str(dfl_snapshot))
    sens = compute_module_sensitivities(loader, target_agent=target_agent, verbose=True)
    cache_path.write_text(json.dumps(sens, ensure_ascii=False, indent=2), encoding="utf-8")


def _algo_hparams(*, algorithm: str, local_steps: int) -> List[str]:
    """Algorithm-specific hyperparameters used by sweep runners.

    We keep per-round local training budgets consistent with the DFL snapshot by
    preferring `*_local_steps` over full-epoch passes where supported. This
    prevents methods like Oblivionis from reprocessing all local data and
    becoming disproportionately slow during sweeps.
    """
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


def _run_jobs(jobs: List[Job], gpus: List[int], seed: int) -> Dict[Tuple[str, str, str], Dict[Union[int, float], float]]:
    results: Dict[Tuple[str, str, str], Dict[Union[int, float], float]] = {}
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
        # `--gpu` is the logical index within this visible set.
        env.setdefault("CUDA_VISIBLE_DEVICES", "2,3")

        cmd = job.cmd + ["--gpu", str(gpu_id)]
        job.log_path.parent.mkdir(parents=True, exist_ok=True)
        log_f = open(job.log_path, "w", encoding="utf-8")
        log_f.write(" ".join(cmd) + "\n\n")
        log_f.flush()

        proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=log_f, stderr=subprocess.STDOUT, env=env)
        proc._log_handle = log_f  # type: ignore[attr-defined]
        return proc

    while pending or running:
        while pending and free_gpus:
            job = pending.pop(0)
            gpu_id = free_gpus.pop(0)
            print(f"[LAUNCH][gpu={gpu_id}] {job.dataset} {job.algorithm} {job.sweep} x={job.x} -> {job.log_path}")
            proc = launch(job, gpu_id)
            running[proc] = (job, gpu_id)
            time.sleep(1.0)

        finished: List[subprocess.Popen] = []
        for proc, (job, gpu_id) in list(running.items()):
            ret = proc.poll()
            if ret is None:
                continue
            finished.append(proc)

            log_f = getattr(proc, "_log_handle", None)
            if log_f is not None:
                try:
                    log_f.close()
                except Exception:
                    pass

            if ret != 0:
                raise RuntimeError(f"Job failed (exit={ret}): {job.log_path}")

            dfu_dir = _latest_dfu_dir(job.output_prefix)
            history_path = dfu_dir / "history.json"
            macro_f1 = _extract_macro_f1(history_path)
            key = (job.dataset, job.algorithm, job.sweep)
            results.setdefault(key, {})[job.x] = macro_f1
            print(f"[DONE] {job.dataset} {job.algorithm} {job.sweep} x={job.x} macro_f1={macro_f1:.4f} ({dfu_dir})")

            free_gpus.append(gpu_id)

        for proc in finished:
            running.pop(proc, None)

        if running:
            time.sleep(10.0)

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", type=str, default="dfu_sweeps_figures_20251231_014459")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target_agent", type=int, default=0)
    parser.add_argument("--max_eval_samples", type=int, default=100)
    parser.add_argument(
        "--gpus",
        type=str,
        default="0,1",
        help="Logical GPU ids within CUDA_VISIBLE_DEVICES (default: 0,1 for physical 2,3).",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=",".join(DATASETS),
        help="Comma-separated datasets to run (default: all).",
    )
    parser.add_argument(
        "--algorithms",
        type=str,
        default=",".join(ALGORITHMS),
        help="Comma-separated DFU algorithms to run (default: all).",
    )
    parser.add_argument(
        "--sweeps",
        type=str,
        default="agent_count,lora_ratio",
        help="Comma-separated sweeps to run: agent_count,lora_ratio (default: both).",
    )
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument(
        "--save_lora_states",
        action="store_true",
        help="Save final LoRA state .pt files for each sweep run. Default is off to keep full paper sweeps disk-safe.",
    )
    parser.add_argument(
        "--snapshot_20news",
        type=str,
        default="checkpoints/20newsgroups/K10/G10_L5/alpha0.5/seed42_20251220_074624",
    )
    parser.add_argument(
        "--snapshot_yahoo",
        type=str,
        default="checkpoints/yahoo_subset/K10/G10_L5/alpha0.5/seed42_20251223_081958",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root)
    logs_dir = output_root / "logs_all_methods_sweeps"

    gpus = [int(x) for x in args.gpus.split(",") if x.strip() != ""]
    if not gpus:
        raise SystemExit("--gpus is empty")

    selected_datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    selected_algorithms = [x.strip() for x in args.algorithms.split(",") if x.strip()]
    selected_sweeps = [x.strip() for x in args.sweeps.split(",") if x.strip()]

    for ds in selected_datasets:
        if ds not in DATASETS:
            raise SystemExit(f"Unknown dataset in --datasets: {ds}")
    for algo in selected_algorithms:
        if algo not in ALGORITHMS:
            raise SystemExit(f"Unknown algorithm in --algorithms: {algo}")
    for sw in selected_sweeps:
        if sw not in {"agent_count", "lora_ratio"}:
            raise SystemExit(f"Unknown sweep in --sweeps: {sw}")

    snapshot_20 = Path(args.snapshot_20news)
    snapshot_yahoo = Path(args.snapshot_yahoo)

    caches_dir = ROOT / "cache" / "sensitivities"
    cache_20 = caches_dir / f"20newsgroups_{snapshot_20.name}_agent{args.target_agent}_abs.json"
    cache_yahoo = caches_dir / f"yahoo_subset_{snapshot_yahoo.name}_agent{args.target_agent}_abs.json"
    _ensure_sensitivity_cache(snapshot_20, args.target_agent, cache_20)
    _ensure_sensitivity_cache(snapshot_yahoo, args.target_agent, cache_yahoo)

    ratios = [round(i / 10, 1) for i in range(1, 11)]
    counts = list(range(1, 10))

    jobs: List[Job] = []
    skipped: Dict[Tuple[str, str, str], Dict[Union[int, float], float]] = {}

    dataset_snaps = {
        "20newsgroups": (snapshot_20, cache_20),
        "yahoo_subset": (snapshot_yahoo, cache_yahoo),
    }

    for dataset in selected_datasets:
        snap, sens_cache = dataset_snaps[dataset]
        dfl_cfg = _load_json(snap / "config.json")
        local_steps = int(dfl_cfg.get("local_steps") or 0)
        for algo in selected_algorithms:
            if "agent_count" in selected_sweeps:
                # Agent-count sweep (ours, full params)
                for k in counts:
                    strategy_dir = _strategy_dir(
                        selection_strategy="ours",
                        selection_ratio=None,
                        selection_count=k,
                        enable_param_selection=False,
                        param_selection_ratio=1.0,
                        param_epsilon_W=0.0,
                        param_random_selection=False,
                        param_selection_mode="epsilon",
                    )
                    prefix = _output_prefix(
                        output_root=output_root,
                        dfl_snapshot=snap,
                        dfu_algorithm=algo,
                        strategy_dir=strategy_dir,
                    )
                    if args.skip_existing:
                        existing = list(prefix.glob("dfu_*/history.json"))
                        if existing:
                            latest = max(existing, key=lambda p: p.stat().st_mtime)
                            macro_f1 = _extract_macro_f1(latest)
                            skipped.setdefault((dataset, algo, "agent_count"), {})[k] = macro_f1
                            continue

                    cmd = [
                        sys.executable,
                        str(ROOT / "scripts" / "run_dfu.py"),
                        "--dfl_snapshot",
                        str(snap),
                        "--dfu_algorithm",
                        algo,
                        "--output_dir",
                        str(output_root),
                        "--target_agent",
                        str(args.target_agent),
                        "--seed",
                        str(args.seed),
                        "--max_eval_samples",
                        str(args.max_eval_samples),
                    ]
                    if not args.save_lora_states:
                        cmd += ["--no_save_lora_states"]
                    cmd += ["--eval_every", "0"]
                    cmd += _algo_hparams(algorithm=algo, local_steps=local_steps)
                    cmd += [
                        "--selection_strategy",
                        "ours",
                        "--selection_count",
                        str(k),
                    ]
                    jobs.append(
                        Job(
                            dataset=dataset,
                            algorithm=algo,
                            sweep="agent_count",
                            x=k,
                            cmd=cmd,
                            output_prefix=prefix,
                            log_path=logs_dir / dataset / algo / "agent_count" / f"k{k}.log",
                        )
                    )

            if "lora_ratio" in selected_sweeps:
                # LoRA-ratio sweep (full agents, top_ratio by sensitivity)
                for r in ratios:
                    strategy_dir = _strategy_dir(
                        selection_strategy="full",
                        selection_ratio=None,
                        selection_count=None,
                        enable_param_selection=True,
                        param_selection_ratio=r,
                        param_epsilon_W=0.0,
                        param_random_selection=False,
                        param_selection_mode="top_ratio",
                    )
                    prefix = _output_prefix(
                        output_root=output_root,
                        dfl_snapshot=snap,
                        dfu_algorithm=algo,
                        strategy_dir=strategy_dir,
                    )
                    if args.skip_existing:
                        existing = list(prefix.glob("dfu_*/history.json"))
                        if existing:
                            latest = max(existing, key=lambda p: p.stat().st_mtime)
                            macro_f1 = _extract_macro_f1(latest)
                            skipped.setdefault((dataset, algo, "lora_ratio"), {})[r] = macro_f1
                            continue

                    cmd = [
                        sys.executable,
                        str(ROOT / "scripts" / "run_dfu.py"),
                        "--dfl_snapshot",
                        str(snap),
                        "--dfu_algorithm",
                        algo,
                        "--output_dir",
                        str(output_root),
                        "--target_agent",
                        str(args.target_agent),
                        "--seed",
                        str(args.seed),
                        "--max_eval_samples",
                        str(args.max_eval_samples),
                    ]
                    if not args.save_lora_states:
                        cmd += ["--no_save_lora_states"]
                    cmd += ["--eval_every", "0"]
                    cmd += _algo_hparams(algorithm=algo, local_steps=local_steps)
                    cmd += [
                        "--selection_strategy",
                        "full",
                        "--enable_param_selection",
                        "--param_selection_mode",
                        "top_ratio",
                        "--param_selection_ratio",
                        str(r),
                        "--param_epsilon_W",
                        "0.0",
                        "--param_sensitivity_cache",
                        str(sens_cache),
                    ]
                    jobs.append(
                        Job(
                            dataset=dataset,
                            algorithm=algo,
                            sweep="lora_ratio",
                            x=r,
                            cmd=cmd,
                            output_prefix=prefix,
                            log_path=logs_dir / dataset / algo / "lora_ratio" / f"r{r}.log",
                        )
                    )

    # Run missing jobs, then merge skipped points.
    run_results = _run_jobs(jobs, gpus=gpus, seed=args.seed) if jobs else {}
    for key, series in skipped.items():
        run_results.setdefault(key, {}).update(series)

    out_json = output_root / "all_methods_sweeps_selected_eval.json"
    out_json.write_text(
        json.dumps({f"{k[0]}::{k[1]}::{k[2]}": v for k, v in run_results.items()}, indent=2),
        encoding="utf-8",
    )
    print(f"[OK] wrote {out_json}")
    print("Tip: run `python scripts/plot_all_methods_sweeps.py --sweep_root ...` to generate figures.")


if __name__ == "__main__":
    main()
