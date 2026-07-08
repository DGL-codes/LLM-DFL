#!/usr/bin/env python3
"""Run clean TDB DFU matrix and paper-grade MIA detector with nonmember=val.

This is intentionally separate from the backdoor pipeline: it reuses the clean
DFL snapshots, reruns DFU while keeping LoRA states, and then runs
eval_unlearning_detectors.py on each Base/AS/LS/DSU cell.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.run_tdb_fair_grid import SNAPSHOTS, algo_args, csv_list, fmt_ratio  # noqa: E402


STRATEGIES = ["full_all", "full_ours", "ours_all", "ours_ours"]


def read_best(meta_csv: Path) -> Dict[Tuple[str, str], Tuple[int, float, int, float]]:
    out: Dict[Tuple[str, str], Tuple[int, float, int, float]] = {}
    with meta_csv.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            dataset = str(row["dataset"])
            algo = str(row["method"])
            as_k = int(float(row.get("as_best_k") or 5))
            ls_r = float(row.get("ls_best_r") or 0.5)
            dsu_k = int(float(row.get("dsu_best_k") or as_k))
            dsu_r = float(row.get("dsu_best_r") or ls_r)
            out[(dataset, algo)] = (as_k, ls_r, dsu_k, dsu_r)
    return out


@dataclass
class SeedJob:
    dataset: str
    seed: int


def run_cmd(cmd: List[str], *, env: Dict[str, str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        f.write(" ".join(cmd) + "\n\n")
        f.flush()
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=f, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed({proc.returncode}): {log_path}")


def find_retrain_dir(retrain_out: Path, dataset: str, snapshot: Path) -> Optional[Path]:
    root = retrain_out / dataset / "K10" / "G10_L5" / "alpha0.5" / "strategy_retrain" / snapshot.name
    candidates = sorted(root.glob("retrain_*"))
    candidates = [p for p in candidates if (p / "round_10").is_dir()]
    return candidates[-1] if candidates else None


def ensure_retrain(
    *,
    dataset: str,
    seed: int,
    snapshot: Path,
    retrain_out: Path,
    max_eval_samples: int,
    env: Dict[str, str],
    log_dir: Path,
) -> Path:
    found = find_retrain_dir(retrain_out, dataset, snapshot)
    if found is not None:
        return found
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_retrain.py"),
        "--dfl_checkpoint",
        str(snapshot),
        "--output_dir",
        str(retrain_out),
        "--target_agent",
        "0",
        "--eval_every",
        "0",
        "--max_eval_samples",
        str(max_eval_samples),
        "--batch_size",
        "2",
        "--grad_accum_steps",
        "4",
        "--gpu",
        "0",
    ]
    run_cmd(cmd, env=env, log_path=log_dir / f"retrain_{dataset}_seed{seed}.log")
    found = find_retrain_dir(retrain_out, dataset, snapshot)
    if found is None:
        raise RuntimeError(f"retrain output not found for {dataset} seed={seed}")
    return found


def strategy_args(
    *,
    strategy: str,
    dataset: str,
    algo: str,
    best: Dict[Tuple[str, str], Tuple[int, float, int, float]],
    sens_cache: Path,
    args: argparse.Namespace,
) -> List[str]:
    as_k, ls_r, dsu_k, dsu_r = best.get((dataset, algo), (5, 0.5, 5, 0.5))
    out: List[str] = []
    if strategy == "full_all":
        out += ["--selection_strategy", "full"]
    elif strategy == "ours_all":
        out += [
            "--selection_strategy",
            "tdb",
            "--selection_count",
            str(as_k),
            "--tdb_sketch_dim",
            str(args.tdb_sketch_dim),
            "--tdb_max_intervals",
            str(args.tdb_max_intervals),
            "--tdb_time_limit",
            str(args.tdb_time_limit),
            "--tdb_alpha_u",
            str(args.tdb_alpha_u),
            "--tdb_alpha_p",
            str(args.tdb_alpha_p),
            "--tdb_alpha_q",
            str(args.tdb_alpha_q),
            "--tdb_tau_q",
            str(args.tdb_tau_q),
            "--tdb_aggregation_scope",
            str(args.tdb_aggregation_scope),
        ]
    elif strategy == "full_ours":
        out += [
            "--selection_strategy",
            "full",
            "--enable_param_selection",
            "--param_selection_mode",
            "top_ratio",
            "--param_selection_ratio",
            fmt_ratio(ls_r),
            "--param_sensitivity_cache",
            str(sens_cache),
        ]
    elif strategy == "ours_ours":
        out += [
            "--selection_strategy",
            "tdb",
            "--selection_count",
            str(dsu_k),
            "--tdb_sketch_dim",
            str(args.tdb_sketch_dim),
            "--tdb_max_intervals",
            str(args.tdb_max_intervals),
            "--tdb_time_limit",
            str(args.tdb_time_limit),
            "--tdb_alpha_u",
            str(args.tdb_alpha_u),
            "--tdb_alpha_p",
            str(args.tdb_alpha_p),
            "--tdb_alpha_q",
            str(args.tdb_alpha_q),
            "--tdb_tau_q",
            str(args.tdb_tau_q),
            "--tdb_aggregation_scope",
            str(args.tdb_aggregation_scope),
            "--enable_param_selection",
            "--param_selection_mode",
            "top_ratio",
            "--param_selection_ratio",
            fmt_ratio(dsu_r),
            "--param_sensitivity_cache",
            str(sens_cache),
        ]
    else:
        raise ValueError(strategy)
    return out


def find_latest_dfu(dfu_out: Path, dataset: str, algo: str, snapshot: Path) -> Optional[Path]:
    root = dfu_out / dataset / algo
    candidates = sorted(root.glob(f"**/{snapshot.name}/dfu_*"))
    candidates = [p for p in candidates if (p / "history.json").exists() and (p / "dfu_config.json").exists()]
    return candidates[-1] if candidates else None


def run_dfu_and_mia(
    *,
    dataset: str,
    seed: int,
    algo: str,
    strategy: str,
    snapshot: Path,
    retrain_dir: Path,
    dfu_out: Path,
    artifact_root: Path,
    sens_cache: Path,
    best: Dict[Tuple[str, str], Tuple[int, float, int, float]],
    env: Dict[str, str],
    log_dir: Path,
    args: argparse.Namespace,
) -> None:
    tag = f"mia_grid_{dataset}_seed{seed}_{algo}_{strategy}_clean_tdb_nonmemberVAL"
    mia_out = artifact_root / "unlearning_audit" / "mia" / tag / "mia_audit.json"
    if mia_out.exists():
        print(f"[skip] existing MIA audit: {tag}", flush=True)
        return

    common = [
        sys.executable,
        str(ROOT / "scripts" / "run_dfu.py"),
        "--dfl_snapshot",
        str(snapshot),
        "--dfu_algorithm",
        algo,
        "--output_dir",
        str(dfu_out),
        "--target_agent",
        "0",
        "--seed",
        str(seed),
        "--batch_size",
        str(args.batch_size),
        "--grad_accum_steps",
        str(args.grad_accum_steps),
        "--lr",
        str(args.lr),
        "--eval_every",
        "0",
        "--max_eval_samples",
        str(args.max_eval_samples),
        "--mia_nonmember_source",
        "val",
        "--gpu",
        "0",
    ]
    common += algo_args(algo, "paper")
    if algo == "d-federaser":
        common += ["--calibration_steps", str(args.federaser_calibration_steps), "--calibration_interval", "2"]
    common += strategy_args(
        strategy=strategy,
        dataset=dataset,
        algo=algo,
        best=best,
        sens_cache=sens_cache,
        args=args,
    )
    run_cmd(common, env=env, log_path=log_dir / f"dfu_{dataset}_seed{seed}_{algo}_{strategy}.log")
    dfu_dir = find_latest_dfu(dfu_out, dataset, algo, snapshot)
    if dfu_dir is None:
        raise RuntimeError(f"DFU output missing: {dataset} seed={seed} {algo} {strategy}")

    mia_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "eval_unlearning_detectors.py"),
        "--dfl_snapshot",
        str(snapshot),
        "--dfu_dir",
        str(dfu_dir),
        "--retrain_dir",
        str(retrain_dir),
        "--target_agent",
        "0",
        "--eval_agent_id",
        "1",
        "--max_samples",
        str(args.audit_max_samples),
        "--batch_size",
        "8",
        "--nonmember_source",
        "val",
        "--gpu",
        "0",
        "--tag",
        tag,
    ]
    mia_env = dict(env)
    mia_env["LLMDFL_ARTIFACT_ROOT"] = str(artifact_root)
    run_cmd(mia_cmd, env=mia_env, log_path=log_dir / f"mia_{dataset}_seed{seed}_{algo}_{strategy}.log")
    if args.cleanup_dfu_states:
        for p in dfu_dir.rglob("lora_state.pt"):
            p.unlink()


def worker(job_queue: List[SeedJob], gpu: str, args: argparse.Namespace, best: Dict[Tuple[str, str], Tuple[int, float, int, float]]) -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["LLMDFL_ALLOWED_PHYSICAL_GPUS"] = str(gpu)
    env.setdefault("LLMDFL_LOCAL_FILES_ONLY", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("PYTHONUNBUFFERED", "1")

    dfu_out = Path(args.dfu_out)
    retrain_out = Path(args.retrain_out)
    artifact_root = Path(args.artifact_root)
    sens_root = Path(args.sens_cache_root)
    log_dir = Path(args.log_dir)

    for job in job_queue:
        snapshot_rel = SNAPSHOTS[(job.dataset, job.seed)]
        snapshot = ROOT / snapshot_rel
        sens_cache = sens_root / f"{job.dataset}_{snapshot.name}_agent0.json"
        if not sens_cache.exists():
            raise FileNotFoundError(f"sensitivity cache missing: {sens_cache}")
        retrain_dir = ensure_retrain(
            dataset=job.dataset,
            seed=job.seed,
            snapshot=snapshot,
            retrain_out=retrain_out,
            max_eval_samples=int(args.max_eval_samples),
            env=env,
            log_dir=log_dir,
        )
        for algo in csv_list(args.algorithms):
            for strategy in csv_list(args.strategies):
                run_dfu_and_mia(
                    dataset=job.dataset,
                    seed=job.seed,
                    algo=algo,
                    strategy=strategy,
                    snapshot=snapshot,
                    retrain_dir=retrain_dir,
                    dfu_out=dfu_out,
                    artifact_root=artifact_root,
                    sens_cache=sens_cache,
                    best=best,
                    env=env,
                    log_dir=log_dir,
                    args=args,
                )


def split_jobs(jobs: List[SeedJob], n: int) -> List[List[SeedJob]]:
    out = [[] for _ in range(n)]
    for idx, job in enumerate(jobs):
        out[idx % n].append(job)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="20newsgroups,yahoo_subset")
    ap.add_argument("--algorithms", default="d-federaser,d-fedosd,d-fedrecovery,d-oblivionis")
    ap.add_argument("--strategies", default="full_all,full_ours,ours_all,ours_ours")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--physical_gpus", default="2,3")
    ap.add_argument("--meta_csv", default="reports/tdb_final_full_sweep_best_fixed_mean_seed424344_20260526.csv")
    ap.add_argument("--dfu_out", default="dfu_checkpoints_clean_tdb_mia_20260526")
    ap.add_argument("--retrain_out", default="retrain_checkpoints_clean_tdb_mia_20260526")
    ap.add_argument("--artifact_root", default="artifacts")
    ap.add_argument("--sens_cache_root", default="artifacts/tdb_as_ls_k1to9_r0p1to1_seed424344_20260526/sens_cache")
    ap.add_argument("--log_dir", default="logs/clean_tdb_mia_20260526")
    ap.add_argument("--max_eval_samples", type=int, default=100)
    ap.add_argument("--audit_max_samples", type=int, default=400)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum_steps", type=int, default=2)
    ap.add_argument("--lr", default="1e-3")
    ap.add_argument("--federaser_calibration_steps", type=int, default=5)
    ap.add_argument("--tdb_sketch_dim", type=int, default=16)
    ap.add_argument("--tdb_max_intervals", type=int, default=2)
    ap.add_argument("--tdb_time_limit", type=float, default=20.0)
    ap.add_argument("--tdb_alpha_u", type=float, default=1.0)
    ap.add_argument("--tdb_alpha_p", type=float, default=1.0)
    ap.add_argument("--tdb_alpha_q", type=float, default=0.1)
    ap.add_argument("--tdb_tau_q", type=float, default=0.0)
    ap.add_argument("--tdb_aggregation_scope", choices=["local", "global"], default="local")
    ap.add_argument("--cleanup_dfu_states", action="store_true")
    args = ap.parse_args()

    best = read_best(ROOT / args.meta_csv)
    datasets = csv_list(args.datasets)
    seeds = [int(x) for x in csv_list(args.seeds)]
    jobs = [SeedJob(dataset=ds, seed=seed) for ds in datasets for seed in seeds]
    gpus = csv_list(args.physical_gpus)
    chunks = split_jobs(jobs, len(gpus))

    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    procs: List[subprocess.Popen] = []
    for gpu, chunk in zip(gpus, chunks):
        if not chunk:
            continue
        payload = "\n".join(f"{j.dataset},{j.seed}" for j in chunk)
        env = os.environ.copy()
        env["CLEAN_MIA_JOBS"] = payload
        cmd = [
            sys.executable,
            "-c",
            (
                "import os, sys; "
                "from pathlib import Path; "
                "sys.path.insert(0, str(Path('.').resolve())); "
                "from scripts.run_clean_tdb_mia_grid import SeedJob, worker, read_best; "
                "import argparse; "
                f"args=argparse.Namespace(**{vars(args)!r}); "
                f"best=read_best(Path({str(ROOT / args.meta_csv)!r})); "
                "jobs=[SeedJob(a,int(b)) for a,b in (line.split(',') for line in os.environ['CLEAN_MIA_JOBS'].splitlines() if line)]; "
                f"worker(jobs, {gpu!r}, args, best)"
            ),
        ]
        log_path = Path(args.log_dir) / f"worker_gpu{gpu}.log"
        f = log_path.open("w", encoding="utf-8")
        f.write(f"jobs={payload}\n\n")
        f.flush()
        procs.append(subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=f, stderr=subprocess.STDOUT))
        print(f"[launch] gpu={gpu} jobs={payload.replace(chr(10), ';')} log={log_path}", flush=True)

    failures = []
    while procs:
        for p in list(procs):
            code = p.poll()
            if code is None:
                continue
            procs.remove(p)
            if code != 0:
                failures.append(code)
                print(f"[failed] worker code={code}", flush=True)
            else:
                print("[done] worker", flush=True)
        time.sleep(5)
    if failures:
        raise SystemExit(f"clean MIA workers failed: {failures}")
    print("Clean TDB MIA grid completed.", flush=True)


if __name__ == "__main__":
    main()
