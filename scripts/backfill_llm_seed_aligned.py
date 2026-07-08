#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import shutil


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_ROOT = Path(os.environ.get("LLMDFL_EXPERIMENT_DIR", "实验结果/运行产物"))
sys.path.insert(0, str(ROOT))
from src.dfu.snapshot_loader import SnapshotLoader  # noqa: E402
from src.dfu.lora_param_selection import compute_module_sensitivities  # noqa: E402


DATASETS = ["20newsgroups", "yahoo_subset"]
ALGORITHMS = ["d-federaser", "d-fedosd", "d-fedrecovery", "d-oblivionis"]


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_ratio(x: float) -> str:
    s = f"{float(x):.6f}".rstrip("0").rstrip(".")
    return s if s else "0"


def _latest_snapshot(dataset: str, seed: int) -> Optional[Path]:
    base = ROOT / "checkpoints" / dataset / "K10" / "G10_L5" / "alpha0.5"
    snaps = sorted(base.glob(f"seed{seed}_*"))
    for snap in reversed(snaps):
        if (snap / "config.json").exists() and (snap / "round_10").exists():
            return snap
    return None


def _path_seed(path_str: str) -> Optional[int]:
    import re

    m = re.search(r"seed(\d+)_", str(path_str))
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _is_aligned_run(cfg_path: Path, expected_seed: int) -> bool:
    if not cfg_path.exists():
        return False
    try:
        cfg = _read_json(cfg_path)
    except Exception:
        return False
    try:
        dfu_seed = int(cfg.get("seed"))
    except Exception:
        return False
    snap = str(cfg.get("dfl_snapshot", ""))
    snap_seed = _path_seed(snap)
    return dfu_seed == int(expected_seed) and snap_seed == int(expected_seed)


def _has_aligned_history(
    run_base: Path,
    seed: int,
    expected_max_eval: int,
    *,
    allow_missing_max_eval: bool = False,
) -> Optional[Path]:
    if not run_base.exists():
        return None
    runs = sorted(run_base.glob("dfu_*"))
    for run_dir in reversed(runs):
        hp = run_dir / "history.json"
        cp = run_dir / "dfu_config.json"
        if not hp.exists() or not cp.exists():
            continue
        if not _is_aligned_run(cp, seed):
            continue
        try:
            cfg = _read_json(cp)
        except Exception:
            continue
        me = cfg.get("max_eval_samples", None)
        if me is None:
            if not allow_missing_max_eval:
                continue
        else:
            try:
                if int(me) != int(expected_max_eval):
                    continue
            except Exception:
                continue
        return run_dir
    return None


def _legacy_aligned_run(
    *,
    dataset: str,
    algo: str,
    cell: str,
    seed: int,
    k: int,
    ratio_str: str,
    expected_max_eval: int,
) -> Optional[Path]:
    roots: List[Path] = []
    if cell == "base":
        if seed == 42:
            roots = [ROOT / "dfu_sweeps_20news_lora_ratio_4methods_20260101_002412"]
        else:
            r0 = ROOT / "dfu_ms_full_sweeps_boxplots_20260101_064248" / f"seed{seed}"
            r1 = ROOT / "dfu_ms_20news_fedrecovery_lora_ratio_20260101_054210" / f"seed{seed}"
            roots = [r0]
            if dataset == "20newsgroups" and algo == "d-fedrecovery":
                roots.append(r1)
        strat_glob = "strategy_full_lora1.0_*"
        fallback_glob = "strategy_full"
    else:
        roots = [ROOT / "dfu_ablation_as_ls_bestcfg_seed42_20260101_130036" / f"seed{seed}"]
        strat_glob = f"strategy_ours_count{k}_lora{ratio_str}_*"
        fallback_glob = ""

    for root in roots:
        base = root / dataset / algo
        if not base.exists():
            continue
        histories = [p for p in base.glob(f"{strat_glob}/**/history.json") if p.is_file()]
        if fallback_glob:
            histories.extend([p for p in base.glob(f"{fallback_glob}/**/history.json") if p.is_file()])
        histories = sorted(set(histories), key=lambda p: p.stat().st_mtime, reverse=True)
        for hp in histories:
            cp = hp.parent / "dfu_config.json"
            if not _is_aligned_run(cp, seed):
                continue
            try:
                cfg = _read_json(cp)
            except Exception:
                continue
            me = cfg.get("max_eval_samples", None)
            if me is None:
                return hp.parent
            try:
                if int(me) == int(expected_max_eval):
                    return hp.parent
            except Exception:
                continue
    return None


def _algo_args(algo: str) -> List[str]:
    if algo == "d-federaser":
        return ["--calibration_steps", "3", "--calibration_interval", "2"]
    if algo == "d-fedosd":
        return [
            "--unlearn_rounds",
            "3",
            "--unlearn_lr",
            "1e-3",
            "--recovery_rounds",
            "2",
            "--recovery_local_steps",
            "5",
            "--recovery_lr",
            "1e-3",
            "--retain_grad_samples",
            "50",
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
            "5",
            "--recovery_lr",
            "1e-3",
        ]
    if algo == "d-oblivionis":
        return [
            "--unlearn_rounds",
            "1",
            "--unlearn_lr",
            "5e-4",
            "--propagation_rounds",
            "3",
            "--propagation_lr",
            "1e-3",
        ]
    raise ValueError(f"Unsupported algorithm: {algo}")


def _run(
    cmd: List[str],
    *,
    env: Dict[str, str],
    log_path: Path,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        f.write(" ".join(cmd) + "\n\n")
        f.flush()
        proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=f, stderr=subprocess.STDOUT)
        code = proc.wait()
    if code != 0:
        raise RuntimeError(f"Command failed({code}): {' '.join(cmd)}; log={log_path}")


def _ensure_sens_cache(snapshot: Path, out_path: Path, target_agent: int = 0) -> None:
    if out_path.exists():
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    loader = SnapshotLoader(str(snapshot))
    sens = compute_module_sensitivities(loader, target_agent=target_agent, verbose=True)
    out_path.write_text(json.dumps(sens, ensure_ascii=False, indent=2), encoding="utf-8")


def _find_existing_sens_cache(dataset: str, snapshot: Path, target_agent: int = 0) -> Optional[Path]:
    candidates = [
        ROOT / "cache" / "sensitivities" / f"{dataset}_{snapshot.name}_agent{target_agent}_abs.json",
        ROOT / "artifacts" / "unlearning_audit" / "sens_cache" / f"{dataset}_{snapshot.name}_agent{target_agent}.json",
        ROOT / "artifacts" / "seed_align_sens_cache" / f"{dataset}_{snapshot.name}_agent{target_agent}.json",
    ]
    for p in candidates:
        if not p.exists():
            continue
        try:
            obj = _read_json(p)
            if isinstance(obj, dict) and len(obj) > 0:
                return p
        except Exception:
            continue
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Backfill LLM DFU runs with strict DFL/DFU seed alignment.")
    ap.add_argument("--seeds", type=str, default="42,43,44")
    ap.add_argument("--datasets", type=str, default="20newsgroups,yahoo_subset")
    ap.add_argument("--algorithms", type=str, default="d-federaser,d-fedosd,d-fedrecovery,d-oblivionis")
    ap.add_argument("--physical_gpu", type=str, default="3")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out_root", type=str, default="")
    ap.add_argument("--max_eval_samples", type=int, default=100)
    ap.add_argument("--meta_json", type=str, default="artifacts/ablation_as_ls_bestcfg_424344.json")
    ap.add_argument("--summary_csv", type=str, default=str(DEFAULT_RESULTS_ROOT / "artifacts" / "seed_alignment_llm.csv"))
    ap.add_argument(
        "--allow_legacy_reuse",
        type=int,
        default=1,
        help="If 1, reuse older matching histories outside --out_root; set 0 for a clean fixed-entry rerun.",
    )
    args = ap.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    algorithms = [x.strip() for x in args.algorithms.split(",") if x.strip()]

    if not args.out_root:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_root = f"dfu_seed_aligned_llm_{ts}"

    out_root = ROOT / args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    meta = _read_json(ROOT / args.meta_json)
    best_k = {str(k): int(v) for k, v in (meta.get("best_k") or {}).items()}
    best_r = {str(k): float(v) for k, v in (meta.get("best_r") or {}).items()}

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.physical_gpu)
    env.setdefault("LLMDFL_ALLOWED_PHYSICAL_GPUS", str(args.physical_gpu))
    env.setdefault("LLMDFL_LOCAL_FILES_ONLY", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("PYTHONUNBUFFERED", "1")

    rows: List[Dict[str, str]] = []
    logs_dir = ROOT / "logs"

    for dataset in datasets:
        for algo in algorithms:
            key = f"{dataset}::{algo}"
            if key not in best_k or key not in best_r:
                raise KeyError(f"Missing best_k/best_r for {key} in {args.meta_json}")
            k = int(best_k[key])
            ratio = float(best_r[key])
            ratio_str = _fmt_ratio(ratio)

            for seed in seeds:
                dfl_snapshot = _latest_snapshot(dataset, seed)
                if dfl_snapshot is None:
                    rows.append(
                        {
                            "dataset": dataset,
                            "algorithm": algo,
                            "cell": "all",
                            "seed": str(seed),
                            "status": "missing_dfl_snapshot",
                            "dfl_snapshot": "",
                            "run_dir": "",
                            "reason": "no_complete_dfl_snapshot",
                        }
                    )
                    continue
                sens_cache = ROOT / "artifacts" / "seed_align_sens_cache" / f"{dataset}_{dfl_snapshot.name}_agent0.json"
                existing_cache = _find_existing_sens_cache(dataset, dfl_snapshot, target_agent=0)
                if existing_cache is not None and existing_cache != sens_cache:
                    sens_cache.parent.mkdir(parents=True, exist_ok=True)
                    if not sens_cache.exists():
                        shutil.copy2(existing_cache, sens_cache)
                _ensure_sens_cache(dfl_snapshot, sens_cache, target_agent=0)

                common = [
                    sys.executable,
                    str(ROOT / "scripts" / "run_dfu.py"),
                    "--dfl_snapshot",
                    str(dfl_snapshot),
                    "--dfu_algorithm",
                    algo,
                    "--output_dir",
                    str(out_root),
                    "--target_agent",
                    "0",
                    "--seed",
                    str(seed),
                    "--batch_size",
                    "4",
                    "--grad_accum_steps",
                    "2",
                    "--lr",
                    "1e-3",
                    "--eval_every",
                    "0",
                    "--max_eval_samples",
                    str(int(args.max_eval_samples)),
                    "--gpu",
                    str(int(args.gpu)),
                ]
                common += _algo_args(algo)

                # base
                base_run_base = (
                    out_root
                    / dataset
                    / algo
                    / "strategy_full_lora1.0_topratio_ours"
                    / "K10"
                    / "G10_L5"
                    / "alpha0.5"
                    / dfl_snapshot.name
                )
                base_done = _has_aligned_history(
                    base_run_base,
                    seed,
                    int(args.max_eval_samples),
                    allow_missing_max_eval=True,
                )
                if base_done is None and int(args.allow_legacy_reuse) == 1:
                    base_done = _legacy_aligned_run(
                        dataset=dataset,
                        algo=algo,
                        cell="base",
                        seed=seed,
                        k=k,
                        ratio_str=ratio_str,
                        expected_max_eval=int(args.max_eval_samples),
                    )
                if base_done is None:
                    cmd = common + [
                        "--selection_strategy",
                        "full",
                        "--enable_param_selection",
                        "--param_selection_mode",
                        "top_ratio",
                        "--param_selection_ratio",
                        "1.0",
                        "--param_sensitivity_cache",
                        str(sens_cache),
                    ]
                    log_path = logs_dir / f"seed_align_llm_{dataset}_seed{seed}_{algo}_base_gpu{args.physical_gpu}.log"
                    _run(cmd, env=env, log_path=log_path)
                    base_done = _has_aligned_history(
                        base_run_base,
                        seed,
                        int(args.max_eval_samples),
                        allow_missing_max_eval=True,
                    )
                if base_done is None:
                    base_done = _has_aligned_history(
                        base_run_base,
                        seed,
                        int(args.max_eval_samples),
                        allow_missing_max_eval=True,
                    )
                rows.append(
                    {
                        "dataset": dataset,
                        "algorithm": algo,
                        "cell": "base",
                        "seed": str(seed),
                        "status": "ok" if base_done else "failed",
                        "dfl_snapshot": str(dfl_snapshot),
                        "run_dir": str(base_done) if base_done else "",
                        "reason": "",
                    }
                )

                # dsu
                dsu_strat = f"strategy_ours_count{k}_lora{ratio_str}_topratio_ours"
                dsu_run_base = (
                    out_root
                    / dataset
                    / algo
                    / dsu_strat
                    / "K10"
                    / "G10_L5"
                    / "alpha0.5"
                    / dfl_snapshot.name
                )
                dsu_done = _has_aligned_history(
                    dsu_run_base,
                    seed,
                    int(args.max_eval_samples),
                    allow_missing_max_eval=True,
                )
                if dsu_done is None and int(args.allow_legacy_reuse) == 1:
                    dsu_done = _legacy_aligned_run(
                        dataset=dataset,
                        algo=algo,
                        cell="dsu",
                        seed=seed,
                        k=k,
                        ratio_str=ratio_str,
                        expected_max_eval=int(args.max_eval_samples),
                    )
                if dsu_done is None:
                    cmd = common + [
                        "--selection_strategy",
                        "ours",
                        "--selection_count",
                        str(int(k)),
                        "--selection_epsilon",
                        "0.1",
                        "--enable_param_selection",
                        "--param_selection_mode",
                        "top_ratio",
                        "--param_selection_ratio",
                        ratio_str,
                        "--param_sensitivity_cache",
                        str(sens_cache),
                    ]
                    log_path = logs_dir / f"seed_align_llm_{dataset}_seed{seed}_{algo}_dsu_gpu{args.physical_gpu}.log"
                    _run(cmd, env=env, log_path=log_path)
                    dsu_done = _has_aligned_history(
                        dsu_run_base,
                        seed,
                        int(args.max_eval_samples),
                        allow_missing_max_eval=True,
                    )
                if dsu_done is None:
                    dsu_done = _has_aligned_history(
                        dsu_run_base,
                        seed,
                        int(args.max_eval_samples),
                        allow_missing_max_eval=True,
                    )
                rows.append(
                    {
                        "dataset": dataset,
                        "algorithm": algo,
                        "cell": "dsu",
                        "seed": str(seed),
                        "status": "ok" if dsu_done else "failed",
                        "dfl_snapshot": str(dfl_snapshot),
                        "run_dir": str(dsu_done) if dsu_done else "",
                        "reason": "",
                    }
                )

    out_csv = ROOT / args.summary_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "algorithm", "cell", "seed", "status", "dfl_snapshot", "run_dir", "reason"]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    ok = sum(1 for r in rows if r["status"] == "ok")
    fail = sum(1 for r in rows if r["status"] != "ok")
    print(f"Wrote: {out_csv}")
    print(f"Rows: {len(rows)} ok={ok} fail={fail}")


if __name__ == "__main__":
    main()
