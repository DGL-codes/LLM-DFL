#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Optional, Tuple


ROOT = Path(__file__).resolve().parent.parent


def _find_snapshot(dataset: str, seed: int) -> Optional[Path]:
    base = ROOT / "checkpoints" / dataset / "K10" / "G10_L5" / "alpha0.5"
    cands = sorted(base.glob(f"seed{seed}_*"))
    cands = [p for p in cands if (p / "config.json").exists() and (p / "round_10").exists()]
    return cands[-1] if cands else None


def _run_eval(snapshot: Path, seed: int, dataset: str, gpu: int, max_samples: int) -> Path:
    tag = f"mia_dflonly_{dataset}_seed{seed}_nonmemberVAL"
    cmd = [
        "python",
        "scripts/eval_unlearning_detectors.py",
        "--dfl_snapshot",
        str(snapshot),
        "--target_agent",
        "0",
        "--eval_agent_id",
        "1",
        "--max_samples",
        str(max_samples),
        "--batch_size",
        "8",
        "--nonmember_source",
        "val",
        "--gpu",
        str(gpu),
        "--tag",
        tag,
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)
    out = ROOT / "artifacts" / "unlearning_audit" / "mia" / tag / "mia_audit.json"
    if not out.exists():
        raise FileNotFoundError(out)
    return out


def _safe_float(x: object) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _fmt_ms(xs: List[float]) -> str:
    if not xs:
        return "-"
    if len(xs) == 1:
        return f"{xs[0]:.4f}±0.0000"
    return f"{mean(xs):.4f}±{stdev(xs):.4f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", type=str, default="20newsgroups,yahoo_subset")
    ap.add_argument("--seeds", type=str, default="42,43,44")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--max_samples", type=int, default=200)
    ap.add_argument("--out_csv", type=str, default="reports/mia_privacy_recomputed.csv")
    ap.add_argument("--out_md", type=str, default="reports/mia_privacy_recomputed.md")
    args = ap.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    rows: List[Dict[str, object]] = []
    for ds in datasets:
        for seed in seeds:
            snap = _find_snapshot(ds, seed)
            if snap is None:
                rows.append(
                    {
                        "dataset": ds,
                        "seed": seed,
                        "snapshot": "",
                        "status": "missing_snapshot",
                        "loss_auc_sym": None,
                        "loss_adv": None,
                        "mink_auc_sym": None,
                        "mink_adv": None,
                    }
                )
                continue

            out = _run_eval(snapshot=snap, seed=seed, dataset=ds, gpu=int(args.gpu), max_samples=int(args.max_samples))
            data = json.loads(out.read_text(encoding="utf-8"))

            loss = data.get("detectors", {}).get("dfl", {}).get("methods", {}).get("loss", {}).get("result", {})
            mink = data.get("detectors", {}).get("dfl", {}).get("methods", {}).get("min_k", {}).get("result", {})

            rows.append(
                {
                    "dataset": ds,
                    "seed": seed,
                    "snapshot": str(snap),
                    "status": "ok",
                    "loss_auc_sym": _safe_float(loss.get("auc_sym")),
                    "loss_adv": _safe_float(loss.get("adv")),
                    "mink_auc_sym": _safe_float(mink.get("auc_sym")),
                    "mink_adv": _safe_float(mink.get("adv")),
                }
            )

    out_csv = ROOT / args.out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    csv_lines = [
        "dataset,seed,status,loss_auc_sym,loss_adv,mink_auc_sym,mink_adv,snapshot"
    ]
    for r in rows:
        csv_lines.append(
            ",".join(
                [
                    str(r["dataset"]),
                    str(r["seed"]),
                    str(r["status"]),
                    "" if r["loss_auc_sym"] is None else f"{float(r['loss_auc_sym']):.6f}",
                    "" if r["loss_adv"] is None else f"{float(r['loss_adv']):.6f}",
                    "" if r["mink_auc_sym"] is None else f"{float(r['mink_auc_sym']):.6f}",
                    "" if r["mink_adv"] is None else f"{float(r['mink_adv']):.6f}",
                    str(r["snapshot"]),
                ]
            )
        )
    out_csv.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    grouped: Dict[Tuple[str, str], List[float]] = {}
    for r in rows:
        if r["status"] != "ok":
            continue
        for k in ["loss_adv", "mink_adv"]:
            v = r[k]
            if v is None:
                continue
            grouped.setdefault((str(r["dataset"]), k), []).append(float(v))

    out_md = ROOT / args.out_md
    md: List[str] = []
    md.append("# Recomputed MIA Privacy (DFL-only, nonmember=val)")
    md.append("")
    md.append(f"- Datasets: `{datasets}`")
    md.append(f"- Seeds: `{seeds}`")
    md.append(f"- Max samples per side: `{args.max_samples}`")
    md.append("")
    md.append("## Per-seed")
    md.append("")
    md.append("| Dataset | Seed | Status | loss_auc_sym | loss_adv | min_k_auc_sym | min_k_adv |")
    md.append("|---|---:|---|---:|---:|---:|---:|")
    for r in rows:
        def _v(x: object) -> str:
            if x is None:
                return "-"
            try:
                return f"{float(x):.4f}"
            except Exception:
                return "-"

        md.append(
            "| "
            + " | ".join(
                [
                    str(r["dataset"]),
                    str(r["seed"]),
                    str(r["status"]),
                    _v(r["loss_auc_sym"]),
                    _v(r["loss_adv"]),
                    _v(r["mink_auc_sym"]),
                    _v(r["mink_adv"]),
                ]
            )
            + " |"
        )

    md.append("")
    md.append("## Mean±Std (adv)")
    md.append("")
    md.append("| Dataset | loss_adv | min_k_adv |")
    md.append("|---|---:|---:|")
    for ds in datasets:
        la = grouped.get((ds, "loss_adv"), [])
        ma = grouped.get((ds, "mink_adv"), [])
        md.append(f"| {ds} | {_fmt_ms(la)} | {_fmt_ms(ma)} |")

    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_md}")


if __name__ == "__main__":
    main()
