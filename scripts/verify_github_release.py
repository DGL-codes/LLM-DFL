#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent

REQUIRED = [
    "README.md",
    "requirements_dfu.txt",
    "environment_dfu.yml",
    "src/dfu/agent_selection.py",
    "src/dfu/trainer.py",
    "src/dfu/d_fedosd.py",
    "src/dfu/d_fedrecovery.py",
    "src/dfu/d_oblivionis.py",
    "src/dfu/lora_param_selection.py",
    "scripts/train_dfl.py",
    "scripts/run_dfu.py",
    "scripts/strict_repro_llm_pipeline.sh",
    "scripts/backdoor_audit_grid_pipeline.sh",
    "scripts/run_tdb_fair_grid.py",
    "scripts/report_final_tables_20260603.py",
    "scripts/report_tdb_clean_final_local_20260602.py",
    "scripts/report_sequential_cumulative_20260603.py",
    "reports/tdb_clean_final_local_f1_mia_20260602.md",
    "reports/tdb_clean_final_local_f1_mia_20260602.csv",
    "reports/tdb_as_ls_k1to9_r0p1to1_seed424344_20260527_all_rows.csv",
    "reports/tdb_dsu_joint_k1to9_r0p1to1_seed424344_20260530_final_local_all_rows.csv",
    "reports/backdoor_forgetting_final_seed42_20260603.md",
    "reports/backdoor_forgetting_final_seed42_20260603.csv",
    "reports/sequential_cumulative_tdb_dsu_20260603.md",
    "reports/sequential_cumulative_tdb_dsu_20260603.csv",
    "reports/unlearning_detector_validation.md",
    "reports/unlearning_detector_validation.csv",
]

FORBIDDEN_DIRS = [
    "models",
    "checkpoints",
    "checkpoints_backdoor_audit",
    "dfu_checkpoints_backdoor_audit",
    "dfu_checkpoints_clean_tdb_mia_20260526",
    "retrain_checkpoints_backdoor_audit",
    "retrain_checkpoints_clean_tdb_mia_20260526",
    "artifacts",
    "logs",
    "cache",
    ".codegraph",
    "实验结果",
    "review",
    "paper_docs",
]

FORBIDDEN_SUFFIXES = {
    ".pt",
    ".bin",
    ".safetensors",
    ".npy",
    ".pyc",
    ".log",
}


def main() -> int:
    missing = [p for p in REQUIRED if not (ROOT / p).exists()]
    forbidden_dirs = [p for p in FORBIDDEN_DIRS if (ROOT / p).exists()]
    forbidden_files = [
        str(p.relative_to(ROOT))
        for p in ROOT.rglob("*")
        if p.is_file() and p.suffix in FORBIDDEN_SUFFIXES
    ]

    print(f"[release] required files: {len(REQUIRED) - len(missing)} ok / {len(REQUIRED)} total")
    for p in missing:
        print(f"  MISS {p}")

    if forbidden_dirs:
        print("[release] forbidden directories present:")
        for p in forbidden_dirs:
            print(f"  FORBIDDEN {p}")

    if forbidden_files:
        print("[release] forbidden binary/cache files present:")
        for p in forbidden_files[:100]:
            print(f"  FORBIDDEN {p}")
        if len(forbidden_files) > 100:
            print(f"  ... {len(forbidden_files) - 100} more")

    if missing or forbidden_dirs or forbidden_files:
        print("[release] FAILED")
        return 1

    print("[release] PASS: GitHub release bundle looks clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
