# Reproduction diff report (LLM part)

This compares locally aggregated results (from existing `history.json`) to the paper tables.

## Local sources
- Base seed42 root: `dfu_sweeps_20news_lora_ratio_4methods_20260101_002412`
- Base multi-seed root: `dfu_ms_full_sweeps_boxplots_20260101_064248`
- DSU bestcfg root: `dfu_ablation_as_ls_bestcfg_seed42_20260101_130036`
- Seed-aligned backfill root: `实验结果/运行产物/full_repro_20260522_full_v2/dfu_seed_aligned_llm_strict_424344`
- DSU meta json: `artifacts/ablation_as_ls_bestcfg_424344.json`
- Seeds: [42, 43, 44]
- Require seed-aligned snapshot: `True`
- MIA protocol: `val`
- MIA metric: `auc`
- MIA override csv: `/home/xzq/private/llm-dfl-0525/实验结果/运行产物/full_repro_20260522_full_v2/artifacts/llm_mia_override.csv`
- Run override csv: `/home/xzq/private/llm-dfl-0525/实验结果/运行产物/full_repro_20260522_full_v2/artifacts/llm_run_override.csv`
- Selected runs csv: `/home/xzq/private/llm-dfl-0525/实验结果/运行产物/full_repro_20260522_full_v2/artifacts/llm_selected_used.csv`

## Diagnostics

- This repo historically used different notions of "seed" (DFL training seed vs DFU/unlearning seed). We attempt to infer the DFL snapshot used per run from sibling `dfu_config.json`.
- If all rows for a method/dataset point to the SAME `dfl_snapshot`, then the reported variance is NOT from different DFL partitions/training; this often explains mismatches in the paper's reported std.
- Note: MIA numbers in old `history.json` may have been computed on forget-vs-retain (both train) rather than member-vs-nonmember. Use `scripts/eval_unlearning_detectors.py` for the intended audit.

## 20newsgroups

| Method | DSU | Local F1 | Paper F1 | Δmean | Δstd | Local MIA | Paper MIA | Δmean | Δstd | n |
|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FedEraser | × | 39.09±0.96 | 40.03±1.70 | -0.94 | -0.74 | 54.04±4.21 | 48.36±1.26 | +5.68 | +2.95 | 3 |
| FedEraser | ✓ | 38.44±2.36 | 44.94±0.63 | -6.50 | +1.73 | 51.75±2.36 | 48.80±1.98 | +2.95 | +0.38 | 3 |
| FedOSD | × | 47.83±2.59 | 47.98±6.63 | -0.15 | -4.04 | 55.17±2.54 | 45.13±2.88 | +10.04 | -0.34 | 3 |
| FedOSD | ✓ | 60.77±2.06 | 58.48±1.00 | +2.29 | +1.06 | 53.49±2.25 | 45.73±2.89 | +7.76 | -0.64 | 3 |
| FedRecovery | × | 48.71±1.86 | 46.18±6.39 | +2.53 | -4.53 | 49.82±1.42 | 47.34±1.60 | +2.48 | -0.18 | 3 |
| FedRecovery | ✓ | 61.15±0.23 | 61.23±3.61 | -0.08 | -3.38 | 55.36±1.70 | 49.47±3.83 | +5.89 | -2.13 | 3 |
| Oblivionis | × | 47.02±0.58 | 46.37±1.53 | +0.65 | -0.95 | 52.33±2.12 | 43.61±3.17 | +8.72 | -1.05 | 3 |
| Oblivionis | ✓ | 58.92±1.30 | 62.01±2.47 | -3.09 | -1.17 | 55.09±2.43 | 43.58±1.49 | +11.51 | +0.94 | 3 |

### Single-seed view (seed=42)

| Method | DSU | Local F1 (seed) | Local MIA (seed) | dfl_snapshot | dfu_seed | history.json |
|---|:---:|---:|---:|---|---:|---|
| FedEraser | × | 39.10 | 57.51 | `/home/xzq/private/llm-dfl-0525/checkpoints/20newsgroups/K10/G10_L5/alpha0.5/seed42_20251220_074624` | 42 | `/home/xzq/private/llm-dfl-0525/实验结果/运行产物/full_repro_20260522_full_v2/dfu_seed_aligned_llm_strict_424344/20newsgroups/d-federaser/strategy_full_lora1.0_topratio_ours/K10/G10_L5/alpha0.5/seed42_20251220_074624/dfu_20260522_112735/history.json` |
| FedEraser | ✓ | 41.16 | 54.20 | `/home/xzq/private/llm-dfl-0525/checkpoints/20newsgroups/K10/G10_L5/alpha0.5/seed42_20251220_074624` | 42 | `/home/xzq/private/llm-dfl-0525/实验结果/运行产物/full_repro_20260522_full_v2/dfu_seed_aligned_llm_strict_424344/20newsgroups/d-federaser/strategy_ours_count6_lora0.4_topratio_ours/K10/G10_L5/alpha0.5/seed42_20251220_074624/dfu_20260522_113214/history.json` |
| FedOSD | × | 47.98 | 57.10 | `/home/xzq/private/llm-dfl-0525/checkpoints/20newsgroups/K10/G10_L5/alpha0.5/seed42_20251220_074624` | 42 | `/home/xzq/private/llm-dfl-0525/实验结果/运行产物/full_repro_20260522_full_v2/dfu_seed_aligned_llm_strict_424344/20newsgroups/d-fedosd/strategy_full_lora1.0_topratio_ours/K10/G10_L5/alpha0.5/seed42_20251220_074624/dfu_20260522_112735/history.json` |
| FedOSD | ✓ | 58.64 | 56.08 | `/home/xzq/private/llm-dfl-0525/checkpoints/20newsgroups/K10/G10_L5/alpha0.5/seed42_20251220_074624` | 42 | `/home/xzq/private/llm-dfl-0525/实验结果/运行产物/full_repro_20260522_full_v2/dfu_seed_aligned_llm_strict_424344/20newsgroups/d-fedosd/strategy_ours_count7_lora0.2_topratio_ours/K10/G10_L5/alpha0.5/seed42_20251220_074624/dfu_20260522_113154/history.json` |
| FedRecovery | × | 46.62 | 51.33 | `/home/xzq/private/llm-dfl-0525/checkpoints/20newsgroups/K10/G10_L5/alpha0.5/seed42_20251220_074624` | 42 | `/home/xzq/private/llm-dfl-0525/实验结果/运行产物/full_repro_20260522_full_v2/dfu_seed_aligned_llm_strict_424344/20newsgroups/d-fedrecovery/strategy_full_lora1.0_topratio_ours/K10/G10_L5/alpha0.5/seed42_20251220_074624/dfu_20260522_112735/history.json` |
| FedRecovery | ✓ | 61.23 | 55.12 | `/home/xzq/private/llm-dfl-0525/checkpoints/20newsgroups/K10/G10_L5/alpha0.5/seed42_20251220_074624` | 42 | `/home/xzq/private/llm-dfl-0525/实验结果/运行产物/full_repro_20260522_full_v2/dfu_seed_aligned_llm_strict_424344/20newsgroups/d-fedrecovery/strategy_ours_count8_lora0.1_topratio_ours/K10/G10_L5/alpha0.5/seed42_20251220_074624/dfu_20260522_113223/history.json` |
| Oblivionis | × | 46.37 | 52.92 | `/home/xzq/private/llm-dfl-0525/checkpoints/20newsgroups/K10/G10_L5/alpha0.5/seed42_20251220_074624` | 42 | `/home/xzq/private/llm-dfl-0525/实验结果/运行产物/full_repro_20260522_full_v2/dfu_seed_aligned_llm_strict_424344/20newsgroups/d-oblivionis/strategy_full_lora1.0_topratio_ours/K10/G10_L5/alpha0.5/seed42_20251220_074624/dfu_20260522_112736/history.json` |
| Oblivionis | ✓ | 60.18 | 56.00 | `/home/xzq/private/llm-dfl-0525/checkpoints/20newsgroups/K10/G10_L5/alpha0.5/seed42_20251220_074624` | 42 | `/home/xzq/private/llm-dfl-0525/实验结果/运行产物/full_repro_20260522_full_v2/dfu_seed_aligned_llm_strict_424344/20newsgroups/d-oblivionis/strategy_ours_count4_lora0.2_topratio_ours/K10/G10_L5/alpha0.5/seed42_20251220_074624/dfu_20260522_113215/history.json` |

## yahoo_subset

| Method | DSU | Local F1 | Paper F1 | Δmean | Δstd | Local MIA | Paper MIA | Δmean | Δstd | n |
|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FedEraser | × | 67.56±1.87 | 67.49±3.47 | +0.07 | -1.60 | 50.52±6.40 | 49.00±1.15 | +1.52 | +5.25 | 3 |
| FedEraser | ✓ | 67.18±0.98 | 69.33±0.90 | -2.15 | +0.08 | 49.88±5.87 | 47.74±0.65 | +2.14 | +5.22 | 3 |
| FedOSD | × | 67.02±4.10 | 60.97±16.98 | +6.05 | -12.88 | 51.75±6.35 | 41.28±1.80 | +10.47 | +4.55 | 3 |
| FedOSD | ✓ | 72.25±2.41 | 75.34±1.05 | -3.09 | +1.36 | 56.31±7.83 | 45.18±0.54 | +11.13 | +7.29 | 3 |
| FedRecovery | × | 69.84±0.80 | 70.82±4.44 | -0.98 | -3.64 | 52.32±6.39 | 50.51±2.98 | +1.81 | +3.41 | 3 |
| FedRecovery | ✓ | 73.98±1.13 | 74.28±3.17 | -0.30 | -2.04 | 54.27±7.41 | 53.11±2.61 | +1.16 | +4.80 | 3 |
| Oblivionis | × | 70.29±0.56 | 67.82±5.10 | +2.47 | -4.54 | 54.58±4.24 | 44.20±2.21 | +10.38 | +2.03 | 3 |
| Oblivionis | ✓ | 72.68±1.90 | 73.64±5.57 | -0.96 | -3.67 | 54.57±8.54 | 48.90±1.39 | +5.67 | +7.15 | 3 |

### Single-seed view (seed=42)

| Method | DSU | Local F1 (seed) | Local MIA (seed) | dfl_snapshot | dfu_seed | history.json |
|---|:---:|---:|---:|---|---:|---|
| FedEraser | × | 69.55 | 55.05 | `/home/xzq/private/llm-dfl-0525/checkpoints/yahoo_subset/K10/G10_L5/alpha0.5/seed42_20260304_164802` | 42 | `/home/xzq/private/llm-dfl-0525/实验结果/运行产物/full_repro_20260522_full_v2/dfu_seed_aligned_llm_strict_424344/yahoo_subset/d-federaser/strategy_full_lora1.0_topratio_ours/K10/G10_L5/alpha0.5/seed42_20260304_164802/dfu_20260522_115034/history.json` |
| FedEraser | ✓ | 66.77 | 52.91 | `/home/xzq/private/llm-dfl-0525/checkpoints/yahoo_subset/K10/G10_L5/alpha0.5/seed42_20260304_164802` | 42 | `/home/xzq/private/llm-dfl-0525/实验结果/运行产物/full_repro_20260522_full_v2/dfu_seed_aligned_llm_strict_424344/yahoo_subset/d-federaser/strategy_ours_count6_lora0.5_topratio_ours/K10/G10_L5/alpha0.5/seed42_20260304_164802/dfu_20260522_115346/history.json` |
| FedOSD | × | 62.29 | 57.78 | `/home/xzq/private/llm-dfl-0525/checkpoints/yahoo_subset/K10/G10_L5/alpha0.5/seed42_20260304_164802` | 42 | `/home/xzq/private/llm-dfl-0525/实验结果/运行产物/full_repro_20260522_full_v2/dfu_seed_aligned_llm_strict_424344/yahoo_subset/d-fedosd/strategy_full_lora1.0_topratio_ours/K10/G10_L5/alpha0.5/seed42_20260304_164802/dfu_20260522_115105/history.json` |
| FedOSD | ✓ | 72.61 | 56.88 | `/home/xzq/private/llm-dfl-0525/checkpoints/yahoo_subset/K10/G10_L5/alpha0.5/seed42_20260304_164802` | 42 | `/home/xzq/private/llm-dfl-0525/实验结果/运行产物/full_repro_20260522_full_v2/dfu_seed_aligned_llm_strict_424344/yahoo_subset/d-fedosd/strategy_ours_count4_lora0.2_topratio_ours/K10/G10_L5/alpha0.5/seed42_20260304_164802/dfu_20260522_115341/history.json` |
| FedRecovery | × | 69.75 | 50.40 | `/home/xzq/private/llm-dfl-0525/checkpoints/yahoo_subset/K10/G10_L5/alpha0.5/seed42_20260304_164802` | 42 | `/home/xzq/private/llm-dfl-0525/实验结果/运行产物/full_repro_20260522_full_v2/dfu_seed_aligned_llm_strict_424344/yahoo_subset/d-fedrecovery/strategy_full_lora1.0_topratio_ours/K10/G10_L5/alpha0.5/seed42_20260304_164802/dfu_20260522_115439/history.json` |
| FedRecovery | ✓ | 74.25 | 55.40 | `/home/xzq/private/llm-dfl-0525/checkpoints/yahoo_subset/K10/G10_L5/alpha0.5/seed42_20260304_164802` | 42 | `/home/xzq/private/llm-dfl-0525/实验结果/运行产物/full_repro_20260522_full_v2/dfu_seed_aligned_llm_strict_424344/yahoo_subset/d-fedrecovery/strategy_ours_count8_lora0.1_topratio_ours/K10/G10_L5/alpha0.5/seed42_20260304_164802/dfu_20260522_115730/history.json` |
| Oblivionis | × | 70.50 | 56.90 | `/home/xzq/private/llm-dfl-0525/checkpoints/yahoo_subset/K10/G10_L5/alpha0.5/seed42_20260304_164802` | 42 | `/home/xzq/private/llm-dfl-0525/实验结果/运行产物/full_repro_20260522_full_v2/dfu_seed_aligned_llm_strict_424344/yahoo_subset/d-oblivionis/strategy_full_lora1.0_topratio_ours/K10/G10_L5/alpha0.5/seed42_20260304_164802/dfu_20260522_114826/history.json` |
| Oblivionis | ✓ | 74.61 | 57.53 | `/home/xzq/private/llm-dfl-0525/checkpoints/yahoo_subset/K10/G10_L5/alpha0.5/seed42_20260304_164802` | 42 | `/home/xzq/private/llm-dfl-0525/实验结果/运行产物/full_repro_20260522_full_v2/dfu_seed_aligned_llm_strict_424344/yahoo_subset/d-oblivionis/strategy_ours_count6_lora0.1_topratio_ours/K10/G10_L5/alpha0.5/seed42_20260304_164802/dfu_20260522_115112/history.json` |

