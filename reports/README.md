# Reports Included in This Release

This directory contains the curated result files used for reproducing the reported experiments and validation evidence.

The repository intentionally does not include raw checkpoints, model weights, or run logs. The CSV and Markdown files here are the compact, auditable result layer. Re-running the fixed entrypoints can regenerate the corresponding tables and plots when the required local model/data caches are available.

## Clean DFL/DFU Results

- `tdb_clean_final_local_f1_mia_20260602.md`
- `tdb_clean_final_local_f1_mia_20260602.csv`
- `table_I_20news.csv`
- `table_II_yahoo.csv`
- `repro_diff_report.md`

## AS/LS/DSU Hyperparameter Sweeps

- `tdb_as_ls_k1to9_r0p1to1_seed424344_20260527_all_rows.csv`
- `tdb_as_ls_k1to9_r0p1to1_seed424344_20260527_aggregate.csv`
- `tdb_as_ls_k1to9_r0p1to1_seed424344_20260527_best.csv`
- `tdb_dsu_joint_k1to9_r0p1to1_seed424344_20260530_final_local_all_rows.csv`
- `tdb_dsu_joint_k1to9_r0p1to1_seed424344_20260530_final_local_aggregate.csv`
- `tdb_dsu_joint_k1to9_r0p1to1_seed424344_20260530_final_local_best.csv`
- `tdb_final_local_as_k_f1_20260603.png`
- `tdb_final_local_ls_r_f1_20260603.png`
- `tdb_final_local_dsu_kr_heatmaps_f1_20260603.png`

## Direct Forgetting and Reviewer Supplements

- `backdoor_forgetting_final_seed42_20260603.md`
- `backdoor_forgetting_final_seed42_20260603.csv`
- `sequential_cumulative_tdb_dsu_20260603.md`
- `sequential_cumulative_tdb_dsu_20260603.csv`
- `sequential_yahoo_step3_k_sensitivity_20260603.md`
- `tdb_proxy_validation_correlation_20260601.md`
- `tdb_proxy_validation_correlation_20260601.csv`
- `tdb_solver_stats_20260602.csv`
- `final_integrity_audit_20260603.md`
- `final_reasonableness_audit_20260603.md`
