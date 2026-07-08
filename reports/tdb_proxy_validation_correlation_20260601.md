# TDB Proxy Validation: Label/Trajectory Discrepancy vs F1

Purpose: provide empirical support for the reviewer question about using lightweight sketches as proxies. This uses the existing clean DSU joint sweep, not a new training run.

Source: `reports/tdb_dsu_joint_k1to9_r0p1to1_seed424344_20260530_final_local_aggregate.csv`; 720 aggregate DSU configurations across 2 datasets, 4 methods, k=1..9, r=0.1..1.0, seeds 42/43/44.

Overall Spearman correlation: F1 vs label L1 = -0.402; F1 vs trajectory L1 = -0.270. Negative is expected: lower discrepancy should correspond to higher F1.

| Dataset | Method | n | rho(F1,label L1) | rho(F1,trajectory L1) | Low-label-L1 F1 | High-label-L1 F1 | Gap |
|---|---|---:|---:|---:|---:|---:|---:|
| 20newsgroups | d-federaser | 90 | -0.556 | -0.556 | 0.433 | 0.321 | 0.113 |
| 20newsgroups | d-fedosd | 90 | -0.578 | -0.578 | 0.525 | 0.367 | 0.158 |
| 20newsgroups | d-fedrecovery | 90 | -0.650 | -0.650 | 0.532 | 0.366 | 0.166 |
| 20newsgroups | d-oblivionis | 90 | -0.517 | -0.517 | 0.541 | 0.402 | 0.140 |
| yahoo_subset | d-federaser | 90 | -0.672 | -0.672 | 0.672 | 0.597 | 0.076 |
| yahoo_subset | d-fedosd | 90 | -0.506 | -0.506 | 0.703 | 0.551 | 0.153 |
| yahoo_subset | d-fedrecovery | 90 | -0.485 | -0.485 | 0.718 | 0.608 | 0.110 |
| yahoo_subset | d-oblivionis | 90 | -0.436 | -0.436 | 0.712 | 0.619 | 0.094 |

Interpretation: across every dataset/method cell, lower label/trajectory discrepancy is associated with higher retained F1 in the DSU sweep. This does not prove the class-conditional consistency assumption, but it gives empirical support that the sketches are meaningful selection proxies in these experiments.
