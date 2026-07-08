# TDB DSU Joint Grid Comparison

本表记录 `k=1..9` 与 `r=0.1..1.0` 联合网格中的 clean F1 最高配置。论文最终配置见 `reports/tdb_clean_final_local_f1_mia_20260602.md`，最终配置还同时考虑节点数、LoRA 更新比例和后门直接遗忘审计。

- Cells: 8
- DSU > Base: 8/8
- DSU > AS and DSU > LS with complete seeds: 7/8

| Dataset | Method | Base | AS best | LS best | DSU joint best | DSU-Base | DSU-AS | DSU-LS | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 20newsgroups | d-federaser | 0.3829 | k=4 0.4683+/-0.0234 | r=0.4 0.4764+/-0.0130 | k=4,r=0.8 0.5032+/-0.0403 (n=3) | 0.1203 | 0.0349 | 0.0268 | PASS |
| 20newsgroups | d-fedosd | 0.3737 | k=7 0.4896+/-0.0431 | r=0.1 0.6025+/-0.0240 | k=8,r=0.1 0.6021+/-0.0283 (n=3) | 0.2284 | 0.1125 | -0.0004 | FAIL_DSU_LE_LS |
| 20newsgroups | d-fedrecovery | 0.4145 | k=5 0.5172+/-0.0051 | r=0.2 0.6100+/-0.0173 | k=8,r=0.1 0.6148+/-0.0192 (n=3) | 0.2002 | 0.0975 | 0.0048 | PASS |
| 20newsgroups | d-oblivionis | 0.4146 | k=7 0.5159+/-0.0239 | r=0.1 0.6071+/-0.0063 | k=8,r=0.1 0.6189+/-0.0119 (n=3) | 0.2043 | 0.1029 | 0.0118 | PASS |
| yahoo_subset | d-federaser | 0.6389 | k=9 0.6873+/-0.0057 | r=0.9 0.6870+/-0.0050 | k=9,r=0.6 0.6904+/-0.0154 (n=3) | 0.0516 | 0.0031 | 0.0034 | PASS |
| yahoo_subset | d-fedosd | 0.5947 | k=7 0.6884+/-0.0038 | r=0.1 0.7327+/-0.0108 | k=5,r=0.2 0.7368+/-0.0270 (n=3) | 0.1421 | 0.0484 | 0.0041 | PASS |
| yahoo_subset | d-fedrecovery | 0.6240 | k=4 0.7133+/-0.0112 | r=0.1 0.7369+/-0.0075 | k=8,r=0.1 0.7515+/-0.0196 (n=3) | 0.1275 | 0.0381 | 0.0145 | PASS |
| yahoo_subset | d-oblivionis | 0.6195 | k=6 0.6979+/-0.0137 | r=0.1 0.7315+/-0.0072 | k=4,r=0.2 0.7406+/-0.0183 (n=3) | 0.1211 | 0.0428 | 0.0091 | PASS |
