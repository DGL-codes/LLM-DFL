# TDB-AS / LS Hyperparameter Sweep

- Per-run rows: `2160`
- Aggregate rows: `720`
- AS sweep: `k=1..9`.
- LS sweep: `r=0.1..1.0`.
- Metric reported here: `macro_f1_best` on the public test set, averaged across seeds.

## 20newsgroups

| Method | Best AS k | AS F1 | Best LS r | LS F1 |
|---|---:|---:|---:|---:|
| d-federaser | - | - | - | - |
| d-fedosd | - | - | - | - |
| d-fedrecovery | - | - | - | - |
| d-oblivionis | - | - | - | - |

### AS k Curve

| Method | k | F1 | MIA AUC | traj L1 | label L1 | target exposure |
|---|---:|---:|---:|---:|---:|---:|

### LS r Curve

| Method | r | F1 | MIA AUC |
|---|---:|---:|---:|

## yahoo_subset

| Method | Best AS k | AS F1 | Best LS r | LS F1 |
|---|---:|---:|---:|---:|
| d-federaser | - | - | - | - |
| d-fedosd | - | - | - | - |
| d-fedrecovery | - | - | - | - |
| d-oblivionis | - | - | - | - |

### AS k Curve

| Method | k | F1 | MIA AUC | traj L1 | label L1 | target exposure |
|---|---:|---:|---:|---:|---:|---:|

### LS r Curve

| Method | r | F1 | MIA AUC |
|---|---:|---:|---:|

