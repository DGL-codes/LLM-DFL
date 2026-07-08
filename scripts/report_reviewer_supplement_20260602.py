#!/usr/bin/env python3
"""Aggregate reviewer-response evidence for the TDB/DSU revision.

Reporting only. The script does not run DFU, does not change selection, and
does not use audit metrics as an optimization signal.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DATASETS = ["20newsgroups", "yahoo_subset"]
METHODS = ["d-federaser", "d-fedosd", "d-fedrecovery", "d-oblivionis"]
METHOD_LABEL = {
    "d-federaser": "FedEraser",
    "d-fedosd": "FedOSD",
    "d-fedrecovery": "FedRecovery",
    "d-oblivionis": "D-Oblivionis",
}
DATASET_LABEL = {"20newsgroups": "20News", "yahoo_subset": "Yahoo"}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fnum(value: Any) -> float | None:
    try:
        if value in ("", None):
            return None
        return float(value)
    except Exception:
        return None


def pct(value: Any, digits: int = 2) -> str:
    x = fnum(value)
    return "-" if x is None else f"{100.0 * x:.{digits}f}"


def pct_mean(values: list[float | None], digits: int = 2) -> str:
    arr = [v for v in values if v is not None]
    if not arr:
        return "-"
    if len(arr) == 1:
        return f"{100.0 * arr[0]:.{digits}f}±0.00"
    return f"{100.0 * mean(arr):.{digits}f}±{100.0 * pstdev(arr):.{digits}f}"


def raw_mean(values: list[float | None], digits: int = 3) -> str:
    arr = [v for v in values if v is not None]
    if not arr:
        return "-"
    if len(arr) == 1:
        return f"{arr[0]:.{digits}f}±0.000"
    return f"{mean(arr):.{digits}f}±{pstdev(arr):.{digits}f}"


def order_key(dataset: str, method: str) -> tuple[int, int]:
    return (
        DATASETS.index(dataset) if dataset in DATASETS else 99,
        METHODS.index(method) if method in METHODS else 99,
    )


def clean_mia_adv_table(clean_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in clean_rows:
        item: dict[str, Any] = {"dataset": row["dataset"], "method": row["method"]}
        for setting in ["base", "as", "ls", "dsu"]:
            mia = fnum(row.get(f"{setting}_mia"))
            item[f"{setting}_mia_adv"] = None if mia is None else abs(mia - 0.5)
            item[f"{setting}_f1"] = fnum(row.get(f"{setting}_f1"))
        out.append(item)
    return out


def backdoor_pairs(rows: list[dict[str, str]]) -> dict[tuple[str, str, int], dict[str, dict[str, str]]]:
    buckets: dict[tuple[str, str, int], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        try:
            key = (row["dataset"], row["method"], int(row["seed"]))
        except Exception:
            continue
        buckets[key][row["setting"]] = row
    return buckets


def summarize_backdoor(rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[tuple[str, str, int, str]]]:
    buckets = backdoor_pairs(rows)
    summary: list[dict[str, Any]] = []
    missing: list[tuple[str, str, int, str]] = []
    for dataset in DATASETS:
        for method in METHODS:
            items: list[dict[str, dict[str, str]]] = []
            for seed in [42, 43, 44]:
                pair = buckets.get((dataset, method, seed), {})
                if "Base" not in pair:
                    missing.append((dataset, method, seed, "Base"))
                if "DSU" not in pair:
                    missing.append((dataset, method, seed, "DSU"))
                if "Base" in pair and "DSU" in pair:
                    items.append(pair)
            base_dfl = [fnum(x["Base"].get("dfl_asr")) for x in items]
            base_dfu = [fnum(x["Base"].get("dfu_asr")) for x in items]
            dsu_dfl = [fnum(x["DSU"].get("dfl_asr")) for x in items]
            dsu_dfu = [fnum(x["DSU"].get("dfu_asr")) for x in items]
            base_drop = [
                None if a is None or b is None else a - b
                for a, b in zip(base_dfl, base_dfu)
            ]
            dsu_drop = [
                None if a is None or b is None else a - b
                for a, b in zip(dsu_dfl, dsu_dfu)
            ]
            dsu_minus_base = [
                None if a is None or b is None else a - b
                for a, b in zip(dsu_dfu, base_dfu)
            ]
            summary.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "n_complete": len(items),
                    "base_dfl_asr": base_dfl,
                    "base_dfu_asr": base_dfu,
                    "dsu_dfl_asr": dsu_dfl,
                    "dsu_dfu_asr": dsu_dfu,
                    "base_drop": base_drop,
                    "dsu_drop": dsu_drop,
                    "dsu_minus_base": dsu_minus_base,
                }
            )
    return summary, missing


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def final_metric(history: dict[str, Any], key: str) -> float | None:
    stats = history.get("final_stats") or {}
    val = fnum(stats.get(key))
    if val is not None:
        return val
    avg_metrics = history.get("avg_metrics") or []
    if avg_metrics:
        avg_key = {
            "macro_f1_mean": "macro_f1",
            "accuracy_mean": "accuracy",
            "precision_mean": "precision",
            "recall_mean": "recall",
        }.get(key, key)
        val = fnum(avg_metrics[-1].get(avg_key))
        if val is not None:
            return val
    metrics = history.get("unlearning_metrics") or []
    if metrics:
        return fnum(metrics[-1].get(key))
    return None


def mean_numeric(values: list[float | None]) -> float | None:
    arr = [v for v in values if v is not None]
    return mean(arr) if arr else None


def raw_mean_short(values: list[float | None], digits: int = 3) -> str:
    arr = [v for v in values if v is not None]
    if not arr:
        return "-"
    return f"{mean(arr):.{digits}f}"


def pct_mean_short(values: list[float | None], digits: int = 2) -> str:
    arr = [v for v in values if v is not None]
    if not arr:
        return "-"
    return f"{100.0 * mean(arr):.{digits}f}"


def summarize_history_metrics(rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Read DFU histories referenced by the final backdoor audit.

    This is reporting-only. It does not rerun evaluation and it does not feed any
    audit metric back into agent/layer selection.
    """
    metric_buckets: dict[tuple[str, str, str], dict[str, list[float | None]]] = defaultdict(lambda: defaultdict(list))
    efficiency_buckets: dict[tuple[str, str, str], dict[str, list[float | None]]] = defaultdict(lambda: defaultdict(list))
    storage_seen: dict[Path, dict[str, Any]] = {}
    runtime_map = load_runtime_log_map()

    for row in rows:
        audit_path = Path(row.get("path") or "")
        audit = load_json(audit_path) if audit_path.exists() else {}
        dfu_dir = Path(audit.get("dfu_dir") or "")
        history = load_json(dfu_dir / "history.json") if dfu_dir else {}
        retrain_dir = Path(audit.get("retrain_dir") or "")
        retrain_history = load_json(retrain_dir / "history.json") if retrain_dir else {}

        dataset = row.get("dataset") or ""
        method = row.get("method") or ""
        setting = row.get("setting") or ""
        if not dataset or not method or not setting:
            continue
        key = (dataset, method, setting)

        for metric in [
            "forget_loss",
            "retain_loss",
            "loss_gap",
            "forget_accuracy",
            "retain_accuracy",
            "accuracy_gap",
            "mia_auc",
            "macro_f1_mean",
        ]:
            metric_buckets[key][metric].append(final_metric(history, metric))

        dfu_f1 = final_metric(history, "macro_f1_mean")
        retrain_f1 = final_metric(retrain_history, "macro_f1_mean")
        metric_buckets[key]["f1_gap_to_retrain"].append(
            None if dfu_f1 is None or retrain_f1 is None else abs(dfu_f1 - retrain_f1)
        )

        selected_count = fnum(row.get("dfu_n_agents"))
        if selected_count is None:
            selected_count = 9.0 if setting == "Base" else fnum(row.get("selection_count"))
        lora_ratio = 1.0 if setting == "Base" else fnum(row.get("param_selection_ratio"))
        if lora_ratio is None:
            lora_ratio = 1.0
        efficiency_buckets[key]["selected_agents"].append(selected_count)
        efficiency_buckets[key]["lora_ratio"].append(lora_ratio)
        efficiency_buckets[key]["updated_proxy"].append(
            None if selected_count is None else (selected_count / 9.0) * lora_ratio
        )
        runtime_key = str(dfu_dir.resolve()) if dfu_dir else ""
        efficiency_buckets[key]["wall_time_sec"].append(runtime_map.get(runtime_key))

        dfl_snapshot = Path(audit.get("dfl_snapshot") or "")
        if dfl_snapshot and dfl_snapshot.exists() and dfl_snapshot not in storage_seen:
            files = list(dfl_snapshot.glob("round_*/agent_*/lora_state*.pt"))
            sizes = [p.stat().st_size for p in files if p.is_file()]
            rounds = sorted({p.parts[-3] for p in files if len(p.parts) >= 3})
            storage_seen[dfl_snapshot] = {
                "dataset": dataset,
                "path": str(dfl_snapshot),
                "n_files": len(sizes),
                "n_rounds": len(rounds),
                "total_mb": sum(sizes) / (1024 * 1024) if sizes else None,
                "avg_file_mb": mean(sizes) / (1024 * 1024) if sizes else None,
            }

    metric_rows = []
    for (dataset, method, setting), vals in sorted(metric_buckets.items(), key=lambda kv: (*order_key(kv[0][0], kv[0][1]), kv[0][2])):
        metric_rows.append(
            {
                "dataset": dataset,
                "method": method,
                "setting": setting,
                "n": len([v for v in vals.get("loss_gap", []) if v is not None]),
                "forget_loss": vals.get("forget_loss", []),
                "retain_loss": vals.get("retain_loss", []),
                "loss_gap": vals.get("loss_gap", []),
                "forget_accuracy": vals.get("forget_accuracy", []),
                "retain_accuracy": vals.get("retain_accuracy", []),
                "mia_adv": [
                    None if v is None else abs(v - 0.5)
                    for v in vals.get("mia_auc", [])
                ],
                "f1_gap_to_retrain": vals.get("f1_gap_to_retrain", []),
            }
        )

    efficiency_rows = []
    for (dataset, method, setting), vals in sorted(efficiency_buckets.items(), key=lambda kv: (*order_key(kv[0][0], kv[0][1]), kv[0][2])):
        efficiency_rows.append(
            {
                "dataset": dataset,
                "method": method,
                "setting": setting,
                "n": len([v for v in vals.get("selected_agents", []) if v is not None]),
                "selected_agents": vals.get("selected_agents", []),
                "lora_ratio": vals.get("lora_ratio", []),
                "updated_proxy": vals.get("updated_proxy", []),
                "wall_time_sec": vals.get("wall_time_sec", []),
            }
        )

    storage_rows = sorted(storage_seen.values(), key=lambda r: (r["dataset"], r["path"]))
    return metric_rows, efficiency_rows, storage_rows


def load_runtime_log_map() -> dict[str, float]:
    """Map DFU output directories to wall-clock seconds from queue logs."""
    log_root = ROOT / "logs/backdoor_final_20260602"
    out: dict[str, float] = {}
    if not log_root.exists():
        return out
    ansi = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
    for path in log_root.glob("**/*.log"):
        current_dir: str | None = None
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            continue
        for raw_line in lines:
            line = ansi.sub("", raw_line).strip()
            marker = "DFU completed! Results saved to "
            if marker in line:
                current_dir = line.split(marker, 1)[1].strip()
                continue
            if current_dir and "总运行时间:" in line and "秒" in line:
                m = re.search(r"总运行时间:\s*([0-9.]+)秒", line)
                if not m:
                    continue
                dfu_path = Path(current_dir)
                if not dfu_path.is_absolute():
                    dfu_path = ROOT / dfu_path
                out[str(dfu_path.resolve())] = float(m.group(1))
                current_dir = None
    return out


def solver_stats(paths: list[Path]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for path in paths:
        for row in read_csv(path):
            setting = row.get("setting") or ""
            if setting not in {"AS", "DSU"}:
                continue
            dataset = row.get("dataset") or ""
            method = row.get("algorithm") or row.get("method") or ""
            buckets[(dataset, method, setting)].append(row)

    out: list[dict[str, Any]] = []
    for (dataset, method, setting), rows in sorted(buckets.items(), key=lambda kv: (*order_key(kv[0][0], kv[0][1]), kv[0][2])):
        success_vals = []
        for row in rows:
            raw = str(row.get("solver_success") or "").strip().lower()
            success_vals.append(raw in {"true", "1", "yes"})
        out.append(
            {
                "dataset": dataset,
                "method": method,
                "setting": setting,
                "n": len(rows),
                "solver_success_rate": sum(success_vals) / len(success_vals) if success_vals else None,
                "solve_time_sec": raw_mean([fnum(r.get("solve_time_sec")) for r in rows]),
                "trajectory_l1": raw_mean([fnum(r.get("trajectory_l1")) for r in rows]),
                "label_l1": raw_mean([fnum(r.get("label_l1")) for r in rows]),
                "target_exposure": raw_mean([fnum(r.get("target_exposure")) for r in rows]),
            }
        )
    return out


def write_solver_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "dataset",
        "method",
        "setting",
        "n",
        "solver_success_rate",
        "solve_time_sec",
        "trajectory_l1",
        "label_l1",
        "target_exposure",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def load_text(path: Path, max_lines: int = 40) -> list[str]:
    if not path.exists():
        return [f"`{path.relative_to(ROOT)}` not found."]
    return path.read_text(encoding="utf-8").splitlines()[:max_lines]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean_csv", default="reports/tdb_clean_final_local_f1_mia_20260602.csv")
    ap.add_argument("--backdoor_csv", default="reports/backdoor_localfix_final_audit_20260602.csv")
    ap.add_argument("--out_md", default="reports/reviewer_supplement_evidence_20260602.md")
    ap.add_argument("--solver_csv", default="reports/tdb_solver_stats_20260602.csv")
    args = ap.parse_args()

    clean_rows = read_csv(ROOT / args.clean_csv)
    backdoor_rows = read_csv(ROOT / args.backdoor_csv)
    mia_adv_rows = clean_mia_adv_table(clean_rows)
    backdoor_summary, missing = summarize_backdoor(backdoor_rows)
    history_metric_rows, efficiency_rows, storage_rows = summarize_history_metrics(backdoor_rows)
    solver_rows = solver_stats(
        [
            ROOT / "reports/tdb_as_ls_k1to9_r0p1to1_seed424344_20260527_all_rows.csv",
            ROOT / "reports/tdb_dsu_joint_k1to9_r0p1to1_seed424344_20260530_final_local_all_rows.csv",
        ]
    )
    solver_csv = ROOT / args.solver_csv
    write_solver_csv(solver_csv, solver_rows)

    lines: list[str] = []
    lines.append("# Reviewer Supplement Evidence")
    lines.append("")
    lines.append("本报告是审稿意见补实验的索引和聚合。它只读已有结果，不改变训练、遗忘、节点选择或层选择。")
    lines.append("")
    lines.append("## 1. Clean Main Experiment")
    lines.append("")
    lines.append("Clean 主实验指不带后门投毒的 20News/Yahoo 主复现实验，用公共测试集 F1 和 MIA AUC 比较 Base、AS、LS、DSU。")
    lines.append("")
    lines.append("- clean table: `reports/tdb_clean_final_local_f1_mia_20260602.md`")
    lines.append("- source: final local-ring DSU joint sweep; no global aggregation result is used.")
    lines.append("")
    lines.append("## 2. Direct Forgetting Metric: Backdoor ASR")
    lines.append("")
    lines.append("后门审计使用 agent0 实际被投毒抽中的训练样本；主指标是 `ASR_non_target` 和 trigger lift。ASR 只用于离线审计，不进入训练或选择。")
    lines.append("")
    lines.append(f"- compact backdoor CSV: `reports/backdoor_localfix_final_audit_20260602.csv`")
    lines.append(f"- per-agent backdoor CSV: `reports/backdoor_localfix_final_per_agent_20260602.csv`")
    lines.append(f"- complete dataset-method-seed pairs: {sum(1 for r in backdoor_summary for _ in range(r['n_complete']))}/24")
    if missing:
        sample = "; ".join(f"{d}/{m}/seed{s}/{setting}" for d, m, s, setting in missing[:16])
        suffix = " ..." if len(missing) > 16 else ""
        lines.append(f"- currently missing audit cells: {len(missing)} ({sample}{suffix})")
    else:
        lines.append("- currently missing audit cells: 0")
    lines.append("")
    lines.append("| Dataset | Method | n | Base DFL ASR | Base DFU ASR | Base ASR drop | DSU DFL ASR | DSU DFU ASR | DSU ASR drop | DSU-Base DFU ASR |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in backdoor_summary:
        lines.append(
            "| "
            f"{DATASET_LABEL.get(row['dataset'], row['dataset'])} | "
            f"{METHOD_LABEL.get(row['method'], row['method'])} | "
            f"{row['n_complete']} | "
            f"{pct_mean(row['base_dfl_asr'])} | {pct_mean(row['base_dfu_asr'])} | {pct_mean(row['base_drop'])} | "
            f"{pct_mean(row['dsu_dfl_asr'])} | {pct_mean(row['dsu_dfu_asr'])} | {pct_mean(row['dsu_drop'])} | "
            f"{pct_mean(row['dsu_minus_base'])} |"
        )

    lines.append("")
    lines.append("## 2b. Forget/Retain Metrics From DFU Histories")
    lines.append("")
    lines.append("这些是 DFU `history.json` 里已有的传统遗忘验证指标。它们只作为辅助证据；本文后门审计的直接遗忘指标仍是上面的 ASR 和 trigger lift。")
    lines.append("")
    lines.append("| Dataset | Method | Setting | n | Forget Loss | Retain Loss | Loss Gap | Forget Acc | Retain Acc | MIA Advantage | F1 Gap to Retrain |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in history_metric_rows:
        lines.append(
            "| "
            f"{DATASET_LABEL.get(row['dataset'], row['dataset'])} | "
            f"{METHOD_LABEL.get(row['method'], row['method'])} | "
            f"{row['setting']} | {row['n']} | "
            f"{raw_mean_short(row['forget_loss'])} | {raw_mean_short(row['retain_loss'])} | "
            f"{raw_mean_short(row['loss_gap'])} | {pct_mean_short(row['forget_accuracy'])} | "
            f"{pct_mean_short(row['retain_accuracy'])} | {pct_mean_short(row['mia_adv'])} | "
            f"{pct_mean_short(row['f1_gap_to_retrain'])} |"
        )

    lines.append("")
    lines.append("## 3. MIA Advantage")
    lines.append("")
    lines.append("这里的 MIA advantage 是 `|MIA AUC - 50%|`，越接近 0 越接近随机猜测。它是辅助隐私/遗忘指标，不是 F1 主目标。")
    lines.append("")
    lines.append("| Dataset | Method | Base | AS | LS | DSU |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for row in sorted(mia_adv_rows, key=lambda r: order_key(r["dataset"], r["method"])):
        lines.append(
            "| "
            f"{DATASET_LABEL.get(row['dataset'], row['dataset'])} | "
            f"{METHOD_LABEL.get(row['method'], row['method'])} | "
            f"{pct(row.get('base_mia_adv'))} | {pct(row.get('as_mia_adv'))} | "
            f"{pct(row.get('ls_mia_adv'))} | {pct(row.get('dsu_mia_adv'))} |"
        )

    lines.append("")
    lines.append("## 4. Efficiency and Storage Proxies")
    lines.append("")
    lines.append("这里统计的是已落盘配置中的真实参与节点数和 LoRA 更新比例，并给出 `参与节点比例 × LoRA 更新比例` 作为通信/更新参数量的相对 proxy。Base 视为 9 个保留节点、全 LoRA 模块更新，proxy=1。")
    lines.append("")
    lines.append("| Dataset | Method | Setting | n | Selected Agents | LoRA Ratio | Relative Update/Communication Proxy | Wall Time Sec |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|")
    for row in efficiency_rows:
        lines.append(
            "| "
            f"{DATASET_LABEL.get(row['dataset'], row['dataset'])} | "
            f"{METHOD_LABEL.get(row['method'], row['method'])} | "
            f"{row['setting']} | {row['n']} | "
            f"{raw_mean_short(row['selected_agents'], digits=2)} | "
            f"{pct_mean_short(row['lora_ratio'])} | "
            f"{pct_mean_short(row['updated_proxy'])} | "
            f"{raw_mean_short(row['wall_time_sec'], digits=1)} |"
        )

    if storage_rows:
        lines.append("")
        lines.append("Snapshot storage is measured from the actual DFL checkpoint folders referenced by the audit JSON files.")
        lines.append("")
        lines.append("| Dataset | Snapshot Dir Counted | LoRA Files | Retained Rounds | Snapshot Storage MB | Avg LoRA File MB |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for row in storage_rows[:12]:
            lines.append(
                "| "
                f"{DATASET_LABEL.get(row['dataset'], row['dataset'])} | "
                f"`{Path(row['path']).name}` | "
                f"{row['n_files']} | {row['n_rounds']} | "
                f"{row['total_mb']:.2f} | {row['avg_file_mb']:.2f} |"
            )

    lines.append("")
    lines.append("## 5. Label Sketch Proxy Support")
    lines.append("")
    lines.extend(load_text(ROOT / "reports/tdb_proxy_validation_correlation_20260601.md", max_lines=28))

    lines.append("")
    lines.append("## 6. Sequential Unlearning Boundary")
    lines.append("")
    lines.extend(load_text(ROOT / "reports/sequential_cumulative_tdb_dsu_20260603.md", max_lines=44))

    lines.append("")
    lines.append("## 7. TDB/MILP Solver Statistics")
    lines.append("")
    lines.append(f"- solver stats CSV: `{solver_csv.relative_to(ROOT)}`")
    lines.append("")
    lines.append("| Dataset | Method | Setting | n | Success | Solve Time | Traj L1 | Label L1 | Exposure |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for row in solver_rows:
        lines.append(
            "| "
            f"{DATASET_LABEL.get(row['dataset'], row['dataset'])} | "
            f"{METHOD_LABEL.get(row['method'], row['method'])} | "
            f"{row['setting']} | {row['n']} | "
            f"{pct(row.get('solver_success_rate'))} | {row['solve_time_sec']} | "
            f"{row['trajectory_l1']} | {row['label_l1']} | {row['target_exposure']} |"
        )

    lines.append("")
    lines.append("## 8. Reviewer Text Items")
    lines.append("")
    lines.append("- Theory: define distribution discrepancy as label-sketch L1, not full distribution distance; bounded loss is an analytical finite-evaluation/clipped-loss assumption.")
    lines.append("- Label sketch: use the proxy correlation table as empirical support, with a discussion that class-conditional consistency is more plausible when agents share the same task taxonomy and preprocessing.")
    lines.append("- Module sensitivity cadence: specify that LS scores sum target-agent LoRA update energy over stored DFL snapshots; current runs use stored retained snapshots for LS and `tdb_max_intervals=2` for compact TDB trajectory sketches.")
    lines.append("- Sequential withdrawal: retained DFL history can be reused for later requests by replaying with the cumulative removed-agent set; TDB-AS and LS should be recomputed for each current target and remaining set. Directly chaining updates without replay may accumulate error, so long deletion sequences need periodic refresh/retraining.")

    out_md = ROOT / args.out_md
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_md)
    print(solver_csv)


if __name__ == "__main__":
    main()
