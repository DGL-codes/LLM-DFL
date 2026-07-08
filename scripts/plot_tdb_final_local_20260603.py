#!/usr/bin/env python3
"""Plot final local-ring selection figures from report CSV files."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"

DATASETS = ["20newsgroups", "yahoo_subset"]
METHODS = ["d-federaser", "d-fedosd", "d-fedrecovery", "d-oblivionis"]
DATASET_LABEL = {"20newsgroups": "20News", "yahoo_subset": "Yahoo"}
METHOD_LABEL = {
    "d-federaser": "FedEraser",
    "d-fedosd": "FedOSD",
    "d-fedrecovery": "FedRecovery",
    "d-oblivionis": "Oblivionis",
}
RETRAIN_F1 = {
    "20newsgroups": (54.93, 1.25),
    "yahoo_subset": (72.18, 1.52),
}
BAR_LABELS = ["Retrain", "Base", "Base+AS", "Base+LS", "Base+AS+LS"]
BAR_COLORS = ["#48A9B8", "#C7C7C7", "#F2D59A", "#F28B61", "#5B84A8"]


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 13,
            "axes.linewidth": 1.0,
            "figure.dpi": 160,
            "savefig.bbox": "tight",
        }
    )


def _save(fig: plt.Figure, report_name: str, paper_name: str | None = None) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(REPORTS / report_name, dpi=240)


def _panel_label(ax: plt.Axes, dataset: str, method: str, *, compact: bool = False) -> None:
    text = f"{DATASET_LABEL[dataset]}, {METHOD_LABEL[method]}"
    ax.text(
        0.04,
        0.94,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9 if compact else 10,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.5},
    )


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.spines["top"].set_linewidth(1.0)
    ax.spines["right"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.spines["left"].set_linewidth(1.0)


def _final_param(clean: pd.DataFrame, dataset: str, method: str) -> tuple[int | None, float | None]:
    row = clean[(clean.dataset == dataset) & (clean.method == method)].iloc[0]
    text = str(row.dsu_param)
    k_match = re.search(r"k=(\d+)", text)
    r_match = re.search(r"r=([0-9.]+)", text)
    k = int(k_match.group(1)) if k_match else None
    r = float(r_match.group(1)) if r_match else None
    return k, r


def plot_as_k(asls: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(15, 6), sharey=False)
    for i, dataset in enumerate(DATASETS):
        for j, method in enumerate(METHODS):
            ax = axes[i, j]
            sub = asls[
                (asls.dataset == dataset)
                & (asls.algorithm == method)
                & (asls.setting == "AS")
            ].copy()
            sub["k"] = sub["k"].astype(int)
            sub = sub.sort_values("k")
            ax.plot(
                sub["k"],
                sub["macro_f1_best_mean"] * 100,
                marker="o",
                linewidth=1.8,
                color="#5B84A8",
            )
            best = sub.loc[sub["macro_f1_best_mean"].idxmax()]
            ax.scatter(
                [best["k"]],
                [best["macro_f1_best_mean"] * 100],
                s=55,
                color="#B22222",
                zorder=3,
            )
            ax.set_title(
                f"{DATASET_LABEL[dataset]} / {METHOD_LABEL[method]}",
                fontsize=11,
                pad=6,
            )
            ax.set_xticks(range(1, 10))
            _style_axis(ax)
            if i == 1:
                ax.set_xlabel("Selected agents k")
            if j == 0:
                ax.set_ylabel("F1 (%)")
    fig.tight_layout(rect=[0, 0, 1, 1])
    _save(
        fig,
        "tdb_final_local_as_k_f1_20260603.png",
        "as_k_sweep_final.png",
    )
    plt.close(fig)


def plot_ls_r(asls: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(15, 6), sharey=False)
    for i, dataset in enumerate(DATASETS):
        for j, method in enumerate(METHODS):
            ax = axes[i, j]
            sub = asls[
                (asls.dataset == dataset)
                & (asls.algorithm == method)
                & (asls.setting == "LS")
            ].copy()
            sub["r"] = sub["r"].astype(float)
            sub = sub.sort_values("r")
            ax.plot(
                sub["r"],
                sub["macro_f1_best_mean"] * 100,
                marker="o",
                linewidth=1.8,
                color="#5B84A8",
            )
            best = sub.loc[sub["macro_f1_best_mean"].idxmax()]
            ax.scatter(
                [best["r"]],
                [best["macro_f1_best_mean"] * 100],
                s=55,
                color="#B22222",
                zorder=3,
            )
            ax.set_title(
                f"{DATASET_LABEL[dataset]} / {METHOD_LABEL[method]}",
                fontsize=11,
                pad=6,
            )
            ax.set_xticks(np.round(np.arange(0.1, 1.01, 0.2), 1))
            _style_axis(ax)
            if i == 1:
                ax.set_xlabel("LoRA ratio r")
            if j == 0:
                ax.set_ylabel("F1 (%)")
    fig.tight_layout(rect=[0, 0, 1, 1])
    _save(
        fig,
        "tdb_final_local_ls_r_f1_20260603.png",
        "ls_ratio_sweep_final.png",
    )
    plt.close(fig)


def plot_dsu_heatmaps(dsu: pd.DataFrame, clean: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    last_im = None
    for i, dataset in enumerate(DATASETS):
        for j, method in enumerate(METHODS):
            ax = axes[i, j]
            sub = dsu[(dsu.dataset == dataset) & (dsu.algorithm == method)].copy()
            sub["k"] = sub["k"].astype(int)
            sub["r"] = sub["r"].astype(float)
            regular_r = np.round(np.arange(0.1, 1.01, 0.1), 2)
            sub = sub[sub["r"].round(2).isin(regular_r)]
            pivot = sub.pivot(index="k", columns="r", values="macro_f1_best_mean").sort_index()
            arr = pivot.values * 100
            rs = list(pivot.columns)
            ks = list(pivot.index)
            r_step = rs[1] - rs[0]
            k_step = ks[1] - ks[0]
            last_im = ax.imshow(
                arr,
                origin="lower",
                aspect="auto",
                cmap="viridis",
                extent=[
                    min(rs) - r_step / 2,
                    max(rs) + r_step / 2,
                    min(ks) - k_step / 2,
                    max(ks) + k_step / 2,
                ],
            )
            final_k, final_r = _final_param(clean, dataset, method)
            if final_k in ks and final_r is not None:
                ax.scatter(
                    [final_r],
                    [final_k],
                    marker="*",
                    s=140,
                    color="red",
                    edgecolor="white",
                    linewidth=0.8,
                )
            ax.set_title(
                f"{DATASET_LABEL[dataset]} / {METHOD_LABEL[method]}",
                fontsize=11,
                pad=6,
            )
            ax.set_xticks(rs)
            ax.set_xticklabels([f"{x:g}" for x in rs], rotation=45, fontsize=8)
            ax.set_yticks(ks)
            ax.set_yticklabels([str(x) for x in ks], fontsize=8)
            if i == 1:
                ax.set_xlabel("LoRA ratio r")
            if j == 0:
                ax.set_ylabel("Selected agents k")
    fig.tight_layout(rect=[0, 0.02, 0.95, 1])
    cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(), shrink=0.8, pad=0.01)
    cbar.set_label("F1 (%)")
    _save(
        fig,
        "tdb_final_local_dsu_kr_heatmaps_f1_20260603.png",
        "dsu_joint_sweep_final.png",
    )
    plt.close(fig)


def plot_ablation(clean: pd.DataFrame, dataset: str, file_stem: str) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    x = np.arange(len(METHODS))
    width = 0.15
    offsets = np.linspace(-2, 2, len(BAR_LABELS)) * width
    values: list[list[float]] = []
    errors: list[list[float]] = []

    retrain_mean, retrain_std = RETRAIN_F1[dataset]
    values.append([retrain_mean] * len(METHODS))
    errors.append([retrain_std] * len(METHODS))

    for prefix in ["base", "as", "ls", "dsu"]:
        means = []
        stds = []
        for method in METHODS:
            row = clean[(clean.dataset == dataset) & (clean.method == method)].iloc[0]
            means.append(float(row[f"{prefix}_f1"]) * 100)
            stds.append(float(row[f"{prefix}_f1_std"]) * 100)
        values.append(means)
        errors.append(stds)

    for idx, label in enumerate(BAR_LABELS):
        ax.bar(
            x + offsets[idx],
            values[idx],
            width,
            yerr=errors[idx],
            color=BAR_COLORS[idx],
            label=label,
            error_kw={"elinewidth": 1.0, "ecolor": "black", "capsize": 3},
        )

    ax.set_ylabel("F1 (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABEL[m] for m in METHODS], fontsize=13)
    ax.set_ylim(0, 90)
    _style_axis(ax)
    ax.legend(
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        frameon=True,
        borderpad=0.35,
        handlelength=1.5,
        columnspacing=1.2,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    _save(fig, f"{file_stem}.pdf", f"{file_stem}.pdf")
    plt.close(fig)


def main() -> None:
    configure_matplotlib()
    asls = pd.read_csv(REPORTS / "tdb_as_ls_k1to9_r0p1to1_seed424344_20260527_aggregate.csv")
    dsu = pd.read_csv(REPORTS / "tdb_dsu_joint_k1to9_r0p1to1_seed424344_20260530_final_local_aggregate.csv")
    clean = pd.read_csv(REPORTS / "tdb_clean_final_local_f1_mia_20260602.csv")
    plot_as_k(asls)
    plot_ls_r(asls)
    plot_dsu_heatmaps(dsu, clean)
    plot_ablation(clean, "20newsgroups", "ablation_20news")
    plot_ablation(clean, "yahoo_subset", "ablation_yahoo")
    print("saved final selection and ablation plots")


if __name__ == "__main__":
    main()
