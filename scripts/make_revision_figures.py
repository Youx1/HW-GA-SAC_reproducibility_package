from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.patches import Circle, Rectangle
from matplotlib.ticker import PercentFormatter


OUT_DIR = ROOT / "figures" / "revision_20260623_final"

SEVEN_METHOD_SUMMARY = ROOT / "artifacts/eval_seed_paper_figures_thesis/summary_mean_std_by_training_seed.csv"
CORE_SEED_METRICS = ROOT / "artifacts/core_5seed_paper_figures/per_training_seed_metrics.csv"
CORE_SUMMARY = ROOT / "_indexes/manuscript_data_update_20260622/main_core_5seed_summary.csv"
CITY_SPLIT = ROOT / "_indexes/manuscript_data_update_20260622/main_core_5seed_city_split.csv"
SAFETY_ABLATION = (
    ROOT
    / "data/evaluation/revision/safety_ablation_city_eval_full/complete_eval_200ep_20260629_merged/"
    / "safety_ablation_variant_summary.csv"
)
SENSOR_STRESS = ROOT / "evaluation/revision/sensor_noise_latency_stress/stress_eval_summary.csv"
SENSOR_GRID = ROOT / "evaluation/revision/sensor_noise_latency_grid/sensor_grid_method_summary.csv"
EXTENDED_GENERALIZATION_CITY = ROOT / "evaluation/generalization_unseen_cities/aggregate/city_summary.csv"
WIND_ROBUSTNESS = ROOT / "evaluation/robustness_wind/aggregate/method_summary.csv"
WO_CAMERA_EPISODES = ROOT / "evaluation/seed_sweep/per_episode_by_method/HW_GA_SAC_wo_camera"
CITY_CONFIG = (
    ROOT
    / "evaluation/revision/main_comparison_seed4_5/configs/HW_GA_SAC/"
    / "config_mujoco_HW_GA_SAC_seed4_city42.yaml"
)
MUJOCO_ENV = Path("/Users/youxi/VSCode_Project/sac_uav_mujoco/environments/mujoco_env.py")

METHOD_ORDER_7 = ["ppo", "td3", "sac", "SAC_Large", "GRU_SAC", "LSTM_SAC", "HW_GA_SAC"]
METHOD_LABELS_7 = {
    "ppo": "PPO",
    "td3": "TD3",
    "sac": "SAC",
    "SAC_Large": "SAC_Large",
    "GRU_SAC": "GRU_SAC",
    "LSTM_SAC": "LSTM_SAC",
    "HW_GA_SAC": "HW_GA_SAC",
}
CORE_METHODS = ["TD3", "SAC", "HW_GA_SAC"]
CORE_IDS = ["td3", "sac", "HW_GA_SAC"]

METHOD_COLORS = {
    "PPO": "#8E8E8E",
    "TD3": "#4E79A7",
    "SAC": "#59A14F",
    "SAC_Large": "#D9902F",
    "GRU_SAC": "#76B7B2",
    "LSTM_SAC": "#AF7AA1",
    "HW_GA_SAC": "#E15759",
}
OUTCOME_COLORS = {
    "Success": "#59A14F",
    "Collision": "#E15759",
    "Timeout": "#9E9E9E",
}
SPLIT_COLORS = {
    "city1_4_unseen": "#6BAED6",
    "city42_train_replay": "#F2A65A",
}
WIND_COLORS = {
    "no_wind": "#6BAED6",
    "strong": "#F2A65A",
}

JOURNAL_FULL_W = 7.10
STANDARD_H = 2.75
TOP_LEGEND_ANCHOR = (0.54, 1.23)
TOP_WITH_LEGEND = 0.76


def configure_style() -> None:
    mpl.rcdefaults()
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.titlesize": 7.4,
            "axes.labelsize": 7.4,
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.6,
            "figure.titlesize": 7.6,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "xtick.major.size": 2.7,
            "ytick.major.size": 2.7,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.40,
            "grid.alpha": 0.80,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.025,
        }
    )


def clean_axis(ax: plt.Axes, *, x_grid: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", visible=x_grid)


def save_figure(fig: mpl.figure.Figure, basename: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{basename}.svg")
    fig.savefig(OUT_DIR / f"{basename}.pdf")
    fig.savefig(OUT_DIR / f"{basename}.png", dpi=900)
    plt.close(fig)


def pct(values: pd.Series | np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float) * 100.0


def add_value_labels(
    ax: plt.Axes,
    xs: np.ndarray,
    values: np.ndarray,
    errors: Optional[np.ndarray] = None,
    fmt: str = "{:.1f}",
    *,
    y_offset: float = 1.0,
    tops: Optional[np.ndarray] = None,
) -> None:
    if errors is None:
        errors = np.zeros_like(values)
    if tops is None:
        tops = values + errors
    for x, value, top in zip(xs, values, tops):
        ax.text(
            x,
            top + y_offset,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=5.9,
            color="#333333",
        )


def fig03_seven_method_outcomes() -> None:
    summary = pd.read_csv(SEVEN_METHOD_SUMMARY).set_index("method").loc[METHOD_ORDER_7]
    x = np.arange(len(summary), dtype=float)
    width = 0.22
    specs = [
        ("success_rate", "Success", OUTCOME_COLORS["Success"]),
        ("collision_rate", "Collision", OUTCOME_COLORS["Collision"]),
        ("timeout_rate", "Timeout", OUTCOME_COLORS["Timeout"]),
    ]

    fig, ax = plt.subplots(figsize=(JOURNAL_FULL_W, STANDARD_H))
    for idx, (metric, label, color) in enumerate(specs):
        values = pct(summary[f"{metric}_mean"])
        errors = pct(summary[f"{metric}_std"].fillna(0.0))
        ax.bar(
            x + (idx - 1) * width,
            values,
            width=width,
            yerr=errors,
            capsize=2.0,
            color=color,
            edgecolor="#3F3F3F",
            linewidth=0.45,
            error_kw={"elinewidth": 0.70, "capthick": 0.70, "ecolor": "#333333"},
            label=label,
        )

    ax.set_title("Seven-Method Benchmark Outcomes", loc="left", fontweight="bold", pad=2.0)
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0.0, 112.0)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS_7[m] for m in METHOD_ORDER_7], rotation=28, ha="right")
    clean_axis(ax)
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=TOP_LEGEND_ANCHOR, handlelength=1.5, columnspacing=1.2)
    fig.subplots_adjust(left=0.075, right=0.995, top=TOP_WITH_LEGEND, bottom=0.25)
    save_figure(fig, "fig03_seven_method_outcomes")


def fig04_core_5seed_stability() -> None:
    seeds = pd.read_csv(CORE_SEED_METRICS)
    seeds["method_label"] = seeds["method_label"].replace({"HW_GA_SAC": "HW_GA_SAC"})
    x = np.arange(len(CORE_METHODS), dtype=float)
    width = 0.22
    specs = [
        ("success_rate", "Success", OUTCOME_COLORS["Success"]),
        ("collision_rate", "Collision", OUTCOME_COLORS["Collision"]),
        ("timeout_rate", "Timeout", OUTCOME_COLORS["Timeout"]),
    ]

    fig, ax = plt.subplots(figsize=(JOURNAL_FULL_W * 0.82, STANDARD_H))
    for metric_idx, (metric, label, color) in enumerate(specs):
        means = []
        stds = []
        for method in CORE_METHODS:
            vals = pct(seeds.loc[seeds["method_label"] == method, metric].to_numpy(dtype=float))
            vals = vals[np.isfinite(vals)]
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
        means_arr = np.asarray(means, dtype=float)
        stds_arr = np.asarray(stds, dtype=float)
        xpos = x + (metric_idx - 1) * width
        ax.bar(
            xpos,
            means_arr,
            width=width,
            yerr=stds_arr,
            capsize=2.0,
            color=color,
            edgecolor="#3F3F3F",
            linewidth=0.45,
            error_kw={"elinewidth": 0.70, "capthick": 0.70, "ecolor": "#333333"},
            label=label,
        )
        for method_idx, method in enumerate(CORE_METHODS):
            vals = pct(seeds.loc[seeds["method_label"] == method, metric].to_numpy(dtype=float))
            vals = vals[np.isfinite(vals)]
            offsets = np.linspace(-0.045, 0.045, len(vals))
            ax.scatter(
                np.full(len(vals), xpos[method_idx]) + offsets,
                vals,
                s=6.5,
                color="white",
                edgecolor="#4A4A4A",
                linewidth=0.35,
                alpha=0.88,
                zorder=4,
            )

    ax.set_title("Five-Seed Core Outcomes", loc="left", fontweight="bold", pad=2.0)
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0.0, 110.0)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xticks(x)
    ax.set_xticklabels(CORE_METHODS)
    clean_axis(ax)
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=TOP_LEGEND_ANCHOR, handlelength=1.5, columnspacing=1.2)
    fig.subplots_adjust(left=0.095, right=0.995, top=TOP_WITH_LEGEND, bottom=0.18)
    save_figure(fig, "fig04_core_5seed_stability")


def fig05_core_path_stretch() -> None:
    seeds = pd.read_csv(CORE_SEED_METRICS)
    seeds["method_label"] = seeds["method_label"].replace({"HW_GA_SAC": "HW_GA_SAC"})
    summary = (
        seeds.groupby("method_label")["path_stretch"]
        .agg(["mean", "std"])
        .reindex(CORE_METHODS)
        .reset_index()
    )
    x = np.arange(len(CORE_METHODS), dtype=float)
    values = summary["mean"].to_numpy(dtype=float)
    errors = summary["std"].fillna(0.0).to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(JOURNAL_FULL_W * 0.64, STANDARD_H))
    ax.bar(
        x,
        values,
        yerr=errors,
        capsize=2.2,
        width=0.58,
        color=[METHOD_COLORS[m] for m in CORE_METHODS],
        edgecolor="#3F3F3F",
        linewidth=0.50,
        error_kw={"elinewidth": 0.75, "capthick": 0.75, "ecolor": "#333333"},
    )
    seed_tops = []
    for idx, label in enumerate(CORE_METHODS):
        vals = seeds.loc[seeds["method_label"] == label, "path_stretch"].to_numpy(dtype=float)
        offsets = np.linspace(-0.09, 0.09, len(vals))
        ax.scatter(
            np.full(len(vals), idx) + offsets,
            vals,
            s=7,
            color="white",
            edgecolor="#666666",
            linewidth=0.38,
            alpha=0.82,
            zorder=3,
        )
        seed_tops.append(float(np.nanmax(vals)))
    label_tops = np.maximum(values + errors, np.asarray(seed_tops, dtype=float))
    add_value_labels(ax, x, values, errors, fmt="{:.2f}", y_offset=0.08, tops=label_tops)
    ax.axhline(1.0, color="#666666", linewidth=0.75, linestyle="--", zorder=1)
    ax.set_title("Successful-Path Stretch Under the Core Protocol", loc="left", fontweight="bold", pad=2.0)
    ax.set_ylabel("Successful-Path Stretch")
    ax.set_ylim(0.0, max(1.85, float(np.max(values + errors)) + 0.25))
    ax.set_xticks(x)
    ax.set_xticklabels(CORE_METHODS)
    clean_axis(ax)
    fig.subplots_adjust(left=0.14, right=0.98, top=0.88, bottom=0.18)
    save_figure(fig, "fig05_core_path_stretch")


def fig06_city_split_success() -> None:
    df = pd.read_csv(CITY_SPLIT)
    split_order = ["city1_4_unseen", "city42_train_replay"]
    split_labels = ["Unseen\ncity1-4", "Training replay\ncity42"]
    x = np.arange(len(CORE_METHODS), dtype=float)
    width = 0.32

    fig, ax = plt.subplots(figsize=(JOURNAL_FULL_W * 0.70, STANDARD_H))
    for idx, split in enumerate(split_order):
        subset = (
            df[df["city_split"] == split]
            .set_index("method")
            .loc[CORE_METHODS]
            .reset_index()
        )
        values = pct(subset["success_rate_mean"])
        errors = pct(subset["success_rate_std"].fillna(0.0))
        xpos = x + (idx - 0.5) * width
        ax.bar(
            xpos,
            values,
            width=width,
            yerr=errors,
            capsize=2.0,
            color=SPLIT_COLORS[split],
            edgecolor="#3F3F3F",
            linewidth=0.45,
            error_kw={"elinewidth": 0.70, "capthick": 0.70, "ecolor": "#333333"},
            label=split_labels[idx].replace("\n", " "),
        )
    ax.set_title("Training-City Replay Versus Unseen-Layout Success", loc="left", fontweight="bold", pad=2.0)
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(0.0, 108.0)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xticks(x)
    ax.set_xticklabels(CORE_METHODS)
    clean_axis(ax)
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=TOP_LEGEND_ANCHOR, handlelength=1.4, columnspacing=1.4)
    fig.subplots_adjust(left=0.12, right=0.995, top=TOP_WITH_LEGEND, bottom=0.18)
    save_figure(fig, "fig06_city_split_success")


def fig07_safety_ablation_success() -> None:
    df = pd.read_csv(SAFETY_ABLATION).set_index("variant")
    wo_camera_frames = []
    for path in sorted(WO_CAMERA_EPISODES.glob("*.csv")):
        wo_camera_frames.append(pd.read_csv(path, encoding="utf-8-sig"))
    wo_camera = pd.concat(wo_camera_frames, ignore_index=True)
    wo_camera_groups = (
        wo_camera.groupby(["train_seed", "city_seed"])["success"]
        .mean()
        .to_numpy(dtype=float)
    )

    variants = [
        ("full", "Full", df.loc["full", "success_rate_mean"], df.loc["full", "success_rate_std"]),
        (
            "wo_camera",
            "w/o camera",
            float(np.mean(wo_camera_groups)),
            float(np.std(wo_camera_groups, ddof=1)),
        ),
        ("wo_lidar", "w/o LiDAR\nsectors", df.loc["wo_lidar", "success_rate_mean"], df.loc["wo_lidar", "success_rate_std"]),
        (
            "wo_handcrafted_safety",
            "w/o handcrafted\nsafety cues",
            df.loc["wo_handcrafted_safety", "success_rate_mean"],
            df.loc["wo_handcrafted_safety", "success_rate_std"],
        ),
    ]
    labels = [item[1] for item in variants]
    values = pct(np.asarray([item[2] for item in variants], dtype=float))
    errors = pct(np.asarray([item[3] for item in variants], dtype=float))
    colors = ["#59A14F", "#4E79A7", "#F28E2B", "#B07AA1"]
    x = np.arange(len(variants), dtype=float)

    fig, ax = plt.subplots(figsize=(JOURNAL_FULL_W * 0.78, STANDARD_H))
    ax.bar(
        x,
        values,
        yerr=errors,
        capsize=2.2,
        width=0.58,
        color=colors,
        edgecolor="#3F3F3F",
        linewidth=0.50,
        error_kw={"elinewidth": 0.75, "capthick": 0.75, "ecolor": "#333333"},
    )
    add_value_labels(ax, x, values, errors, fmt="{:.1f}", y_offset=1.0)
    ax.set_title("Decoupled Safety-Input Ablation", loc="left", fontweight="bold", pad=2.0)
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(0.0, 118.0)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    clean_axis(ax)
    fig.subplots_adjust(left=0.105, right=0.99, top=0.88, bottom=0.25)
    save_figure(fig, "fig07_safety_ablation_success")


def fig08_sensor_stress_success() -> None:
    df = pd.read_csv(SENSOR_STRESS)
    condition_order = ["clean", "noise", "noise_latency"]
    condition_labels = ["Clean", "Moderate\nnoise", "Moderate noise\n+ 50 ms latency"]
    methods = ["SAC", "HW_GA_SAC"]
    x = np.arange(len(condition_order), dtype=float)
    width = 0.32

    fig, ax = plt.subplots(figsize=(JOURNAL_FULL_W * 0.70, STANDARD_H))
    for idx, method in enumerate(methods):
        subset = df[df["method"] == method].set_index("condition").loc[condition_order]
        values = pct(subset["success_rate"])
        xpos = x + (idx - 0.5) * width
        ax.bar(
            xpos,
            values,
            width=width,
            color=METHOD_COLORS[method],
            edgecolor="#3F3F3F",
            linewidth=0.45,
            label=method,
        )
        for bar_x, value in zip(xpos, values):
            ax.text(
                bar_x,
                min(100.3, value + 0.55),
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=5.6,
                color="#333333",
            )

    ax.set_title("Robustness Under Sensor Noise and Latency", loc="left", fontweight="bold", pad=2.0)
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(84.0, 101.0)
    ax.set_yticks([85, 90, 95, 100])
    ax.set_xticks(x)
    ax.set_xticklabels(condition_labels)
    clean_axis(ax)
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=TOP_LEGEND_ANCHOR, handlelength=1.4, columnspacing=1.5)
    fig.subplots_adjust(left=0.11, right=0.99, top=TOP_WITH_LEGEND, bottom=0.25)
    save_figure(fig, "fig08_sensor_stress_success")


def figA3_sensor_noise_latency_grid() -> None:
    df = pd.read_csv(SENSOR_GRID)
    methods = ["TD3", "SAC", "HW_GA_SAC"]
    panels = [
        (
            "Noise",
            ["clean", "mild_noise", "moderate_noise"],
            ["Clean", "Mild", "Moderate"],
        ),
        (
            "Latency",
            ["clean", "latency_10ms", "latency_30ms", "latency_50ms"],
            ["Clean", "10 ms", "30 ms", "50 ms"],
        ),
        (
            "Noise + Latency",
            ["clean", "mild_noise_latency_30ms", "moderate_noise_latency_50ms"],
            ["Clean", "Mild\n+30 ms", "Moderate\n+50 ms"],
        ),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(JOURNAL_FULL_W, STANDARD_H * 1.05), sharey=True)
    for ax_idx, (ax, (title, conditions, labels)) in enumerate(zip(axes, panels)):
        x = np.arange(len(conditions), dtype=float)
        for method in methods:
            subset = df[df["method"] == method].set_index("condition").loc[conditions]
            values = pct(subset["success_rate_mean"])
            errors = pct(subset["success_rate_std"].fillna(0.0))
            ax.errorbar(
                x,
                values,
                yerr=errors,
                marker="o",
                markersize=3.2,
                linewidth=1.15,
                elinewidth=0.65,
                capsize=2.0,
                capthick=0.65,
                color=METHOD_COLORS[method],
                markeredgecolor="white",
                markeredgewidth=0.40,
                label=method,
                zorder=3,
            )
        ax.set_title(f"({chr(97 + ax_idx)}) {title}", loc="left", fontweight="bold", pad=2.0)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(20.0, 108.0)
        ax.set_yticks([25, 50, 75, 100])
        ax.grid(axis="x", visible=False)
        clean_axis(ax)
        ax.margins(x=0.08)

    axes[0].set_ylabel("Success Rate (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.52, 1.03),
        frameon=False,
        handlelength=1.7,
        columnspacing=1.4,
    )
    fig.subplots_adjust(left=0.075, right=0.995, top=0.82, bottom=0.23, wspace=0.24)
    save_figure(fig, "figA3_sensor_noise_latency_grid")


def figA4_extended_unseen_city_generalization() -> None:
    df = pd.read_csv(EXTENDED_GENERALIZATION_CITY)
    methods = ["TD3", "SAC", "HW_GA_SAC"]
    x = np.arange(len(methods), dtype=float)

    grouped = (
        df.groupby(["method_label", "model_seed"])["success_rate"]
        .mean()
        .reset_index()
    )
    city_grouped = (
        df.groupby(["method_label", "city_seed"])["success_rate"]
        .mean()
        .reset_index()
    )

    means = []
    stds = []
    for method in methods:
        vals = grouped.loc[grouped["method_label"] == method, "success_rate"].to_numpy(dtype=float)
        means.append(float(np.mean(vals)))
        stds.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
    values = pct(np.asarray(means, dtype=float))
    errors = pct(np.asarray(stds, dtype=float))

    fig, ax = plt.subplots(figsize=(JOURNAL_FULL_W * 0.70, STANDARD_H))
    ax.bar(
        x,
        values,
        yerr=errors,
        capsize=2.2,
        width=0.58,
        color=[METHOD_COLORS[m] for m in methods],
        edgecolor="#3F3F3F",
        linewidth=0.50,
        error_kw={"elinewidth": 0.75, "capthick": 0.75, "ecolor": "#333333"},
        zorder=2,
    )
    for idx, method in enumerate(methods):
        city_vals = pct(city_grouped.loc[city_grouped["method_label"] == method, "success_rate"].to_numpy(dtype=float))
        offsets = np.linspace(-0.13, 0.13, len(city_vals))
        ax.scatter(
            np.full(len(city_vals), idx) + offsets,
            city_vals,
            s=9,
            color="white",
            edgecolor="#555555",
            linewidth=0.40,
            alpha=0.82,
            zorder=4,
        )

    add_value_labels(ax, x, values, errors, fmt="{:.1f}", y_offset=1.0)
    ax.set_title("Extended Unseen-City Generalization", loc="left", fontweight="bold", pad=2.0)
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(0.0, 112.0)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    clean_axis(ax)
    fig.text(
        0.99,
        0.035,
        "hollow points: city43-52 means",
        ha="right",
        va="bottom",
        fontsize=5.8,
        color="#555555",
    )
    fig.subplots_adjust(left=0.12, right=0.99, top=0.88, bottom=0.22)
    save_figure(fig, "figA4_extended_unseen_city_generalization")


def figA5_wind_robustness() -> None:
    df = pd.read_csv(WIND_ROBUSTNESS)
    methods = ["TD3", "SAC", "HW_GA_SAC"]
    wind_order = ["no_wind", "strong"]
    wind_labels = ["No wind", "Strong wind"]
    x = np.arange(len(methods), dtype=float)
    width = 0.32

    fig, ax = plt.subplots(figsize=(JOURNAL_FULL_W * 0.70, STANDARD_H))
    for idx, wind_level in enumerate(wind_order):
        subset = (
            df[df["wind_level"] == wind_level]
            .set_index("method_label")
            .loc[methods]
            .reset_index()
        )
        values = pct(subset["success_rate_mean"])
        errors = pct(subset["success_rate_std"].fillna(0.0))
        xpos = x + (idx - 0.5) * width
        ax.bar(
            xpos,
            values,
            width=width,
            yerr=errors,
            capsize=2.0,
            color=WIND_COLORS[wind_level],
            edgecolor="#3F3F3F",
            linewidth=0.45,
            error_kw={"elinewidth": 0.70, "capthick": 0.70, "ecolor": "#333333"},
            label=wind_labels[idx],
        )

    ax.set_title("Wind Robustness Under the Simplified MuJoCo Force Model", loc="left", fontweight="bold", pad=2.0)
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(0.0, 108.0)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    clean_axis(ax)
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=TOP_LEGEND_ANCHOR, handlelength=1.4, columnspacing=1.4)
    fig.subplots_adjust(left=0.12, right=0.995, top=TOP_WITH_LEGEND, bottom=0.18)
    save_figure(fig, "figA5_wind_robustness")


def figA1_success_return() -> None:
    summary = pd.read_csv(SEVEN_METHOD_SUMMARY).set_index("method").loc[METHOD_ORDER_7]
    x = summary["success_rate_mean"].to_numpy(dtype=float)
    y = summary["mean_reward_mean"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(JOURNAL_FULL_W * 0.72, STANDARD_H * 1.02))
    for method in METHOD_ORDER_7:
        label = METHOD_LABELS_7[method]
        ax.scatter(
            summary.loc[method, "success_rate_mean"],
            summary.loc[method, "mean_reward_mean"],
            s=31 if method == "HW_GA_SAC" else 27,
            color=METHOD_COLORS[label],
            edgecolor="white",
            linewidth=0.45,
            alpha=0.95,
            zorder=4 if method == "HW_GA_SAC" else 3,
        )

    positions = {
        "ppo": (0.165, -65.0, "left", "center"),
        "td3": (0.865, 137.0, "left", "top"),
        "sac": (0.890, 196.0, "center", "bottom"),
        "SAC_Large": (0.790, 124.0, "left", "top"),
        "GRU_SAC": (0.655, 112.0, "center", "top"),
        "LSTM_SAC": (0.655, 164.0, "center", "bottom"),
        "HW_GA_SAC": (0.965, 229.0, "center", "bottom"),
    }
    for method in METHOD_ORDER_7:
        text_x, text_y, ha, va = positions[method]
        ax.annotate(
            METHOD_LABELS_7[method],
            xy=(summary.loc[method, "success_rate_mean"], summary.loc[method, "mean_reward_mean"]),
            xytext=(text_x, text_y),
            textcoords="data",
            ha=ha,
            va=va,
            fontsize=6.6,
            fontweight="bold" if method == "HW_GA_SAC" else "normal",
            color="#222222",
            arrowprops={
                "arrowstyle": "->",
                "mutation_scale": 5.5,
                "color": "#8C8C8C",
                "linewidth": 0.55,
                "shrinkA": 3,
                "shrinkB": 3,
            },
        )

    ax.set_xlabel("Success Rate")
    ax.set_ylabel("Mean Reward")
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(-95.0, 245.0)
    ax.set_xticks(np.linspace(0.0, 1.0, 6))
    ax.set_yticks([-50, 0, 50, 100, 150, 200])
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.axhline(0.0, color="#999999", linewidth=0.70, linestyle="--", zorder=1)
    clean_axis(ax, x_grid=True)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.94, bottom=0.18)
    save_figure(fig, "figA1_success_return")


def load_city_generator_class() -> type:
    text = MUJOCO_ENV.read_text(encoding="utf-8-sig")
    tree = ast.parse(text)
    method_names = {
        "_sample_building_height_profile",
        "_sanitize_height_range",
        "_vary_rgba",
        "_append_box_obstacle",
        "_sample_axis_partitions",
        "_point_in_rotated_rect",
        "_point_in_rotated_ellipse",
        "_classify_city_zone",
        "_generate_city_obstacles",
        "_normalize_obstacles",
    }
    env_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MuJoCoUAVEnv")
    methods = [node for node in env_class.body if isinstance(node, ast.FunctionDef) and node.name in method_names]
    class_def = ast.ClassDef(name="CityGenerator", bases=[], keywords=[], body=methods, decorator_list=[])
    module = ast.Module(body=[class_def], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: Dict[str, Any] = {
        "np": np,
        "Any": Any,
        "Dict": Dict,
        "List": List,
        "Optional": Optional,
        "Tuple": Tuple,
    }
    exec(compile(module, filename=str(MUJOCO_ENV), mode="exec"), namespace)
    return namespace["CityGenerator"]


def make_city_generator(mujoco_cfg: Dict[str, Any]) -> Any:
    cls = load_city_generator_class()
    gen = cls()
    gen.world_size = float(mujoco_cfg.get("world_size", 2000.0))
    gen.world_x_range = tuple(float(v) for v in mujoco_cfg.get("world_x_range", [-1000.0, 1000.0]))
    gen.world_y_range = tuple(float(v) for v in mujoco_cfg.get("world_y_range", [-1000.0, 1000.0]))
    gen.z_floor = float(mujoco_cfg.get("z_floor", 2.0))
    gen.z_ceil = float(mujoco_cfg.get("z_ceil", 180.0))
    return gen


def figA2_city_layout_topdown() -> None:
    with CITY_CONFIG.open("r", encoding="utf-8-sig") as handle:
        config = yaml.safe_load(handle)
    mujoco_cfg = config["env"]["mujoco"]
    city_template = dict(mujoco_cfg["city"])
    city_template.pop("palette", None)
    gen = make_city_generator(mujoco_cfg)

    seeds = [42, 1, 2, 3, 4]
    titles = ["City 42\ntraining replay", "City 1\nunseen", "City 2\nunseen", "City 3\nunseen", "City 4\nunseen"]
    fig, axes = plt.subplots(2, 3, figsize=(JOURNAL_FULL_W, 4.55))
    flat_axes = axes.ravel()
    height_bins = np.array([2, 30, 60, 90, 120, 150, 180], dtype=float)
    cmap = mpl.colormaps["turbo"].resampled(len(height_bins) - 1)
    norm = mpl.colors.BoundaryNorm(height_bins, cmap.N, clip=True)

    for idx, (seed, title) in enumerate(zip(seeds, titles)):
        ax = flat_axes[idx]
        city_cfg = dict(city_template)
        city_cfg["seed"] = int(seed)
        obstacles, summary = gen._generate_city_obstacles(city_cfg)
        obstacles = gen._normalize_obstacles(obstacles)
        for obstacle in sorted(obstacles, key=lambda item: float(item["pos"][2] + item["size"][2])):
            pos = np.asarray(obstacle["pos"], dtype=float)
            size = np.asarray(obstacle["size"], dtype=float)
            x0 = pos[0] - size[0]
            y0 = pos[1] - size[1]
            width = 2.0 * size[0]
            depth = 2.0 * size[1]
            top_z = pos[2] + size[2]
            ax.add_patch(
                Rectangle(
                    (x0, y0),
                    width,
                    depth,
                    facecolor=cmap(norm(top_z)),
                    edgecolor="#262626",
                    linewidth=0.10,
                    alpha=0.96,
                )
            )

        ax.add_patch(Circle((0.0, 0.0), 35.0, fill=False, edgecolor="#2F6B9A", linewidth=0.75))
        ax.add_patch(Circle((0.0, 0.0), 850.0, fill=False, edgecolor="#555555", linewidth=0.45, linestyle="--"))
        ax.add_patch(Circle((0.0, 0.0), 1000.0, fill=False, edgecolor="#555555", linewidth=0.45, linestyle="--"))
        ax.text(42, 42, "start", fontsize=5.6, color="#2F6B9A")

        ax.set_title(f"({chr(97 + idx)}) {title}", loc="left", fontweight="bold", pad=2.0)
        ax.set_xlim(gen.world_x_range)
        ax.set_ylim(gen.world_y_range)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([-1000, 0, 1000])
        ax.set_yticks([-1000, 0, 1000])
        ax.tick_params(labelsize=5.7, pad=1.0)
        ax.grid(True, linewidth=0.20, alpha=0.30)
        ax.text(
            0.98,
            0.97,
            f"{summary['obstacle_geoms']} boxes",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=5.6,
            color="#333333",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65, "pad": 0.4},
        )

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=flat_axes[-1], boundaries=height_bins, ticks=height_bins)
    cbar.set_label("Building Top Height (m)", fontsize=6.6)
    cbar.ax.tick_params(labelsize=5.8, width=0.6, length=2.0)
    fig.supxlabel("x (m)", y=0.035, fontsize=7.0)
    fig.supylabel("y (m)", x=0.018, fontsize=7.0)
    fig.subplots_adjust(left=0.065, right=0.985, top=0.94, bottom=0.10, wspace=0.28, hspace=0.26)
    save_figure(fig, "figA2_city_layout_topdown")


def write_readme() -> None:
    lines = [
        "# Revision Figure Set",
        "",
        "Main-text candidate figures:",
        "- `fig03_seven_method_outcomes`: seven-method three-seed benchmark outcomes.",
        "- `fig04_core_5seed_stability`: five-seed success, collision, and timeout outcomes for TD3, SAC, and HW_GA_SAC.",
        "- `fig05_core_path_stretch`: successful-path stretch under the core five-seed protocol.",
        "- `fig06_city_split_success`: unseen city1-4 versus training replay city42 success.",
        "- `fig07_safety_ablation_success`: camera, LiDAR-sector, and handcrafted-safety ablation success.",
        "- `fig08_sensor_stress_success`: synthetic sensor noise and observation latency stress test.",
        "",
        "Appendix candidate figures:",
        "- `figA1_success_return`: seven-method success-return relationship.",
        "- `figA2_city_layout_topdown`: city42 and city1-4 top-down layout montage.",
        "- `figA3_sensor_noise_latency_grid`: full sensor noise, latency, and combined stress grid.",
        "- `figA4_extended_unseen_city_generalization`: city43-52 extended unseen-layout generalization.",
        "- `figA5_wind_robustness`: no-wind versus strong-wind success under the simplified force model.",
        "",
        "Notes:",
        "- Small hollow points in `fig04_core_5seed_stability` show individual training-seed outcomes; bars and error bars show mean +/- sample standard deviation.",
        "- `fig08_sensor_stress_success` uses SAC and HW_GA_SAC only to keep the main-text stress figure compact.",
        "- Numeric labels in `fig08_sensor_stress_success` are absolute success rates, not changes relative to the clean condition.",
        "- `fig08_sensor_stress_success` uses the clean condition of the stress-test protocol; it is not identical to the main five-seed evaluation protocol.",
        "- The higher success values under sensor noise or noise+latency should be treated as evaluation variability, not evidence that degradation improves the policy.",
        "- Hollow points in `fig05_core_path_stretch` are seed-level values; bars and error bars show mean +/- sample standard deviation.",
        "- `figA2_city_layout_topdown` uses discrete height bins to make building-height differences visually separable.",
        "- Figure 1 and Figure 2 are architecture diagrams from the manuscript and should be handled in the manuscript figure assets, not in this data plotting script.",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    configure_style()
    fig03_seven_method_outcomes()
    fig04_core_5seed_stability()
    fig05_core_path_stretch()
    fig06_city_split_success()
    fig07_safety_ablation_success()
    fig08_sensor_stress_success()
    figA1_success_return()
    figA2_city_layout_topdown()
    figA3_sensor_noise_latency_grid()
    figA4_extended_unseen_city_generalization()
    figA5_wind_robustness()
    write_readme()
    print(f"Saved revision figures to: {OUT_DIR}")


if __name__ == "__main__":
    main()

