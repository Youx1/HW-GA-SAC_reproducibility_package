from pathlib import Path
import argparse
from typing import Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter


# ============================================================
# Paths
# ============================================================

def get_default_root() -> Path:
    try:
        return Path(__file__).resolve().parents[1]
    except NameError:
        return Path.cwd()


ROOT = get_default_root()
DEFAULT_INPUT_CSV = ROOT / "artifacts" / "eval_seed_city_metrics.csv"
DEFAULT_OUT_DIR = ROOT / "artifacts" / "eval_seed_paper_figures_thesis"


# ============================================================
# Methods
# ============================================================

METHOD_ORDER = [
    "ppo",
    "td3",
    "sac",
    "SAC_Large",
    "GRU_SAC",
    "LSTM_SAC",
    "HW_GA_SAC",
]

METHOD_LABELS = {
    "ppo": "PPO",
    "td3": "TD3",
    "sac": "SAC",
    "SAC_Large": "SAC_Large",
    "GRU_SAC": "GRU_SAC",
    "LSTM_SAC": "LSTM_SAC",
    "HW_GA_SAC": "HW_GA_SAC",
}

# Short labels for crowded axes
DISPLAY_LABELS = {
    "ppo": "PPO",
    "td3": "TD3",
    "sac": "SAC",
    "SAC_Large": "SAC-L",
    "GRU_SAC": "GRU",
    "LSTM_SAC": "LSTM",
    "HW_GA_SAC": "HW_GA_SAC",
}

# Medium-saturation solid colors suitable for thesis/paper figures
COLORS = {
    "ppo": "#9C9C9C",       # grey
    "td3": "#5E88B0",       # muted blue
    "sac": "#68A95C",       # muted green
    "SAC_Large": "#D69035", # muted orange
    "GRU_SAC": "#74AFA8",   # muted teal
    "LSTM_SAC": "#B082AF",      # muted purple
    "HW_GA_SAC": "#D95A5A",     # muted red
}

RATE_COLS = ["success_rate", "collision_rate", "timeout_rate"]
SUMMARY_COLS = RATE_COLS + ["mean_reward", "path_efficiency"]


# Thesis figure sizes.
# A typical thesis text width is about 150-160 mm, i.e. 5.9-6.3 inches.
# Use slightly wider width to retain readability in Word/PDF.
THESIS_TEXT_W = 6.8
FIG1_H = 2.35
STANDARD_H = 2.75


# ============================================================
# Style
# ============================================================

def configure_style() -> None:
    mpl.rcdefaults()
    mpl.style.use("default")

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],

            # Moderate sizes for thesis figures.
            # They look large when previewed as PNG, but are suitable after insertion.
            "font.size": 7.5,
            "axes.titlesize": 8.2,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 6.8,
            "figure.titlesize": 8.5,

            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",

            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "xtick.major.size": 2.8,
            "ytick.major.size": 2.8,

            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.45,
            "grid.alpha": 0.80,

            "legend.frameon": False,

            # Keep text editable in vector files
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",

            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.025,
        }
    )


# ============================================================
# Data
# ============================================================

def load_seed_summary(input_csv: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)

    required_cols = {"method", "train_seed"} | set(SUMMARY_COLS)
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in CSV: {sorted(missing_cols)}")

    missing_methods = set(METHOD_ORDER) - set(df["method"].unique())
    if missing_methods:
        raise ValueError(f"Missing methods in CSV: {sorted(missing_methods)}")

    # First average city seeds within each training seed.
    seed_summary = (
        df.groupby(["method", "train_seed"], as_index=False)[SUMMARY_COLS]
        .mean()
        .assign(method=lambda x: pd.Categorical(x["method"], METHOD_ORDER, ordered=True))
        .sort_values(["method", "train_seed"])
    )

    # Then compute mean/std across training seeds.
    summary = (
        seed_summary.groupby("method", observed=False)[SUMMARY_COLS]
        .agg(["mean", "std"])
        .reindex(METHOD_ORDER)
    )

    flat_summary = pd.DataFrame(index=METHOD_ORDER)
    for col in SUMMARY_COLS:
        flat_summary[f"{col}_mean"] = summary[(col, "mean")]
        flat_summary[f"{col}_std"] = summary[(col, "std")].fillna(0.0)

    flat_summary.insert(0, "method_label", [METHOD_LABELS[m] for m in METHOD_ORDER])

    return df, seed_summary, flat_summary


# ============================================================
# Helpers
# ============================================================

def arr(values) -> np.ndarray:
    return np.nan_to_num(np.asarray(values, dtype=float), nan=0.0)


def save_figure(fig: mpl.figure.Figure, out_dir: Path, basename: str, dpi: int = 900) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(out_dir / f"{basename}.svg")
    fig.savefig(out_dir / f"{basename}.pdf")
    fig.savefig(out_dir / f"{basename}.png", dpi=dpi)

    plt.close(fig)


def format_method_axis(ax: plt.Axes, x: np.ndarray, rotation: int = 28) -> None:
    ax.set_xticks(x)
    ax.set_xticklabels(
        [DISPLAY_LABELS[m] for m in METHOD_ORDER],
        rotation=rotation,
        ha="right" if rotation else "center",
        rotation_mode="anchor",
    )
    ax.tick_params(axis="x", pad=1.5)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", visible=False)


def padded_ylim(
    values: np.ndarray,
    errors: Optional[np.ndarray] = None,
    pad_ratio: float = 0.12,
    min_lo: Optional[float] = None,
    min_hi: Optional[float] = None,
) -> Tuple[float, float]:
    values = arr(values)

    if errors is not None:
        errors = arr(errors)
        lo = float(np.min(values - errors))
        hi = float(np.max(values + errors))
    else:
        lo = float(np.min(values))
        hi = float(np.max(values))

    if min_lo is not None:
        lo = min(lo, min_lo)
    if min_hi is not None:
        hi = max(hi, min_hi)

    pad = max((hi - lo) * pad_ratio, 1e-6)
    return lo - pad, hi + pad


def bar_with_error(
    ax: plt.Axes,
    values: np.ndarray,
    errors: np.ndarray,
    ylabel: str,
    title: str,
    ylim: Tuple[float, float],
    percent: bool = False,
    show_value_labels: bool = False,
    value_fmt: str = "{:.2f}",
    rotation: int = 28,
) -> None:
    values = arr(values)
    errors = arr(errors)

    x = np.arange(len(METHOD_ORDER))

    ax.bar(
        x,
        values,
        yerr=errors,
        capsize=2.2,
        width=0.62,
        color=[COLORS[m] for m in METHOD_ORDER],
        edgecolor="#444444",
        linewidth=0.55,
        error_kw={
            "elinewidth": 0.75,
            "capthick": 0.75,
            "ecolor": "#333333",
        },
    )

    ax.set_title(title, loc="left", fontweight="bold", pad=3)
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)

    if percent:
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))

    format_method_axis(ax, x, rotation=rotation)

    if show_value_labels:
        span = ylim[1] - ylim[0]
        for xpos, val, err in zip(x, values, errors):
            y_pos = min(ylim[1] - 0.035 * span, val + err + 0.035 * span)
            ax.text(
                xpos,
                y_pos,
                value_fmt.format(val),
                ha="center",
                va="bottom",
                fontsize=6.4,
                color="#333333",
            )


def validate_outcome_composition(summary: pd.DataFrame, tol: float = 1e-3) -> None:
    total = (
        summary["success_rate_mean"]
        + summary["collision_rate_mean"]
        + summary["timeout_rate_mean"]
    )

    bad = np.abs(total - 1.0) > tol
    if bad.any():
        bad_methods = [METHOD_LABELS[m] for m in total.index[bad]]
        raise ValueError(
            "Outcome rates do not sum to 1.0, so stacked composition is invalid. "
            f"Problem methods: {bad_methods}. "
            f"Totals: {total[bad].round(4).to_dict()}"
        )


# ============================================================
# Figures
# ============================================================

def plot_fig1(summary: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    """
    Thesis version: keep Fig.1 as a 1 x 3 horizontal panel.
    This is suitable for thesis/page-width insertion, not strict journal single-column insertion.
    """
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(THESIS_TEXT_W, FIG1_H),
        sharey=True,
    )

    specs = [
        ("success_rate", "(a) Success ↑"),
        ("collision_rate", "(b) Collision ↓"),
        ("timeout_rate", "(c) Timeout ↓"),
    ]

    for ax, (col, title) in zip(axes, specs):
        values = arr(summary[f"{col}_mean"])
        errors = arr(summary[f"{col}_std"])

        bar_with_error(
            ax=ax,
            values=values,
            errors=errors,
            ylabel="Rate",
            title=title,
            ylim=(0.0, 1.16),
            percent=True,
            show_value_labels=False,
            rotation=34,
        )
        ax.set_yticks(np.linspace(0.0, 1.0, 6))

    axes[1].set_ylabel("")
    axes[2].set_ylabel("")
    axes[1].tick_params(axis="y", labelleft=False)
    axes[2].tick_params(axis="y", labelleft=False)

    fig.subplots_adjust(
        left=0.065,
        right=0.995,
        top=0.86,
        bottom=0.30,
        wspace=0.13,
    )

    save_figure(fig, out_dir, "fig1_outcome_rates_1x3_thesis", dpi=dpi)


def plot_fig2(seed_summary: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    """
    Seed stability figure.
    Points represent individual training seeds; black horizontal lines represent means.
    SD error bars are intentionally omitted to avoid overloading the figure when n=3.
    """
    fig, ax = plt.subplots(figsize=(THESIS_TEXT_W * 0.72, STANDARD_H))

    x = np.arange(len(METHOD_ORDER))

    for idx, method in enumerate(METHOD_ORDER):
        vals = seed_summary.loc[
            seed_summary["method"] == method, "success_rate"
        ].to_numpy(dtype=float)

        vals = arr(vals)
        mean_val = float(np.mean(vals))
        jitter = np.linspace(-0.09, 0.09, len(vals))

        ax.scatter(
            idx + jitter,
            vals,
            s=24,
            color=COLORS[method],
            edgecolor="white",
            linewidth=0.50,
            alpha=0.95,
            zorder=3,
        )

        ax.plot(
            [idx - 0.18, idx + 0.18],
            [mean_val, mean_val],
            color="#222222",
            lw=0.95,
            zorder=4,
        )

    ax.set_title("Success Rate Across Training Seeds", loc="left", fontweight="bold", pad=3)
    ax.set_ylabel("Success Rate")
    ax.set_ylim(-0.03, 1.04)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))

    format_method_axis(ax, x, rotation=28)

    fig.subplots_adjust(
        left=0.13,
        right=0.98,
        top=0.88,
        bottom=0.25,
    )

    save_figure(fig, out_dir, "fig2_success_seed_stability_thesis", dpi=dpi)


def plot_fig3(summary: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    """
    Reward-success trade-off.
    To avoid label congestion, only the HW_GA_SAC method is annotated directly.
    Other methods are shown by legend.
    """
    fig, ax = plt.subplots(figsize=(THESIS_TEXT_W * 0.78, 3.10))

    x = arr(summary["success_rate_mean"])
    xerr = arr(summary["success_rate_std"])
    y = arr(summary["mean_reward_mean"])
    yerr = arr(summary["mean_reward_std"])

    for i, method in enumerate(METHOD_ORDER):
        ax.errorbar(
            x[i],
            y[i],
            xerr=xerr[i],
            yerr=yerr[i],
            fmt="o",
            markersize=5.7 if method == "HW_GA_SAC" else 5.0,
            color=COLORS[method],
            markeredgecolor="white",
            markeredgewidth=0.60,
            ecolor=COLORS[method],
            elinewidth=0.75,
            capsize=2.0,
            alpha=0.95,
            zorder=4 if method == "HW_GA_SAC" else 3,
            label=DISPLAY_LABELS[method],
        )

    gidx = METHOD_ORDER.index("HW_GA_SAC")
    ax.text(
        x[gidx] - 0.105,
        y[gidx] + 12,
        "HW_GA_SAC",
        fontsize=7.0,
        fontweight="bold",
        color="#222222",
        zorder=5,
    )

    ax.set_title("Reward-Success Trade-off", loc="left", fontweight="bold", pad=3)
    ax.set_xlabel("Success Rate")
    ax.set_ylabel("Mean Reward")

    x_lo = max(0.0, float(np.min(x - xerr)) - 0.04)
    x_hi = min(1.02, float(np.max(x + xerr)) + 0.04)

    y_lo = float(np.min(y - yerr))
    y_hi = float(np.max(y + yerr))
    y_pad = 0.12 * (y_hi - y_lo)

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo - y_pad, y_hi + y_pad)

    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.axhline(
        0.0,
        color="#999999",
        linewidth=0.70,
        linestyle="--",
        zorder=1,
    )

    ax.legend(
        ncols=2,
        loc="lower right",
        fontsize=6.2,
        handlelength=1.0,
        columnspacing=0.8,
        borderaxespad=0.2,
    )

    fig.subplots_adjust(
        left=0.13,
        right=0.98,
        top=0.88,
        bottom=0.18,
    )

    save_figure(fig, out_dir, "fig3_reward_success_tradeoff_thesis", dpi=dpi)


def plot_fig4(summary: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(THESIS_TEXT_W * 0.72, STANDARD_H))

    values = arr(summary["path_efficiency_mean"])
    errors = arr(summary["path_efficiency_std"])

    y_upper = max(2.0, float(np.max(values + errors)) * 1.12)

    bar_with_error(
        ax=ax,
        values=values,
        errors=errors,
        ylabel="Path Efficiency",
        title="Path Efficiency ↓",
        ylim=(0.0, y_upper),
        percent=False,
        show_value_labels=True,
        value_fmt="{:.2f}",
        rotation=28,
    )

    # Explain this line in the caption:
    # The dashed line denotes the theoretical lower bound of path efficiency.
    ax.axhline(
        1.0,
        color="#666666",
        linewidth=0.80,
        linestyle="--",
        zorder=1,
    )

    fig.subplots_adjust(
        left=0.13,
        right=0.98,
        top=0.88,
        bottom=0.25,
    )

    save_figure(fig, out_dir, "fig4_path_efficiency_thesis", dpi=dpi)


def plot_figs1(summary: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    """
    Optional supplementary figure.
    Only valid when success/collision/timeout are mutually exclusive and exhaustive.
    """
    validate_outcome_composition(summary)

    fig, ax = plt.subplots(figsize=(THESIS_TEXT_W * 0.82, 3.00))
    x = np.arange(len(METHOD_ORDER))

    success = arr(summary["success_rate_mean"])
    collision = arr(summary["collision_rate_mean"])
    timeout = arr(summary["timeout_rate_mean"])

    stacks = [
        ("Success", success, "#4A965F"),
        ("Collision", collision, "#D97706"),
        ("Timeout", timeout, "#75879A"),
    ]

    bottom = np.zeros(len(METHOD_ORDER), dtype=float)

    for label, values, color in stacks:
        ax.bar(
            x,
            values,
            width=0.62,
            bottom=bottom,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.60,
        )

        for xpos, btm, val in zip(x, bottom, values):
            if val >= 0.08:
                ax.text(
                    xpos,
                    btm + val / 2.0,
                    f"{val:.0%}",
                    ha="center",
                    va="center",
                    fontsize=6.6,
                    color="white",
                    fontweight="bold",
                )

        bottom += values

    ax.set_title("Outcome Composition", loc="left", fontweight="bold", pad=3)
    ax.set_ylabel("Rate")
    ax.set_ylim(0.0, 1.02)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))

    format_method_axis(ax, x, rotation=28)

    # Put legend below the figure to avoid title overlap.
    ax.legend(
        ncols=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.28),
        columnspacing=1.2,
        handlelength=1.4,
    )

    fig.subplots_adjust(
        left=0.12,
        right=0.98,
        top=0.88,
        bottom=0.34,
    )

    save_figure(fig, out_dir, "figS1_outcome_composition_thesis", dpi=dpi)


# ============================================================
# Outputs
# ============================================================

def write_readme(
    input_csv: Path,
    out_dir: Path,
    summary: pd.DataFrame,
    seed_summary: pd.DataFrame,
    dpi: int,
) -> None:
    lines = [
        "# Thesis Evaluation Figures",
        "",
        f"Source: `{input_csv}`",
        "",
        "Aggregation rule:",
        "city seeds are averaged within each training seed; reported bars and error bars use mean +/- sample standard deviation across training seeds.",
        "",
        f"PNG resolution: {dpi} dpi.",
        "Vector formats: SVG and PDF.",
        "",
        "Generated figures:",
        "- `fig1_outcome_rates_1x3_thesis`: success, collision, and timeout rates in a 1 x 3 panel.",
        "- `fig2_success_seed_stability_thesis`: success-rate stability across training seeds.",
        "- `fig3_reward_success_tradeoff_thesis`: mean reward versus success rate.",
        "- `fig4_path_efficiency_thesis`: path-efficiency comparison.",
        "- `figS1_outcome_composition_thesis`: optional stacked outcome composition.",
        "",
        "Caption notes:",
        "- For bar charts, error bars denote standard deviation across training seeds.",
        "- In the path-efficiency figure, the dashed line denotes the theoretical lower bound of path efficiency.",
        "- For the seed-stability figure, points denote individual training seeds and black horizontal lines denote means.",
        "",
        "Method labels:",
    ]

    for method in METHOD_ORDER:
        lines.append(f"- `{method}` -> {METHOD_LABELS[method]}")

    lines.append("")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")

    summary.to_csv(out_dir / "summary_mean_std_by_training_seed.csv", index_label="method")

    seed_summary.assign(
        method_label=seed_summary["method"].astype(str).map(METHOD_LABELS)
    ).to_csv(
        out_dir / "per_training_seed_metrics.csv",
        index=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate thesis-style paper-quality evaluation figures."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help=f"Input CSV path. Default: {DEFAULT_INPUT_CSV}",
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory. Default: {DEFAULT_OUT_DIR}",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=900,
        help="PNG resolution. Default: 900.",
    )

    parser.add_argument(
        "--skip-supp",
        action="store_true",
        help="Skip supplementary outcome-composition figure.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    configure_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    _, seed_summary, summary = load_seed_summary(args.input)

    plot_fig1(summary, args.out_dir, args.dpi)
    plot_fig2(seed_summary, args.out_dir, args.dpi)
    plot_fig3(summary, args.out_dir, args.dpi)
    plot_fig4(summary, args.out_dir, args.dpi)

    if not args.skip_supp:
        plot_figs1(summary, args.out_dir, args.dpi)

    write_readme(
        input_csv=args.input,
        out_dir=args.out_dir,
        summary=summary,
        seed_summary=seed_summary,
        dpi=args.dpi,
    )

    print(f"Saved figures to: {args.out_dir}")
    print()
    print(
        summary[
            [
                "method_label",
                "success_rate_mean",
                "collision_rate_mean",
                "timeout_rate_mean",
                "mean_reward_mean",
                "path_efficiency_mean",
            ]
        ].round(4).to_string()
    )


if __name__ == "__main__":
    main()
