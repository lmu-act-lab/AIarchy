# Import plotting packages, and read through Training Environment to see how we originally plot
import os
import pandas as pd
from matplotlib import pyplot as plt
from src.training_environment import lookup_color


def _ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _legend_large(ax) -> None:
    leg = ax.legend(loc="best", frameon=False)
    if leg is not None:
        for txt in leg.get_texts():
            txt.set_fontsize(12)


def _resolve_color(name: str, index: int = 0) -> str:
    """Use registered color if available; otherwise fall back to the mpl cycle by index."""
    c = lookup_color(name)
    if c:
        return c
    cycle = plt.rcParams.get("axes.prop_cycle")
    if cycle:
        colors = cycle.by_key().get("color", ["#333333"])
        if colors:
            return colors[index % len(colors)]
    return "#333333"


def write_rewards_summary(dir_path: str) -> None:
    """If rewards.csv exists under dir_path, write rewards_summary.csv with mean/std/count."""
    try:
        rewards_csv = os.path.join(dir_path, "rewards.csv")
        if os.path.exists(rewards_csv):
            df = pd.read_csv(rewards_csv)
            if not df.empty:
                col = df.columns[0]
                summary_df = pd.DataFrame([
                    {
                        "mean": df[col].mean(),
                        "std": df[col].std(ddof=1),
                        "count": int(df[col].shape[0]),
                    }
                ])
                summary_df.to_csv(os.path.join(dir_path, "rewards_summary.csv"), index=False)
    except Exception:
        pass


def plot_monte_carlo_compare(csv_a: str, label_a: str, csv_b: str, label_b: str, out_png: str, ylim: tuple[float, float], xlim: tuple[int, int]) -> None:
    df_a = pd.read_csv(csv_a)
    df_b = pd.read_csv(csv_b)
    fig, ax = plt.subplots(figsize=(12, 6))
    color_a = _resolve_color(label_a, 0)
    color_b = _resolve_color(label_b, 1)
    ax.plot(df_a["iteration"], df_a["mean_reward"], color=color_a, label=label_a, linewidth=2)
    ax.plot(df_b["iteration"], df_b["mean_reward"], color=color_b, label=label_b, linewidth=2)
    ax.fill_between(df_a["iteration"], df_a["mean_reward"] - df_a["std_reward"], df_a["mean_reward"] + df_a["std_reward"], color=color_a, alpha=0.15)
    ax.fill_between(df_b["iteration"], df_b["mean_reward"] - df_b["std_reward"], df_b["mean_reward"] + df_b["std_reward"], color=color_b, alpha=0.15)
    ax.set_ylim(*ylim)
    ax.set_xlim(*xlim)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean Reward")
    ax.set_title("Monte Carlo Rewards")
    _legend_large(ax)
    _ensure_dir(os.path.dirname(out_png))
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def plot_monte_carlo_single(csv_path: str, label: str, out_png: str, ylim: tuple[float, float], xlim: tuple[int, int]) -> None:
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(12, 6))
    color = _resolve_color(label, 0)
    ax.plot(df["iteration"], df["mean_reward"], color=color, label=label, linewidth=2)
    if "std_reward" in df.columns:
        ax.fill_between(df["iteration"], df["mean_reward"] - df["std_reward"], df["mean_reward"] + df["std_reward"], color=color, alpha=0.15)
    ax.set_ylim(*ylim)
    ax.set_xlim(*xlim)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean Reward")
    ax.set_title("Monte Carlo Rewards")
    _legend_large(ax)
    _ensure_dir(os.path.dirname(out_png))
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def plot_u_hat_losses_compare(csv_a: str, label_a: str, csv_b: str, label_b: str, out_png: str, ylim: tuple[float, float], xlim: tuple[int, int]) -> None:
    df_a = pd.read_csv(csv_a)
    df_b = pd.read_csv(csv_b)
    fig, ax = plt.subplots(figsize=(12, 6))
    # mean_loss per iteration per model; multiple models possible – plot each model separately, color by model name
    for i, model_name in enumerate(df_a["model"].unique()):
        sub = df_a[df_a["model"] == model_name]
        color = _resolve_color(model_name, i)
        ax.plot(sub["iteration"], sub["mean_loss"], color=color, label=f"{label_a}: {model_name}", linewidth=2)
    for i, model_name in enumerate(df_b["model"].unique()):
        sub = df_b[df_b["model"] == model_name]
        color = _resolve_color(model_name, i)
        ax.plot(sub["iteration"], sub["mean_loss"], color=color, linestyle="--", label=f"{label_b}: {model_name}", linewidth=2)
    ax.set_ylim(*ylim)
    ax.set_xlim(*xlim)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean Loss")
    ax.set_title("U-hat Model Losses")
    _legend_large(ax)
    _ensure_dir(os.path.dirname(out_png))
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def plot_ema_only(ema_csv: str, out_png: str, ema_ylim=(0, 3), xlim: tuple[int, int] | None = None, title: str = "EMA Over Iterations") -> None:
    ema_df = pd.read_csv(ema_csv)
    fig, ax1 = plt.subplots(figsize=(12, 6))
    x_ema = range(len(ema_df))
    for i, col in enumerate(ema_df.columns):
        ax1.plot(x_ema, ema_df[col], color=_resolve_color(col, i), linewidth=2, label=f"EMA: {col}")
    ax1.set_ylabel("EMA Value")
    ax1.set_ylim(*ema_ylim)
    if xlim is not None:
        ax1.set_xlim(*xlim)
    ax1.set_xlabel("Iteration")
    ax1.set_title(title)
    _legend_large(ax1)
    _ensure_dir(os.path.dirname(out_png))
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def plot_weights_only(weights_csv: str, out_png: str, weights_ylim=(0, 1), xlim: tuple[int, int] | None = None, title: str = "Average Utility Weights") -> None:
    weights_df = pd.read_csv(weights_csv)
    fig, ax2 = plt.subplots(figsize=(12, 6))
    x_w = range(len(weights_df))
    for i, col in enumerate(weights_df.columns):
        ax2.plot(x_w, weights_df[col], color=_resolve_color(col, i), linewidth=2, label=f"{col} weight")
    ax2.set_ylabel("Average Utility Weight")
    ax2.set_ylim(*weights_ylim)
    if xlim is not None:
        ax2.set_xlim(*xlim)
    ax2.set_xlabel("Iteration")
    ax2.set_title(title)
    _legend_large(ax2)
    _ensure_dir(os.path.dirname(out_png))
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def plot_weighted_rewards(csv_path: str, out_png: str, ylim: tuple[float, float], xlim: tuple[int, int]) -> None:
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(12, 6))
    # subjective vs objective utility series if present, else single series
    if {"utility", "type", "mean"}.issubset(df.columns):
        types = sorted(df["type"].unique())
        style_map = {"subjective": "-", "objective": "--"}
        for i, util_name in enumerate(sorted(df["utility"].unique())):
            color = _resolve_color(util_name, i)
            for t in types:
                sub = df[(df["utility"] == util_name) & (df["type"] == t)]
                if not sub.empty:
                    ax.plot(sub["iteration"], sub["mean"], color=color, linestyle=style_map.get(t, "-"), linewidth=2, label=f"{util_name} ({t})")
    elif {"iteration", "mean_reward"}.issubset(df.columns):
        color = _resolve_color("util", 0)
        ax.plot(df["iteration"], df["mean_reward"], color=color, linewidth=2, label="weighted reward")
        if "std_reward" in df.columns:
            ax.fill_between(df["iteration"], df["mean_reward"] - df["std_reward"], df["mean_reward"] + df["std_reward"], color=color, alpha=0.15)
    ax.set_ylim(*ylim)
    ax.set_xlim(*xlim)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Reward")
    ax.set_title("Weighted Rewards")
    _legend_large(ax)
    _ensure_dir(os.path.dirname(out_png))
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


# ============= Tasks =============
# Confounding: hidden vs observed
def run_confounding_plots() -> None:
    base = "base_structs"
    hidden = os.path.join(base, "confound_hidden")
    obs = os.path.join(base, "confound_obs")
    # Rewards summary
    write_rewards_summary(hidden)
    write_rewards_summary(obs)
    # Monte Carlo compare
    plot_monte_carlo_compare(
        os.path.join(hidden, "monte_carlo_rewards.csv"),
        "hidden",
        os.path.join(obs, "monte_carlo_rewards.csv"),
        "observed",
        out_png=os.path.join(base, "confounding_monte_carlo_compare.png"),
        ylim=(0, 2),
        xlim=(0, 2000),
    )
    # U-hat losses compare
    plot_u_hat_losses_compare(
        os.path.join(hidden, "u_hat_model_losses.csv"),
        "hidden",
        os.path.join(obs, "u_hat_model_losses.csv"),
        "observed",
        out_png=os.path.join(base, "confounding_u_hat_losses_compare.png"),
        ylim=(-0.2, 1),
        xlim=(0, 2000),
    )


# Frustration (downweigh_struct)
def run_frustration_plots() -> None:
    path = os.path.join("base_structs", "downweigh_struct")
    write_rewards_summary(path)
    plot_ema_only(
        os.path.join(path, "ema_history.csv"),
        out_png=os.path.join(path, "frustration_ema.png"),
        ema_ylim=(0, 3),
        xlim=(0, 2000),
        title="Frustration EMA"
    )
    plot_weights_only(
        os.path.join(path, "avg_util_weights.csv"),
        out_png=os.path.join(path, "frustration_avg_util_weights.png"),
        weights_ylim=(0, 1),
        xlim=(0, 2000),
        title="Frustration Average Utility Weights"
    )
    plot_weighted_rewards(
        os.path.join(path, "weighted_rewards.csv"),
        out_png=os.path.join(path, "frustration_weighted_rewards.png"),
        ylim=(0, 2),
        xlim=(0, 2000),
    )
    plot_monte_carlo_single(
        os.path.join(path, "monte_carlo_rewards.csv"),
        label="frustration",
        out_png=os.path.join(path, "frustration_monte_carlo.png"),
        ylim=(0, 2),
        xlim=(0, 2000),
    )


# Mediation hidden vs observed
def run_mediation_plots() -> None:
    base = "base_structs"
    hidden = os.path.join(base, "mediation_hidden")
    obs = os.path.join(base, "mediation_obs")
    write_rewards_summary(hidden)
    write_rewards_summary(obs)
    plot_monte_carlo_compare(
        os.path.join(hidden, "monte_carlo_rewards.csv"),
        "hidden",
        os.path.join(obs, "monte_carlo_rewards.csv"),
        "observed",
        out_png=os.path.join(base, "mediation_monte_carlo_compare.png"),
        ylim=(0, 1),
        xlim=(0, 2000),
    )


# Amy vs Leo
def run_amy_vs_leo() -> None:
    base = os.path.join("pub_sims")
    amy = os.path.join(base, "Amy")
    leo = os.path.join(base, "Leo")
    write_rewards_summary(amy)
    write_rewards_summary(leo)
    # Monte Carlo compare
    plot_monte_carlo_compare(
        os.path.join(amy, "monte_carlo_rewards.csv"),
        "Amy",
        os.path.join(leo, "monte_carlo_rewards.csv"),
        "Leo",
        out_png=os.path.join(base, "Amy_vs_Leo_monte_carlo.png"),
        ylim=(0.8, 3.5),
        xlim=(0, 4000),
    )
    # Separated EMA and weights for each
    plot_ema_only(
        os.path.join(amy, "ema_history.csv"),
        out_png=os.path.join(base, "Amy_ema.png"),
        ema_ylim=(0, 3),
        xlim=(0, 4000),
        title="Amy EMA"
    )
    plot_weights_only(
        os.path.join(amy, "avg_util_weights.csv"),
        out_png=os.path.join(base, "Amy_avg_util_weights.png"),
        weights_ylim=(0, 1),
        xlim=(0, 4000),
        title="Amy Average Utility Weights"
    )
    plot_ema_only(
        os.path.join(leo, "ema_history.csv"),
        out_png=os.path.join(base, "Leo_ema.png"),
        ema_ylim=(0, 3),
        xlim=(0, 4000),
        title="Leo EMA"
    )
    plot_weights_only(
        os.path.join(leo, "avg_util_weights.csv"),
        out_png=os.path.join(base, "Leo_avg_util_weights.png"),
        weights_ylim=(0, 1),
        xlim=(0, 4000),
        title="Leo Average Utility Weights"
    )
    # Weighted rewards
    plot_weighted_rewards(
        os.path.join(amy, "weighted_rewards.csv"),
        out_png=os.path.join(base, "Amy_weighted_rewards.png"),
        ylim=(-0.2, 3.5),
        xlim=(0, 4000),
    )
    plot_weighted_rewards(
        os.path.join(leo, "weighted_rewards.csv"),
        out_png=os.path.join(base, "Leo_weighted_rewards.png"),
        ylim=(-0.2, 3.5),
        xlim=(0, 4000),
    )


# Hierarchy
def run_hierarchy_plots() -> None:
    base = os.path.join("src", "final_actlab_sims", "actlab_main", "teacher_hierarchy")
    teacher = os.path.join(base, "Teacher")
    write_rewards_summary(teacher)
    # Teacher MC (single series)
    plot_monte_carlo_single(
        os.path.join(teacher, "monte_carlo_rewards.csv"),
        label="Teacher",
        out_png=os.path.join(base, "Teacher_monte_carlo.png"),
        ylim=(0, 3),
        xlim=(0, 800),
    )
    # Teacher weighted rewards
    plot_weighted_rewards(
        os.path.join(teacher, "weighted_rewards.csv"),
        out_png=os.path.join(base, "Teacher_weighted_rewards.png"),
        ylim=(-0.5, 3),
        xlim=(0, 800),
    )
    # Teacher separate EMA + weights
    plot_ema_only(
        os.path.join(teacher, "ema_history.csv"),
        out_png=os.path.join(base, "Teacher_ema.png"),
        ema_ylim=(0, 3),
        xlim=(0, 800),
        title="Teacher EMA"
    )
    plot_weights_only(
        os.path.join(teacher, "avg_util_weights.csv"),
        out_png=os.path.join(base, "Teacher_avg_util_weights.png"),
        weights_ylim=(0, 1),
        xlim=(0, 800),
        title="Teacher Average Utility Weights"
    )

    # Students vs pub_sims counterparts
    students = ["Amy", "Isla", "Jonas", "Leo", "Mei", "Nikhil"]
    pub_base = os.path.join("pub_sims")
    for student in students:
        stud_dir = os.path.join(base, student)
        pub_dir = os.path.join(pub_base, student)
        write_rewards_summary(stud_dir)
        write_rewards_summary(pub_dir)
        # MC
        plot_monte_carlo_compare(
            os.path.join(stud_dir, "monte_carlo_rewards.csv"),
            "hierarchical",
            os.path.join(pub_dir, "monte_carlo_rewards.csv"),
            "base",
            out_png=os.path.join(base, f"{student}_vs_pub_monte_carlo.png"),
            ylim=(0, 3),
            xlim=(0, 4000),
        )
        # Separate EMA + weights
        plot_ema_only(
            os.path.join(stud_dir, "ema_history.csv"),
            out_png=os.path.join(base, f"{student}_ema.png"),
            ema_ylim=(0, 3),
            xlim=(0, 4000),
            title=f"{student} EMA"
        )
        plot_weights_only(
            os.path.join(stud_dir, "avg_util_weights.csv"),
            out_png=os.path.join(base, f"{student}_avg_util_weights.png"),
            weights_ylim=(0, 1),
            xlim=(0, 4000),
            title=f"{student} Average Utility Weights"
        )
        # Weighted rewards (both teacher student file and pub counterpart)
        plot_weighted_rewards(
            os.path.join(stud_dir, "weighted_rewards.csv"),
            out_png=os.path.join(base, f"{student}_weighted_rewards_teacher_side.png"),
            ylim=(-0.2, 3.5),
            xlim=(0, 4000),
        )
        plot_weighted_rewards(
            os.path.join(pub_dir, "weighted_rewards.csv"),
            out_png=os.path.join(base, f"{student}_weighted_rewards_pub_side.png"),
            ylim=(-0.2, 3.5),
            xlim=(0, 4000),
        )


if __name__ == "__main__":
    # Run subsets as needed; by default, do nothing to avoid long batch work unintentionally.
    run_confounding_plots()
    run_frustration_plots()
    run_mediation_plots()
    run_amy_vs_leo()
    run_hierarchy_plots()
    pass
