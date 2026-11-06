"""
analyze_results.py
==================
Reads all CSV files produced by run_all_envs.py and generates every figure
and statistical table needed for the thesis.

Figures (PNG):
  fig1_learning_curves_{env}.png   — Mean ± 95% CI reward over episodes
  fig2_final_performance.png       — Bar chart of final eval reward (all envs)
  fig3_success_rate.png            — Bar chart of success rate (all envs)
  fig4_beta_evolution.png          — Avg β_t over training (AC-PPO only)
  fig5_divergence.png              — JS divergence over training (AC-PPO only)
  fig6_violation_rate.png          — Violation rate over training (AC-PPO only)

Tables (CSV):
  table2_final_performance.csv     — Mean ± Std, Best, 95% CI per method × env
  table3_statistics.csv            — Normality, p-value, effect size (B3 vs B1/B2)

Usage:
  python analyze_results.py --results_dir results --output_dir analysis
  python analyze_results.py --results_dir results --output_dir analysis --env CartPole-v1
"""

import argparse
import os
import sys
import warnings
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

warnings.filterwarnings("ignore")

# ── Mirror constants from run_all_envs.py ─────────────────────────────────────

ALL_ENVS    = ["CartPole-v1", "Acrobot-v1", "LunarLander-v3"]
MAIN_MODES  = ["baseline", "affect_reward", "full"]
ABLATION_MODES = ["appraisal_only"]
MODES       = MAIN_MODES + ABLATION_MODES

MODE_LABELS = {
    "baseline":       "B1: Standard PPO",
    "affect_reward":  "B2: Affect-as-Reward",
    "full":           "B3: AC-PPO (proposed)",
    "appraisal_only": "A1: AC-PPO (no constraint)",
}
MODE_COLOURS = {
    "baseline":       "#888888",
    "affect_reward":  "#e07b39",
    "full":           "#2ca02c",
    "appraisal_only": "#5b8dd9",
}
MODE_LINESTYLE = {
    "baseline":       "-",
    "affect_reward":  "-",
    "full":           "-",
    "appraisal_only": "--",
}
ENV_DISPLAY = {
    "CartPole-v1":    "CartPole-v1",
    "Acrobot-v1":     "Acrobot-v1",
    "LunarLander-v3": "LunarLander-v3",
}

# Plot style
plt.rcParams.update({
    "font.family":     "sans-serif",
    "font.size":       11,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "figure.dpi":         130,
})


# ── CSV loading ────────────────────────────────────────────────────────────────

def _env_dir(results_dir: str, env: str) -> str:
    return os.path.join(results_dir, env.replace("-", "_"), "csv")


def load_eval_csvs(results_dir: str, env: str, mode: str, seeds: list[int]) -> list[dict]:
    """Load eval CSVs for a given (env, mode) across seeds. Returns list of dicts-of-lists."""
    runs = []
    d = _env_dir(results_dir, env)
    for seed in seeds:
        path = os.path.join(d, f"{mode}_s{seed}_eval.csv")
        if not os.path.exists(path):
            continue
        rows = []
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                rows.append(row)
        if not rows:
            continue
        # Convert to dict-of-arrays
        run = {}
        for key in rows[0]:
            vals = []
            for r in rows:
                v = r[key]
                try:
                    vals.append(float(v))
                except (ValueError, TypeError):
                    vals.append(float("nan"))
            run[key] = np.array(vals)
        run["_seed"] = seed
        runs.append(run)
    return runs


def load_train_csvs(results_dir: str, env: str, mode: str, seeds: list[int]) -> list[dict]:
    """Load train CSVs for a given (env, mode) across seeds."""
    runs = []
    d = _env_dir(results_dir, env)
    for seed in seeds:
        path = os.path.join(d, f"{mode}_s{seed}_train.csv")
        if not os.path.exists(path):
            continue
        rows = []
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                rows.append(row)
        if not rows:
            continue
        run = {}
        for key in rows[0]:
            vals = []
            for r in rows:
                v = r[key]
                try:
                    vals.append(float(v))
                except (ValueError, TypeError):
                    vals.append(float("nan"))
            run[key] = np.array(vals)
        run["_seed"] = seed
        runs.append(run)
    return runs


def discover_seeds(results_dir: str, env: str, mode: str) -> list[int]:
    """Discover which seeds have eval CSVs for a given env/mode."""
    d = _env_dir(results_dir, env)
    seeds = []
    if not os.path.isdir(d):
        return seeds
    for fname in sorted(os.listdir(d)):
        if fname.startswith(f"{mode}_s") and fname.endswith("_eval.csv"):
            try:
                s = int(fname[len(f"{mode}_s"):-len("_eval.csv")])
                seeds.append(s)
            except ValueError:
                pass
    return sorted(seeds)


# ── Statistical helpers ────────────────────────────────────────────────────────

def _ci95(arr: np.ndarray) -> float:
    """95% CI half-width for an array of values."""
    n = len(arr)
    if n < 2:
        return 0.0
    return 1.96 * np.nanstd(arr, ddof=1) / np.sqrt(n)


def _agg_eval(runs: list[dict], key: str = "eval_reward"):
    """Stack eval metric across seeds → (mean, lo, hi) aligned to shortest run."""
    arrays = [r[key] for r in runs if key in r and not np.all(np.isnan(r[key]))]
    if not arrays:
        return None, None, None
    n = min(len(a) for a in arrays)
    mat = np.stack([a[:n] for a in arrays])          # (seeds, checkpoints)
    mean = np.nanmean(mat, axis=0)
    ci   = 1.96 * np.nanstd(mat, axis=0, ddof=1) / np.sqrt(mat.shape[0])
    return mean, mean - ci, mean + ci


def _final_rewards(runs: list[dict], last_n: int = 5) -> np.ndarray:
    """Mean of last_n eval checkpoints per seed → 1-D array of length n_seeds."""
    vals = []
    for r in runs:
        er = r.get("eval_reward", np.array([]))
        er = er[~np.isnan(er)]
        if len(er) == 0:
            continue
        vals.append(float(np.mean(er[-last_n:])))
    return np.array(vals)


def _final_success(runs: list[dict], last_n: int = 5) -> np.ndarray:
    vals = []
    for r in runs:
        es = r.get("eval_success", np.array([]))
        es = es[~np.isnan(es)]
        if len(es) == 0:
            continue
        vals.append(float(np.mean(es[-last_n:])))
    return np.array(vals)


def run_stats(a: np.ndarray, b: np.ndarray, label_a: str, label_b: str) -> dict:
    """
    Normality → Welch's t-test (+ Cohen's d) or Mann-Whitney U (+ rank-biserial).
    Returns a dict of results.
    """
    result = {"label_a": label_a, "label_b": label_b,
              "n_a": len(a), "n_b": len(b),
              "mean_a": float(np.nanmean(a)), "std_a": float(np.nanstd(a, ddof=1)),
              "mean_b": float(np.nanmean(b)), "std_b": float(np.nanstd(b, ddof=1))}
    try:
        from scipy import stats as _stats
    except ImportError:
        result["test"] = "scipy not available"
        return result

    # Normality (Shapiro-Wilk, only reliable for n >= 3)
    def _sw(x):
        if len(x) < 3:
            return 1.0
        _, p = _stats.shapiro(x)
        return p

    p_norm_a = _sw(a)
    p_norm_b = _sw(b)
    result["shapiro_p_a"] = round(p_norm_a, 4)
    result["shapiro_p_b"] = round(p_norm_b, 4)
    normal = (p_norm_a > 0.05) and (p_norm_b > 0.05)

    if normal and len(a) >= 3 and len(b) >= 3:
        t, p = _stats.ttest_ind(a, b, equal_var=False)  # Welch's
        pooled = np.sqrt((np.std(a, ddof=1)**2 + np.std(b, ddof=1)**2) / 2)
        d = (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else 0.0
        result.update({"test": "Welch's t-test", "statistic": round(t, 4),
                        "p_value": round(p, 4), "effect_size": round(d, 4),
                        "effect_label": "Cohen's d"})
    else:
        u, p = _stats.mannwhitneyu(a, b, alternative="two-sided")
        n1, n2 = len(a), len(b)
        r = 1 - 2*u / (n1 * n2)
        result.update({"test": "Mann-Whitney U", "statistic": round(u, 2),
                        "p_value": round(p, 4), "effect_size": round(r, 4),
                        "effect_label": "rank-biserial r"})
    return result


# ── Rolling smooth ─────────────────────────────────────────────────────────────

def _smooth(arr: np.ndarray, w: int = 10) -> np.ndarray:
    if len(arr) < w:
        return arr
    return np.convolve(arr, np.ones(w) / w, mode="valid")


# ── Figure 1: Learning curves ──────────────────────────────────────────────────

def fig1_learning_curves(results_dir: str, output_dir: str, envs: list[str]) -> None:
    print("  Figure 1: Learning curves …")
    for env in envs:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        plotted = False

        for mode in MODES:
            seeds = discover_seeds(results_dir, env, mode)
            runs  = load_eval_csvs(results_dir, env, mode, seeds)
            if not runs:
                continue

            mean, lo, hi = _agg_eval(runs, "eval_reward")
            if mean is None:
                continue

            n = len(mean)
            x = [(i + 1) * 100 for i in range(n)]   # eval every 100 eps

            ls = MODE_LINESTYLE[mode]
            lw = 2.2 if mode in MAIN_MODES else 1.6
            ax.plot(x, mean, color=MODE_COLOURS[mode], linewidth=lw,
                    linestyle=ls, label=MODE_LABELS[mode], zorder=3)
            ax.fill_between(x, lo, hi, color=MODE_COLOURS[mode], alpha=0.15)
            plotted = True

        if not plotted:
            plt.close()
            continue

        ax.set_title(f"Learning Curves — {ENV_DISPLAY[env]}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Episode", fontsize=11)
        ax.set_ylabel("Mean Evaluation Reward", fontsize=11)
        ax.legend(fontsize=9, framealpha=0.7)
        ax.annotate("Shaded region = 95% CI over seeds",
                    xy=(0.02, 0.02), xycoords="axes fraction", fontsize=7.5, color="gray")
        plt.tight_layout()
        path = os.path.join(output_dir, f"fig1_learning_curves_{env.replace('-','_')}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    → {path}")


# ── Figure 2: Final performance bar chart ─────────────────────────────────────

def fig2_final_performance(results_dir: str, output_dir: str, envs: list[str]) -> None:
    print("  Figure 2: Final performance bar chart …")
    n_env  = len(envs)
    n_mode = len(MODES)
    fig, axes = plt.subplots(1, n_env, figsize=(5.5 * n_env, 5), sharey=False)
    if n_env == 1:
        axes = [axes]

    fig.suptitle("Final Evaluation Reward (mean of last 5 checkpoints, ±95% CI)",
                 fontsize=13, fontweight="bold")

    for ax, env in zip(axes, envs):
        means, cis, labels, colours = [], [], [], []
        for mode in MODES:
            seeds = discover_seeds(results_dir, env, mode)
            runs  = load_eval_csvs(results_dir, env, mode, seeds)
            fr    = _final_rewards(runs)
            if len(fr) == 0:
                continue
            means.append(np.mean(fr))
            cis.append(_ci95(fr))
            labels.append(MODE_LABELS[mode].split(":")[0])  # "B1", "B2" etc.
            colours.append(MODE_COLOURS[mode])
            ls = MODE_LINESTYLE[MODES[len(means) - 1]]
            # hatching for ablation
            ax.bar(len(means) - 1, np.mean(fr), color=colours[-1],
                   alpha=0.85, width=0.6, zorder=2,
                   hatch="///" if mode in ABLATION_MODES else "")

        for i, (m, ci) in enumerate(zip(means, cis)):
            ax.errorbar(i, m, yerr=ci, fmt="none", color="black",
                        capsize=5, linewidth=1.5, zorder=3)

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_title(ENV_DISPLAY[env], fontsize=11, fontweight="bold")
        ax.set_ylabel("Mean Final Eval Reward", fontsize=10)
        ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")

    plt.tight_layout()
    path = os.path.join(output_dir, "fig2_final_performance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    → {path}")


# ── Figure 3: Success rate ─────────────────────────────────────────────────────

def fig3_success_rate(results_dir: str, output_dir: str, envs: list[str]) -> None:
    print("  Figure 3: Success rate …")
    n_env = len(envs)
    fig, axes = plt.subplots(1, n_env, figsize=(5 * n_env, 5), sharey=True)
    if n_env == 1:
        axes = [axes]

    fig.suptitle("Success Rate — Mean of Last 5 Eval Checkpoints (±95% CI)",
                 fontsize=13, fontweight="bold")

    for ax, env in zip(axes, envs):
        for i, mode in enumerate(MODES):
            seeds = discover_seeds(results_dir, env, mode)
            runs  = load_eval_csvs(results_dir, env, mode, seeds)
            fs    = _final_success(runs)
            if len(fs) == 0:
                continue
            ax.bar(i, np.mean(fs) * 100, color=MODE_COLOURS[mode],
                   width=0.6, alpha=0.85, zorder=2,
                   hatch="///" if mode in ABLATION_MODES else "")
            ax.errorbar(i, np.mean(fs) * 100, yerr=_ci95(fs) * 100,
                        fmt="none", color="black", capsize=5, linewidth=1.5)

        labels = [MODE_LABELS[m].split(":")[0] for m in MODES]
        ax.set_xticks(range(len(MODES)))
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_title(ENV_DISPLAY[env], fontsize=11, fontweight="bold")
        ax.set_ylabel("Success Rate (%)", fontsize=10)
        ax.set_ylim(0, 110)
        ax.axhline(100, color="gray", linewidth=0.8, linestyle=":")

    plt.tight_layout()
    path = os.path.join(output_dir, "fig3_success_rate.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    → {path}")


# ── Figures 4-6: AC-PPO cognitive metrics over training ───────────────────────

def _plot_ac_metric(results_dir, output_dir, envs, csv_col, ylabel, title, fname,
                    source="eval", smooth_w=1):
    """Generic plotter for AC-PPO specific metrics."""
    print(f"  {title} …")
    fig, axes = plt.subplots(1, len(envs), figsize=(6 * len(envs), 4.5), sharey=False)
    if len(envs) == 1:
        axes = [axes]

    plotted_any = False
    for ax, env in zip(axes, envs):
        for mode in ["full"]:    # AC-PPO only
            seeds = discover_seeds(results_dir, env, mode)
            if source == "eval":
                runs = load_eval_csvs(results_dir, env, mode, seeds)
            else:
                runs = load_train_csvs(results_dir, env, mode, seeds)
            if not runs:
                continue

            mean, lo, hi = _agg_eval(runs, csv_col)
            if mean is None:
                continue

            n = len(mean)
            if source == "eval":
                x = [(i + 1) * 100 for i in range(n)]
            else:
                x = list(range(1, n + 1))
                if smooth_w > 1:
                    mean = _smooth(mean, smooth_w)
                    lo   = _smooth(lo,   smooth_w)
                    hi   = _smooth(hi,   smooth_w)
                    x    = x[:len(mean)]

            ax.plot(x, mean, color=MODE_COLOURS[mode], linewidth=2.2, zorder=3,
                    label=MODE_LABELS[mode])
            ax.fill_between(x, lo, hi, color=MODE_COLOURS[mode], alpha=0.15)
            plotted_any = True

        ax.set_title(ENV_DISPLAY[env], fontsize=11, fontweight="bold")
        ax.set_xlabel("Episode", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(fontsize=8)

    if not plotted_any:
        plt.close()
        return

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, fname)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    → {path}")


def fig4_beta_evolution(results_dir, output_dir, envs):
    _plot_ac_metric(results_dir, output_dir, envs,
                    csv_col="avg_beta", ylabel="Average β_t",
                    title="Adaptive Cognitive Influence (β_t) over Training — B3: AC-PPO",
                    fname="fig4_beta_evolution.png",
                    source="eval")


def fig5_divergence(results_dir, output_dir, envs):
    _plot_ac_metric(results_dir, output_dir, envs,
                    csv_col="avg_divergence", ylabel="Avg JS Divergence",
                    title="Policy Divergence (JS) over Training — B3: AC-PPO",
                    fname="fig5_divergence.png",
                    source="eval")


def fig6_violation_rate(results_dir, output_dir, envs):
    _plot_ac_metric(results_dir, output_dir, envs,
                    csv_col="avg_violation", ylabel="Avg Violation Rate",
                    title="Affective Constraint Violation Rate over Training — B3: AC-PPO",
                    fname="fig6_violation_rate.png",
                    source="eval")


# ── Table 2: Final performance ─────────────────────────────────────────────────

def table2_final_performance(results_dir: str, output_dir: str, envs: list[str]) -> None:
    print("  Table 2: Final performance …")
    rows = []
    for env in envs:
        for mode in MODES:
            seeds = discover_seeds(results_dir, env, mode)
            runs  = load_eval_csvs(results_dir, env, mode, seeds)
            fr    = _final_rewards(runs)
            if len(fr) == 0:
                continue
            mean = float(np.mean(fr))
            std  = float(np.std(fr, ddof=1)) if len(fr) > 1 else 0.0
            best = float(np.max(fr))
            ci   = _ci95(fr)
            rows.append({
                "environment":    env,
                "method_id":      MODE_LABELS[mode].split(":")[0],
                "method_name":    MODE_LABELS[mode],
                "n_seeds":        len(fr),
                "mean":           round(mean, 3),
                "std":            round(std,  3),
                "best":           round(best, 3),
                "ci95_low":       round(mean - ci, 3),
                "ci95_high":      round(mean + ci, 3),
            })
    path = os.path.join(output_dir, "table2_final_performance.csv")
    if rows:
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"    → {path}")


# ── Table 3: Statistical tests ─────────────────────────────────────────────────

def table3_statistics(results_dir: str, output_dir: str, envs: list[str]) -> None:
    print("  Table 3: Statistical tests …")
    rows = []
    comparisons = [
        ("full", "baseline",      "B3 vs B1"),
        ("full", "affect_reward", "B3 vs B2"),
        ("full", "appraisal_only","B3 vs A1"),
    ]
    for env in envs:
        for (mode_a, mode_b, label) in comparisons:
            seeds_a = discover_seeds(results_dir, env, mode_a)
            seeds_b = discover_seeds(results_dir, env, mode_b)
            runs_a  = load_eval_csvs(results_dir, env, mode_a, seeds_a)
            runs_b  = load_eval_csvs(results_dir, env, mode_b, seeds_b)
            fa = _final_rewards(runs_a)
            fb = _final_rewards(runs_b)
            if len(fa) < 2 or len(fb) < 2:
                continue
            res = run_stats(fa, fb, MODE_LABELS[mode_a], MODE_LABELS[mode_b])
            res["environment"]  = env
            res["comparison"]   = label
            rows.append(res)

    path = os.path.join(output_dir, "table3_statistics.csv")
    if rows:
        all_keys = ["environment", "comparison"] + [k for k in rows[0] if k not in
                    ("environment", "comparison")]
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=all_keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
    print(f"    → {path}")


def fig7_beta_vs_divergence(results_dir: str, output_dir: str, envs: list[str]) -> None:
    """Figure 7: Adaptive Influence (beta_t) vs Policy Divergence (D_t)."""
    print("  Figure 7: beta_t vs Divergence (D_t) …")
    fig, axes = plt.subplots(1, len(envs), figsize=(5.5 * len(envs), 4.5), sharey=True)
    if len(envs) == 1:
        axes = [axes]

    plotted_any = False
    for ax, env in zip(axes, envs):
        seeds = discover_seeds(results_dir, env, "full")
        runs  = load_eval_csvs(results_dir, env, "full", seeds)
        all_d, all_b = [], []
        for r in runs:
            d = r.get("avg_divergence", np.array([]))
            b = r.get("avg_beta",       np.array([]))
            mask = ~np.isnan(d) & ~np.isnan(b)
            all_d.extend(d[mask].tolist())
            all_b.extend(b[mask].tolist())

        if len(all_d) == 0:
            continue

        ax.scatter(all_d, all_b, color="#2ca02c", alpha=0.6, edgecolors="none", s=25)
        # Theoretical sigmoid curve: beta(D) = beta_max * sigmoid(k * (D - tau))
        d_grid = np.linspace(0, max(max(all_d), 0.3), 100)
        sig_grid = 0.30 * (1.0 / (1.0 + np.exp(-10.0 * (d_grid - 0.10))))
        ax.plot(d_grid, sig_grid, color="darkred", linestyle="--", linewidth=1.8, label=r"Theoretical $\beta(D_t)$")

        ax.set_title(ENV_DISPLAY[env], fontsize=11, fontweight="bold")
        ax.set_xlabel(r"JS Divergence ($D_t$)", fontsize=10)
        ax.set_ylabel(r"Adaptive Influence ($\beta_t$)", fontsize=10)
        ax.legend(fontsize=8, loc="upper left")
        ax.axvline(0.10, color="gray", linestyle=":", alpha=0.7, label=r"$\tau = 0.10$")
        plotted_any = True

    if not plotted_any:
        plt.close()
        return

    fig.suptitle(r"Adaptive Cognitive Influence ($\beta_t$) vs. Policy Divergence ($D_t$)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "fig7_beta_vs_divergence.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    → {path}")


def fig8_beta_vs_reward(results_dir: str, output_dir: str, envs: list[str]) -> None:
    """Figure 8: Adaptive Influence (beta_t) vs Evaluation Reward."""
    print("  Figure 8: beta_t vs Evaluation Reward …")
    fig, axes = plt.subplots(1, len(envs), figsize=(5.5 * len(envs), 4.5))
    if len(envs) == 1:
        axes = [axes]

    plotted_any = False
    for ax, env in zip(axes, envs):
        seeds = discover_seeds(results_dir, env, "full")
        runs  = load_eval_csvs(results_dir, env, "full", seeds)
        all_b, all_r = [], []
        for r in runs:
            b = r.get("avg_beta",     np.array([]))
            rew = r.get("eval_reward", np.array([]))
            mask = ~np.isnan(b) & ~np.isnan(rew)
            all_b.extend(b[mask].tolist())
            all_r.extend(rew[mask].tolist())

        if len(all_b) == 0:
            continue

        ax.scatter(all_b, all_r, color="#2ca02c", alpha=0.6, edgecolors="none", s=25)
        if len(all_b) > 2:
            m, b_int = np.polyfit(all_b, all_r, 1)
            b_grid = np.linspace(min(all_b), max(all_b), 50)
            ax.plot(b_grid, m * b_grid + b_int, color="darkblue", linestyle="-.", linewidth=1.5, label="Linear Trend")

        ax.set_title(ENV_DISPLAY[env], fontsize=11, fontweight="bold")
        ax.set_xlabel(r"Adaptive Influence ($\beta_t$)", fontsize=10)
        ax.set_ylabel("Evaluation Reward", fontsize=10)
        ax.legend(fontsize=8, loc="best")
        plotted_any = True

    if not plotted_any:
        plt.close()
        return

    fig.suptitle(r"Adaptive Cognitive Influence ($\beta_t$) vs. Task Evaluation Reward", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "fig8_beta_vs_reward.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    → {path}")


# ── Additional AC-PPO correlation analysis ─────────────────────────────────────

def additional_analyses(results_dir: str, output_dir: str, envs: list[str]) -> None:
    print("  Additional AC-PPO analyses (correlations between beta, divergence, reward) …")
    rows = []
    for env in envs:
        seeds = discover_seeds(results_dir, env, "full")
        runs  = load_eval_csvs(results_dir, env, "full", seeds)
        divs, betas, rewards = [], [], []
        for r in runs:
            d = r.get("avg_divergence", np.array([]))
            b = r.get("avg_beta",       np.array([]))
            e = r.get("eval_reward",    np.array([]))
            mask = ~np.isnan(d) & ~np.isnan(b) & ~np.isnan(e)
            divs.extend(d[mask].tolist())
            betas.extend(b[mask].tolist())
            rewards.extend(e[mask].tolist())

        if len(divs) < 3:
            continue

        try:
            from scipy.stats import pearsonr
            corr_dr, p_dr = pearsonr(divs, rewards)
            corr_bd, p_bd = pearsonr(betas, divs)
            corr_br, p_br = pearsonr(betas, rewards)
            rows.append({
                "environment": env,
                "corr_D_reward": round(corr_dr, 4), "p_D_reward": round(p_dr, 4),
                "corr_beta_D": round(corr_bd, 4),    "p_beta_D": round(p_bd, 4),
                "corr_beta_reward": round(corr_br, 4), "p_beta_reward": round(p_br, 4),
                "n_points": len(divs)
            })
        except ImportError:
            corr_dr = np.corrcoef(divs, rewards)[0, 1]
            corr_bd = np.corrcoef(betas, divs)[0, 1]
            corr_br = np.corrcoef(betas, rewards)[0, 1]
            rows.append({
                "environment": env,
                "corr_D_reward": round(corr_dr, 4),
                "corr_beta_D": round(corr_bd, 4),
                "corr_beta_reward": round(corr_br, 4),
                "n_points": len(divs)
            })

    path = os.path.join(output_dir, "additional_acppo_correlations.csv")
    if rows:
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"    → {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="AC-PPO Results Analysis: generate all figures and tables",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--results_dir", default="results",
                   help="Directory produced by run_all_envs.py")
    p.add_argument("--output_dir",  default="analysis",
                   help="Where to save figures and tables")
    p.add_argument("--envs", nargs="+", default=ALL_ENVS,
                   help="Environments to analyse")
    p.add_argument("--skip_figs",   nargs="*", default=[],
                   help="Figure numbers to skip, e.g. --skip_figs 4 5 6")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  AC-PPO Analysis Suite")
    print(f"  Results dir : {args.results_dir}")
    print(f"  Output dir  : {args.output_dir}")
    print(f"  Environments: {args.envs}")
    print(f"{'='*65}\n")

    skip = {int(x) for x in args.skip_figs}

    if 1 not in skip:
        fig1_learning_curves(args.results_dir, args.output_dir, args.envs)
    if 2 not in skip:
        fig2_final_performance(args.results_dir, args.output_dir, args.envs)
    if 3 not in skip:
        fig3_success_rate(args.results_dir, args.output_dir, args.envs)
    if 4 not in skip:
        fig4_beta_evolution(args.results_dir, args.output_dir, args.envs)
    if 5 not in skip:
        fig5_divergence(args.results_dir, args.output_dir, args.envs)
    if 6 not in skip:
        fig6_violation_rate(args.results_dir, args.output_dir, args.envs)
    if 7 not in skip:
        fig7_beta_vs_divergence(args.results_dir, args.output_dir, args.envs)
    if 8 not in skip:
        fig8_beta_vs_reward(args.results_dir, args.output_dir, args.envs)

    table2_final_performance(args.results_dir, args.output_dir, args.envs)
    table3_statistics(args.results_dir, args.output_dir, args.envs)
    additional_analyses(args.results_dir, args.output_dir, args.envs)

    print(f"\n  Done. All outputs saved to: {args.output_dir}/\n")


if __name__ == "__main__":
    main()
