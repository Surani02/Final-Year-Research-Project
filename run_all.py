"""
Usage:
    python run_all.py                          # default 1000 episodes
    python run_all.py --episodes 300           # quick smoke test
    python run_all.py --episodes 1000 --seeds 42 1 2   # multi-seed
    python run_all.py --episodes 1000 --seed 42        # single seed (default)
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from train_full_model import FullConfig, train


# ============================================================================ #
#  Display labels for each ablation mode                                        #
# ============================================================================ #

MODE_LABELS = {
    "baseline":      "B1: Standard PPO",
    "affect_reward": "B2: Affect-as-Reward",
    "appraisal_only":"B3: Appraisal-PPO",
    "full":          "B4: AC-PPO (proposed)",
}

MODE_COLOURS = {
    "baseline":       "#888888",
    "affect_reward":  "#e07b39",
    "appraisal_only": "#5b8dd9",
    "full":           "#2ca02c",
}

MODES = list(MODE_LABELS.keys())


# ============================================================================ #
#  Multi-seed runner                                                            #
# ============================================================================ #

def run_all(
    episodes:   int,
    seeds:      list[int],
    base_cfg:   FullConfig,
    save_dir:   str,
) -> dict[str, list[dict]]:
    """
    Run every mode × every seed. Returns nested dict:
        results[mode][seed_idx] = history_dict
    """
    results: dict[str, list[dict]] = {m: [] for m in MODES}

    total_runs = len(MODES) * len(seeds)
    run_idx    = 0

    for mode in MODES:
        for seed in seeds:
            run_idx += 1
            label = MODE_LABELS[mode]
            print(f"\n{'─'*65}")
            print(f"  Run {run_idx}/{total_runs}:  {label}  |  seed={seed}")
            print(f"{'─'*65}")

            cfg = FullConfig(
                # copy all base settings
                env_name        = base_cfg.env_name,
                total_episodes  = episodes,
                rollout_steps   = base_cfg.rollout_steps,
                ppo_epochs      = base_cfg.ppo_epochs,
                mini_batch      = base_cfg.mini_batch,
                eval_interval   = base_cfg.eval_interval,
                eval_episodes   = base_cfg.eval_episodes,
                log_interval    = base_cfg.log_interval,
                lr              = base_cfg.lr,
                gamma           = base_cfg.gamma,
                gae_lambda      = base_cfg.gae_lambda,
                clip_eps        = base_cfg.clip_eps,
                value_coef      = base_cfg.value_coef,
                entropy_coef    = base_cfg.entropy_coef,
                max_grad_norm   = base_cfg.max_grad_norm,
                hidden_size     = base_cfg.hidden_size,
                mental_dim      = base_cfg.mental_dim,
                latent_dim      = base_cfg.latent_dim,
                appraisal_lr    = base_cfg.appraisal_lr,
                beta_r          = base_cfg.beta_r,
                alpha_g         = base_cfg.alpha_g,
                alpha_p         = base_cfg.alpha_p,
                alpha_ema       = base_cfg.alpha_ema,
                tau             = base_cfg.tau,
                k_slope         = base_cfg.k_slope,
                beta_max        = base_cfg.beta_max,
                d_threshold     = base_cfg.d_threshold,
                lambda_coef     = base_cfg.lambda_coef,
                adaptive_lambda = base_cfg.adaptive_lambda,
                # per-run settings
                mode            = mode,
                seed            = seed,
                save_dir        = save_dir,
                run_name        = f"{mode}_s{seed}",
                device          = base_cfg.device,
                render          = False,
            )

            history = train(cfg)
            results[mode].append(history)

    return results


# ============================================================================ #
#  Aggregation helpers                                                          #
# ============================================================================ #

def smooth(values: list[float], window: int = 20) -> list[float]:
    if len(values) < window:
        return values
    k = np.ones(window) / window
    return np.convolve(values, k, mode="valid").tolist()


def aggregate(runs: list[dict], key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arrays = [np.array(r.get(key, [])) for r in runs if r.get(key)]
    if not arrays:
        return np.array([]), np.array([]), np.array([])
    min_len = min(len(a) for a in arrays)
    stacked = np.stack([a[:min_len] for a in arrays])
    mean    = stacked.mean(axis=0)
    std     = stacked.std(axis=0)
    return mean, mean - std, mean + std


# ============================================================================ #
#  Summary table                                                                #
# ============================================================================ #

def print_summary(results: dict[str, list[dict]], episodes: int) -> None:
    eval_key = "eval_reward"

    print(f"\n{'='*75}")
    print(f"  COMPARISON SUMMARY  ({episodes} episodes)")
    print(f"{'='*75}")
    print(f"  {'Mode':<30} {'Mean Eval Reward':>18} {'Std':>8} {'Best Eval':>10} {'Succ Rate':>10}")
    print(f"  {'-'*70}")

    for mode in MODES:
        runs    = results[mode]
        rewards = [r.get(eval_key, []) for r in runs if r.get(eval_key)]
        succs   = [r.get("eval_success", []) for r in runs if r.get("eval_success")]

        if not rewards:
            print(f"  {MODE_LABELS[mode]:<30} {'—':>18}")
            continue

        all_r = np.concatenate(rewards)
        mean_r = all_r.mean()
        std_r  = all_r.std()
        best_r = all_r.max()

        mean_s = np.concatenate(succs).mean() if succs else float("nan")

        print(
            f"  {MODE_LABELS[mode]:<30} "
            f"{mean_r:>+18.3f} "
            f"{std_r:>8.3f} "
            f"{best_r:>+10.3f} "
            f"{mean_s:>10.2f}"
        )

    print(f"{'='*75}\n")


# ============================================================================ #
#  Comparison plot                                                              #
# ============================================================================ #

def plot_comparison(
    results:   dict[str, list[dict]],
    save_dir:  str,
    episodes:  int,
    eval_interval: int,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("  (matplotlib not found — skipping comparison plot)")
        return

    # Metrics to plot: (history_key, title, is_eval)
    panels = [
        ("ep_reward",      "Episode Reward",         False),
        ("goal_rate",      "Goal Success Rate",       False),
        ("divergence",     "JS Divergence D_t",       False),
        ("beta",           "Influence β_t",           False),
        ("affective_cost", "Affective Cost C_t",      False),
        ("eval_reward",    "Eval Mean Reward",        True),
    ]

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(
        f"Ablation Comparison — {episodes} episodes  "
        f"({'×'.join(str(s) for s in sorted({r for runs in results.values() for r in [len(runs)]}))} seeds)",
        fontsize=13, fontweight="bold"
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.32)

    for idx, (key, title, is_eval) in enumerate(panels):
        row, col = divmod(idx, 3)
        ax = fig.add_subplot(gs[row, col])
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Episode", fontsize=8)
        ax.tick_params(labelsize=8)

        has_data = False
        for mode in MODES:
            runs   = results[mode]
            colour = MODE_COLOURS[mode]
            label  = MODE_LABELS[mode]

            mean, lo, hi = aggregate(runs, key)
            if len(mean) == 0:
                continue

            has_data = True

            if is_eval:
                x = [i * eval_interval for i in range(len(mean))]
                ax.plot(x, mean, "o-", color=colour, label=label,
                        markersize=3, linewidth=1.5)
                if len(runs) > 1:
                    ax.fill_between(x, lo, hi, color=colour, alpha=0.15)
            else:
                x    = list(range(len(mean)))
                smth = smooth(mean.tolist(), window=20)
                x_sm = list(range(len(mean) - len(smth), len(mean)))

                ax.plot(x, mean, color=colour, alpha=0.15, linewidth=0.8)
                ax.plot(x_sm, smth, color=colour, label=label, linewidth=1.8)
                if len(runs) > 1:
                    sm_lo = smooth(lo.tolist(), window=20)
                    sm_hi = smooth(hi.tolist(), window=20)
                    ax.fill_between(x_sm, sm_lo, sm_hi, color=colour, alpha=0.12)

        if has_data and idx == 0:
            ax.legend(fontsize=7, loc="lower right")

    # Shared legend below the plot
    handles = [
        plt.Line2D([0], [0], color=MODE_COLOURS[m], linewidth=2, label=MODE_LABELS[m])
        for m in MODES
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
    )

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "comparison_all_modes.png")
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"\n  Comparison plot saved → {path}")


# ============================================================================ #
#  CLI                                                                          #
# ============================================================================ #

def parse_args():
    p = argparse.ArgumentParser(description="Run all AC-PPO ablation modes")
    p.add_argument("--env",       type=str,   default="LunarLander-v3",
                   choices=["CartPole-v1", "LunarLander-v3", "MountainCar-v0"])
    p.add_argument("--episodes",  type=int,   default=1_000)
    p.add_argument("--seeds",     type=int,   nargs="+", default=[42],
                   help="One or more random seeds, e.g. --seeds 42 1 2")
    p.add_argument("--lr",        type=float, default=3e-4)
    p.add_argument("--tau",       type=float, default=0.2)
    p.add_argument("--beta_max",  type=float, default=0.5)
    p.add_argument("--beta_r",    type=float, default=0.5)
    p.add_argument("--d",         type=float, default=0.3)
    p.add_argument("--lambda_coef",type=float,default=0.1)
    p.add_argument("--log_interval",type=int, default=50)
    p.add_argument("--eval_interval",type=int,default=100)
    p.add_argument("--save_dir",  type=str,   default="results")
    p.add_argument("--device",    type=str,   default="cpu")
    return p.parse_args()


def main():
    args = parse_args()

    base_cfg = FullConfig(
        env_name        = args.env,
        lr              = args.lr,
        tau             = args.tau,
        beta_max        = args.beta_max,
        beta_r          = args.beta_r,
        d_threshold     = args.d,
        lambda_coef     = args.lambda_coef,
        log_interval    = args.log_interval,
        eval_interval   = args.eval_interval,
        save_dir        = args.save_dir,
        device          = args.device,
    )

    print(f"\n{'='*65}")
    print(f"  AC-PPO Ablation Suite")
    print(f"  Environment: {args.env}")
    print(f"  Modes:    {' | '.join(MODE_LABELS.values())}")
    print(f"  Episodes: {args.episodes}  |  Seeds: {args.seeds}")
    print(f"{'='*65}")

    t0      = time.time()
    results = run_all(args.episodes, args.seeds, base_cfg, args.save_dir)
    elapsed = time.time() - t0

    print_summary(results, args.episodes)

    plot_comparison(results, args.save_dir, args.episodes, args.eval_interval)

    print(f"  Total wall time: {elapsed:.0f}s  ({elapsed/60:.1f} min)")
    print(f"  All results in: {args.save_dir}/\n")


if __name__ == "__main__":
    main()
