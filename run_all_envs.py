"""
Usage:
    python run_all_envs.py                             # all 3 envs, 2000 eps, 6 workers
    python run_all_envs.py --episodes 300              # quick smoke test
    python run_all_envs.py --episodes 1000 --seeds 42 1 2
    python run_all_envs.py --envs CartPole-v1 Acrobot-v1  # subset
    python run_all_envs.py --workers 6                 # explicit parallelism (default)

Structure
---------
  Main baselines  : B1 Standard PPO | B2 Affect-as-Reward | B3 AC-PPO (proposed)
  Ablation study  : A1 AC-PPO without affective constraint

Parallelism
-----------
  All (env × mode × seed) combinations are submitted at once to a
  ProcessPoolExecutor.  With --workers 6 and 120 total runs the wall-time
  drops to roughly 1/6 of the sequential time.
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from train_full_model import FullConfig, train

# ── Constants ────────────────────────────────────────────────────────────────

# Ordered by environment complexity (easy → hard)
ALL_ENVS = [
    "CartPole-v1",    # simplest  — continuous reward, easy exploration
    "Acrobot-v1",     # moderate  — sparse reward, needs momentum
    "LunarLander-v3", # hard      — dense-but-shaped, multi-objective
]

# ── Mode grouping ─────────────────────────────────────────────────────────────
# Main baselines run first (B1, B2, B3); ablation study appended after.
MAIN_MODES     = ["baseline", "affect_reward", "full"]
ABLATION_MODES = ["appraisal_only"]
MODES          = MAIN_MODES + ABLATION_MODES   # ordering: B1, B2, B3, A1

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

# Solid for main baselines, dashed for ablation — visual distinction in plots
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


# ── Runner helpers ────────────────────────────────────────────────────────────

def _worker_init():
    """Called once per worker process.
    Limit torch to 1 intra-op thread so N workers don't saturate all cores.
    NOTE: set_num_interop_threads cannot be called after torch starts,
    so we rely on OMP_NUM_THREADS (set in main before pool spawn) instead."""
    torch.set_num_threads(1)


def _make_cfg(env_name: str, mode: str, seed: int,
              episodes: int, base_cfg: FullConfig, env_dir: str) -> FullConfig:
    """Build a FullConfig for a single (env, mode, seed) run."""
    return FullConfig(
        env_name        = env_name,
        seed            = seed,
        mode            = mode,
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
        save_dir        = env_dir,
        run_name        = f"{mode}_s{seed}",
        results_dir     = os.path.join(env_dir, "csv"),
        device          = base_cfg.device,
        render          = False,
    )


def _run_one(cfg: FullConfig) -> tuple:
    """Top-level picklable function executed in a worker process.
    Returns (env_name, mode, history).
    """
    # Limit intra-op threads (set_num_interop_threads cannot be called here
    # because torch may already have initialized its thread pool on import)
    torch.set_num_threads(1)
    history = train(cfg)
    return cfg.env_name, cfg.mode, history


# ── Runner ────────────────────────────────────────────────────────────────────

def run_all_envs(
    envs:     list[str],
    episodes: int,
    seeds:    list[int],
    base_cfg: FullConfig,
    save_dir: str,
    workers:  int = 6,
) -> dict:
    """Run all (env × mode × seed) combinations in parallel.

    Parameters
    ----------
    workers : int
        Number of parallel processes.  Set to 1 for sequential (debug) mode.
    """
    results = {env: {mode: [] for mode in MODES} for env in envs}

    # Pre-create output directories (must happen in main process)
    env_dirs = {}
    for env_name in envs:
        env_dir = os.path.join(save_dir, env_name.replace("-", "_"))
        os.makedirs(env_dir, exist_ok=True)
        os.makedirs(os.path.join(env_dir, "csv"), exist_ok=True)
        env_dirs[env_name] = env_dir

    # Build the flat list of configs (one per run)
    configs = [
        _make_cfg(env_name, mode, seed, episodes, base_cfg, env_dirs[env_name])
        for env_name in envs
        for mode in MODES
        for seed in seeds
    ]

    total    = len(configs)
    done     = 0
    t_launch = time.time()

    print(f"\n  Launching {total} runs across {workers} workers …")

    if workers == 1:
        # Sequential fallback (useful for debugging / profiling)
        for cfg in configs:
            env_name, mode, history = _run_one(cfg)
            results[env_name][mode].append(history)
            done += 1
            elapsed = time.time() - t_launch
            eta     = (elapsed / done) * (total - done)
            print(f"  [{done}/{total}]  {ENV_DISPLAY.get(env_name, env_name)}  "
                  f"|  {MODE_LABELS[mode]}  |  seed={cfg.seed}  "
                  f"(elapsed {elapsed:.0f}s  ETA {eta:.0f}s)")
    else:
        # Tell OpenMP / MKL to use 1 thread per process.
        # Must be set before the pool spawns (children inherit the env).
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_worker_init,
        ) as pool:
            future_to_cfg = {pool.submit(_run_one, cfg): cfg for cfg in configs}

            for future in as_completed(future_to_cfg):
                cfg = future_to_cfg[future]
                try:
                    env_name, mode, history = future.result()
                except Exception as exc:
                    print(f"  ✗  {cfg.env_name} | {cfg.mode} | seed={cfg.seed}  "
                          f"FAILED: {exc}")
                    history = {}
                    env_name, mode = cfg.env_name, cfg.mode

                results[env_name][mode].append(history)
                done += 1
                elapsed = time.time() - t_launch
                eta     = (elapsed / done) * (total - done) if done < total else 0
                print(f"  [{done}/{total}]  ✓  {ENV_DISPLAY.get(env_name, env_name)}  "
                      f"|  {MODE_LABELS[mode]}  |  seed={cfg.seed}  "
                      f"(elapsed {elapsed:.0f}s  ETA {eta:.0f}s)")

    return results


# ── Aggregation ───────────────────────────────────────────────────────────────

def _agg(runs: list[dict], key: str):
    """Return (mean, lo, hi) across seeds, truncated to shortest run."""
    arrays = [np.array(r.get(key, [])) for r in runs if r.get(key)]
    if not arrays:
        return np.array([]), np.array([]), np.array([])
    n = min(len(a) for a in arrays)
    s = np.stack([a[:n] for a in arrays])
    m = s.mean(0); sd = s.std(0)
    return m, m - sd, m + sd

def _smooth(v: list, w: int = 20):
    if len(v) < w: return v
    return np.convolve(v, np.ones(w)/w, mode="valid").tolist()


# ── Summary table ─────────────────────────────────────────────────────────────

def _mode_row(mode: str, runs: list, lines: list) -> None:
    """Append one formatted row for a single mode into lines."""
    rewards = [r.get("eval_reward", []) for r in runs if r.get("eval_reward")]
    succs   = [r.get("eval_success", []) for r in runs if r.get("eval_success")]
    if not rewards:
        lines.append(f"  {MODE_LABELS[mode]:<34} {'—':>12}")
        return
    all_r  = np.concatenate(rewards)
    mean_s = np.concatenate(succs).mean() if succs else float("nan")
    lines.append(
        f"  {MODE_LABELS[mode]:<34} "
        f"{all_r.mean():>+12.3f} "
        f"{all_r.std():>8.3f} "
        f"{all_r.max():>+10.3f} "
        f"{mean_s:>8.2f}"
    )


def print_summary(results: dict, episodes: int, save_dir: str) -> None:
    lines = []
    lines.append(f"\n{'='*84}")
    lines.append(f"  CROSS-ENVIRONMENT SUMMARY  ({episodes} episodes)")
    lines.append(f"{'='*84}")

    HDR = f"  {'Mode':<34} {'Mean Eval':>12} {'Std':>8} {'Best':>10} {'Succ':>8}"

    for env_name in results:
        lines.append(f"\n  {ENV_DISPLAY.get(env_name, env_name)}")

        # ── Main Baselines ─────────────────────────────────────────────────
        lines.append(f"  {'─'*78}")
        lines.append(f"  Main Baselines")
        lines.append(f"  {'─'*78}")
        lines.append(HDR)
        lines.append(f"  {'─'*78}")
        for mode in MAIN_MODES:
            _mode_row(mode, results[env_name][mode], lines)

        # ── Ablation Study ─────────────────────────────────────────────────
        lines.append(f"  {'·'*78}")
        lines.append(f"  Ablation Study")
        lines.append(f"  {'·'*78}")
        lines.append(HDR)
        lines.append(f"  {'·'*78}")
        for mode in ABLATION_MODES:
            _mode_row(mode, results[env_name][mode], lines)

    lines.append(f"\n{'='*84}\n")
    text = "\n".join(lines)
    print(text)

    path = os.path.join(save_dir, "summary_table.txt")
    with open(path, "w") as f:
        f.write(text)
    print(f"  Summary saved → {path}")


# ── Cross-environment comparison plot ─────────────────────────────────────────

def plot_cross_env(
    results:       dict,
    save_dir:      str,
    episodes:      int,
    eval_interval: int,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not available — skipping plot)")
        return

    envs    = list(results.keys())
    metrics = [
        ("ep_reward",   "Episode Reward",    False),
        ("eval_reward", "Eval Mean Reward",  True),
        ("divergence",  "JS Divergence D_t", False),
    ]

    n_rows = len(envs)
    n_cols = len(metrics)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(6 * n_cols, 4 * n_rows),
        squeeze=False,
    )
    fig.suptitle(
        f"AC-PPO — Cross-Environment Comparison  ({episodes} episodes)",
        fontsize=14, fontweight="bold", y=1.01,
    )

    for r, env_name in enumerate(envs):
        for c, (key, title, is_eval) in enumerate(metrics):
            ax = axes[r][c]

            if c == 0:
                ax.set_ylabel(ENV_DISPLAY.get(env_name, env_name),
                              fontsize=11, fontweight="bold")
            if r == 0:
                ax.set_title(title, fontsize=10)

            ax.tick_params(labelsize=8)
            ax.set_xlabel("Episode", fontsize=8)

            any_data = False
            for mode in MODES:
                runs      = results[env_name][mode]
                colour    = MODE_COLOURS[mode]
                ls        = MODE_LINESTYLE[mode]   # solid for baselines, dashed for ablation
                label     = MODE_LABELS[mode]
                lw_main   = 1.8 if mode in MAIN_MODES else 1.4
                alpha_raw = 0.12 if mode in MAIN_MODES else 0.06

                mean, lo, hi = _agg(runs, key)
                if len(mean) == 0:
                    continue
                any_data = True

                if is_eval:
                    x = [i * eval_interval for i in range(len(mean))]
                    marker = "o" if mode in MAIN_MODES else "s"
                    ax.plot(x, mean, linestyle=ls, marker=marker,
                            color=colour, label=label,
                            markersize=3, linewidth=lw_main)
                    if len(runs) > 1:
                        ax.fill_between(x, lo, hi, alpha=0.10, color=colour)
                else:
                    x    = list(range(len(mean)))
                    smth = _smooth(mean.tolist())
                    x_sm = list(range(len(mean) - len(smth), len(mean)))
                    ax.plot(x, mean, color=colour, alpha=alpha_raw, linewidth=0.7)
                    ax.plot(x_sm, smth, color=colour, linestyle=ls,
                            label=label, linewidth=lw_main)
                    if len(runs) > 1:
                        ax.fill_between(
                            x_sm,
                            _smooth(lo.tolist()),
                            _smooth(hi.tolist()),
                            alpha=0.08, color=colour,
                        )

            if not any_data:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                        ha="center", va="center", color="gray")

    # Legend: main baselines (solid) then ablation (dashed)
    handles = []
    for m in MAIN_MODES:
        handles.append(plt.Line2D([0], [0], color=MODE_COLOURS[m],
                                  linewidth=2.5, linestyle="-",
                                  label=MODE_LABELS[m]))
    for m in ABLATION_MODES:
        handles.append(plt.Line2D([0], [0], color=MODE_COLOURS[m],
                                  linewidth=2.0, linestyle="--",
                                  label=MODE_LABELS[m] + "  [ablation]"))

    fig.legend(
        handles=handles, loc="lower center", ncol=len(MODES),
        fontsize=9, bbox_to_anchor=(0.5, -0.03), frameon=False,
    )

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "cross_env_comparison.png")
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"\n  Cross-environment plot saved → {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="AC-PPO: run all envs × all modes  (final experimental protocol)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--envs",     nargs="+",  default=ALL_ENVS,
                   choices=ALL_ENVS, metavar="ENV",
                   help=(f"Environments to run. Choices: {ALL_ENVS}"))
    p.add_argument("--episodes", type=int,   default=2_000,
                   help="Training episodes per run")
    p.add_argument("--seeds",    type=int,   nargs="+",
                   default=list(range(10)),
                   help="Random seeds (default: 0–9, 10 seeds)")
    p.add_argument("--workers",  type=int,   default=6,
                   help="Parallel worker processes (set to 1 for sequential/debug)")
    # ── Base PPO (SB3 locked — do not change for published results)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--rollout_steps", type=int,   default=1_024)
    p.add_argument("--log_interval",  type=int,   default=50)
    p.add_argument("--eval_interval", type=int,   default=100)
    # ── AC-PPO (FROZEN from two-stage tuning — do not change for published results)
    p.add_argument("--tau",         type=float, default=0.10)
    p.add_argument("--beta_max",    type=float, default=0.30)
    p.add_argument("--beta_r",      type=float, default=0.50)
    p.add_argument("--d",           type=float, default=0.20)
    p.add_argument("--lambda_coef", type=float, default=0.05)
    # ── Paths
    p.add_argument("--save_dir", type=str, default="results")
    p.add_argument("--device",   type=str, default="cpu")
    return p.parse_args()


def main():
    args = parse_args()

    base_cfg = FullConfig(
        lr              = args.lr,
        rollout_steps   = args.rollout_steps,
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

    total_runs = len(args.envs) * len(MODES) * len(args.seeds)
    print(f"\n{'='*65}")
    print(f"  AC-PPO  Cross-Environment Ablation Suite")
    print(f"  Environments : {args.envs}")
    print(f"  Modes        : {list(MODE_LABELS.values())}")
    print(f"  Episodes     : {args.episodes}  |  Seeds: {args.seeds}")
    print(f"  Total runs   : {total_runs}  |  Workers: {args.workers}")
    print(f"  [Frozen] τ={args.tau}  β_max={args.beta_max}  "
          f"d={args.d}  λ={args.lambda_coef}  rollout={args.rollout_steps}")
    print(f"{'='*65}")

    t0      = time.time()
    results = run_all_envs(
        envs     = args.envs,
        episodes = args.episodes,
        seeds    = args.seeds,
        base_cfg = base_cfg,
        save_dir = args.save_dir,
        workers  = args.workers,
    )
    elapsed = time.time() - t0

    print_summary(results, args.episodes, args.save_dir)
    plot_cross_env(results, args.save_dir, args.episodes, args.eval_interval)

    print(f"  Total wall time: {elapsed:.0f}s  ({elapsed/60:.1f} min)")
    print(f"  All outputs in: {args.save_dir}/\n")


if __name__ == "__main__":
    main()
