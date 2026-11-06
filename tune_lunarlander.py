"""
Two-stage tuning strategy:
  Stage 1 — Random search over all 5 AC-PPO hyperparameters (short runs, 1 seed)
             to identify promising regions of the space.
  Stage 2 — Confirmation: retrain top-K Stage-1 configurations with longer
             episodes and multiple seeds for publication-quality statistics.
"""

import argparse
import itertools
import json
import os
import random
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from train_full_model import FullConfig, train

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ── Base PPO parameters (SB3 / Schulman-2017 defaults) ────────────────────────
# These remain IDENTICAL across B1, B2, B3, A1 — never tuned.
BASE_PPO = dict(
    env_name       = "LunarLander-v3",
    mode           = "full",        # B3: AC-PPO (proposed)
    rollout_steps  = 1024,          # SB3 LunarLander default
    ppo_epochs     = 4,
    mini_batch     = 64,
    lr             = 3e-4,
    gamma          = 0.99,
    gae_lambda     = 0.95,
    clip_eps       = 0.2,
    value_coef     = 0.5,
    entropy_coef   = 0.01,
    max_grad_norm  = 0.5,
    eval_episodes  = 10,
)

# ── AC-PPO hyperparameter search space ────────────────────────────────────────
PARAM_GRID = {
    "tau":         [0.01, 0.05, 0.10, 0.15, 0.20],
    "beta_max":    [0.30, 0.50, 0.70],
    "d_threshold": [0.05, 0.10, 0.20],
    "lambda_coef": [0.05, 0.10, 0.20],
    "k_slope":     [5.0, 10.0, 15.0],
}

PARAM_DISPLAY = {
    "tau":         "τ (conflict threshold)",
    "beta_max":    "β_max (max cognitive influence)",
    "d_threshold": "d (affective cost tolerance)",
    "lambda_coef": "λ (constraint weight)",
    "k_slope":     "k (sigmoid slope)",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_mean(lst: list) -> float:
    arr = [x for x in lst if x is not None]
    return float(np.mean(arr)) if arr else float("nan")

def _final_third_mean(arr: np.ndarray) -> float:
    """Mean of the final third of an array — emphasises late-training performance.
    Falls back to the full mean when fewer than 3 eval checkpoints exist."""
    if len(arr) == 0:
        return float("nan")
    if len(arr) < 3:
        return float(np.mean(arr))
    cut   = int(len(arr) * 0.66)
    final = arr[cut:]
    return float(np.mean(final)) if len(final) > 0 else float(np.mean(arr))

def _last_n_checkpoints_mean(per_seed_arrays: list, n: int = 5) -> float:
    """Mean reward over the last `n` evaluation checkpoints, averaged across seeds.
    Each seed's eval history may have a different number of checkpoints, so the
    last `n` (or fewer, if a run has less) are taken per-seed before combining."""
    tails = []
    for arr in per_seed_arrays:
        arr = np.asarray(arr)
        if len(arr) == 0:
            continue
        tails.append(arr[-n:])
    if not tails:
        return float("nan")
    return float(np.mean(np.concatenate(tails)))

def _generate_trials(method: str, num_samples: int, seed: int = 42) -> list[dict]:
    keys = list(PARAM_GRID.keys())
    rng  = random.Random(seed)
    if method == "grid":
        combos = list(itertools.product(*[PARAM_GRID[k] for k in keys]))
        if len(combos) > num_samples:
            combos = rng.sample(combos, num_samples)
        return [dict(zip(keys, c)) for c in combos]
    else:   # random
        seen, trials = set(), []
        for _ in range(num_samples * 10):
            t = tuple(rng.choice(PARAM_GRID[k]) for k in keys)
            if t not in seen:
                seen.add(t)
                trials.append(dict(zip(keys, t)))
            if len(trials) == num_samples:
                break
        return trials


def _run_one_seed(params: dict, episodes: int, seed: int,
                   save_dir: str, device: str, trial_id: int) -> dict:
    """Run a single (config, seed) pair. Standalone/picklable so it can be
    dispatched to a worker process by ProcessPoolExecutor."""
    trial_dir = os.path.join(save_dir, f"trial_{trial_id}")
    eval_interval = 100  # fixed

    cfg = FullConfig(
        **BASE_PPO,
        seed=seed,
        total_episodes=episodes,
        eval_interval=eval_interval,
        save_dir=trial_dir,
        run_name=f"trial_{trial_id}_s{seed}",
        device=device,
        tau=params["tau"],
        beta_max=params["beta_max"],
        d_threshold=params["d_threshold"],
        lambda_coef=params["lambda_coef"],
        k_slope=params["k_slope"],
    )
    history = train(cfg)
    return {
        "eval_reward":    history.get("eval_reward", []),
        "eval_success":   history.get("eval_success", []),
        "divergence":     history.get("divergence", []),
        "beta":           history.get("beta", []),
        "violation_rate": history.get("violation_rate", []),
    }


def _aggregate_seed_histories(seed_histories: list[dict], params: dict,
                               seeds: list[int], trial_id: int) -> dict | None:
    """Combine per-seed histories (as returned by _run_one_seed) into the
    same summary record previously produced inline by _run_one."""
    eval_rewards, eval_succs = [], []
    avg_divs, avg_betas, avg_viols = [], [], []

    for history in seed_histories:
        er = history.get("eval_reward", [])
        es = history.get("eval_success", [])
        dv = history.get("divergence", [])
        bt = history.get("beta", [])
        vr = history.get("violation_rate", [])

        if er:
            eval_rewards.append(er)
        if es:
            eval_succs.append(es)
        if dv:
            avg_divs.append(float(np.mean(dv)))
        if bt:
            avg_betas.append(float(np.mean(bt)))
        if vr:
            avg_viols.append(float(np.mean(vr)))

    if not eval_rewards:
        return None

    all_r = np.concatenate(eval_rewards)
    final_mean_eval = round(_last_n_checkpoints_mean(eval_rewards, n=5), 3)
    std_eval        = round(float(np.std(all_r)), 3)
    score           = round(final_mean_eval - 0.10 * std_eval, 3)
    return {
        "trial_id":        trial_id,
        "params":          params,
        "mean_eval":       round(float(np.mean(all_r)), 3),
        "final_mean_eval": final_mean_eval,   # mean over last 5 eval checkpoints (per seed, pooled)
        "std_eval":        std_eval,
        "score":           score,             # final_mean_eval - 0.10 * std_eval
        "best_eval":       round(float(np.max(all_r)), 3),
        "succ_rate":       round(float(np.mean(np.concatenate(eval_succs))), 3)
                           if eval_succs else float("nan"),
        "avg_divergence":  round(_safe_mean(avg_divs), 4),
        "avg_beta":        round(_safe_mean(avg_betas), 4),
        "avg_violation":   round(_safe_mean(avg_viols), 4),
        "n_seeds":         len(seeds),
    }


def _run_one(params: dict, episodes: int, seeds: list[int],
             save_dir: str, device: str, trial_id: int) -> dict | None:
    """Run a single hyperparameter configuration over all seeds sequentially
    (single process) and return the aggregated metrics record."""
    seed_histories = [
        _run_one_seed(params, episodes, seed, save_dir, device, trial_id)
        for seed in seeds
    ]
    return _aggregate_seed_histories(seed_histories, params, seeds, trial_id)


def _run_configs_parallel(config_records: list[tuple[int, dict]], episodes: int,
                           seeds: list[int], save_dir: str, device: str,
                           n_jobs: int) -> list[dict]:
    """Run several (trial_id, params) configs, parallelizing every individual
    (config, seed) training run across a process pool of size `n_jobs`.

    This is the parallelization strategy for a 6-core machine: with e.g.
    3 configs x 5 seeds = 15 independent training runs, up to `n_jobs`
    (default 6) run concurrently regardless of which config they belong to,
    keeping all cores busy until every run finishes.
    """
    # Flatten to a list of individual (trial_id, params, seed) jobs.
    jobs = [
        (trial_id, params, seed)
        for trial_id, params in config_records
        for seed in seeds
    ]

    # trial_id -> list of histories collected so far
    collected: dict[int, list] = {tid: [] for tid, _ in config_records}

    print(f"  Launching {len(jobs)} training runs across {n_jobs} worker process(es)...")

    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        future_to_job = {
            pool.submit(_run_one_seed, params, episodes, seed, save_dir, device, trial_id): (trial_id, seed)
            for trial_id, params, seed in jobs
        }
        done = 0
        for future in as_completed(future_to_job):
            trial_id, seed = future_to_job[future]
            history = future.result()
            collected[trial_id].append(history)
            done += 1
            print(f"  [{done}/{len(jobs)}] finished trial_{trial_id} seed={seed}")

    results = []
    for trial_id, params in config_records:
        rec = _aggregate_seed_histories(collected[trial_id], params, seeds, trial_id)
        if rec:
            results.append(rec)
    return results


def _print_banner(method: str, n_trials: int, episodes: int):
    print("\n" + "=" * 78)
    print(f"  AC-PPO Hyperparameter Tuning — LunarLander-v3")
    print(f"  Method: {method.upper()}  |  Trials: {n_trials}  |  Episodes/trial: {episodes}")
    print(f"  Base PPO (locked): rollout={BASE_PPO['rollout_steps']}  "
          f"epochs={BASE_PPO['ppo_epochs']}  lr={BASE_PPO['lr']}")
    print("  τ grid calibrated to observed D_t range (0.005–0.15 from training logs)")
    print("=" * 78 + "\n")


def _print_and_save_table(results: list[dict], save_dir: str, label: str,
                           elapsed: float) -> None:
    os.makedirs(save_dir, exist_ok=True)
    lines = []
    lines.append("\n" + "=" * 100)
    lines.append(f"  TUNING RESULTS  [{label}]  — LunarLander-v3")
    lines.append("=" * 100)
    lines.append(
        f"  {'Rk':<3} {'#':<4} {'Score':>8} {'FinalMean':>10} {'MeanEval':>10} {'Std':>7} "
        f"{'Best':>8} {'Succ':>5} {'AvgD':>7} {'Avgβ':>7} {'AvgViol':>8}  "
        f"Params"
    )
    lines.append("  " + "─" * 97)
    for rank, r in enumerate(results, 1):
        p = r["params"]
        ps = (f"τ={p['tau']}  β_max={p['beta_max']}  "
              f"d={p['d_threshold']}  λ={p['lambda_coef']}  k={p['k_slope']}")
        lines.append(
            f"  {rank:<3} #{r['trial_id']:<3} "
            f"{r.get('score', float('nan')):>+8.2f} "
            f"{r['final_mean_eval']:>+10.2f} "
            f"{r['mean_eval']:>+10.2f} "
            f"{r['std_eval']:>7.2f} "
            f"{r['best_eval']:>+8.2f} "
            f"{r['succ_rate']:>5.2f} "
            f"{r['avg_divergence']:>7.4f} "
            f"{r['avg_beta']:>7.4f} "
            f"{r['avg_violation']:>8.4f}  "
            f"{ps}"
        )
    lines.append("=" * 100)

    if results:
        best = results[0]
        lines.append(f"\n  🏆 BEST CONFIGURATION  (Trial #{best['trial_id']}, {best['n_seeds']} seed(s)):")
        lines.append(f"     {best['params']}")
        lines.append(f"     Score={best.get('score', float('nan')):+.3f}  "
                     f"FinalMeanEval={best['final_mean_eval']:+.3f}  "
                     f"Best={best['best_eval']:+.3f}  "
                     f"Succ={best['succ_rate']:.2f}  "
                     f"AvgD={best['avg_divergence']:.4f}  "
                     f"Avgβ={best['avg_beta']:.4f}  "
                     f"AvgViol={best['avg_violation']:.4f}")

    lines.append(f"\n  Wall time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    lines.append("=" * 100 + "\n")

    text = "\n".join(lines)
    print(text)

    txt_path  = os.path.join(save_dir, "tuning_summary.txt")
    json_path = os.path.join(save_dir, "tuning_summary.json")
    with open(txt_path,  "w") as f: f.write(text)
    with open(json_path, "w") as f: json.dump(results, f, indent=2)
    print(f"  Saved → {txt_path}")
    print(f"  JSON  → {json_path}")


# ── Stage 1: random / grid search ─────────────────────────────────────────────

def cmd_stage1(args):
    trials  = _generate_trials(args.method, args.num_samples)
    _print_banner(args.method, len(trials), args.episodes)

    results = []
    t0 = time.time()
    for i, params in enumerate(trials, 1):
        print(f"\n{'─'*78}")
        print(f"  Stage-1 Trial [{i}/{len(trials)}]  {params}")
        print(f"{'─'*78}")
        rec = _run_one(params, args.episodes, [args.seed],
                       args.save_dir, args.device, i)
        if rec:
            results.append(rec)

    # Rank by score = final_mean_eval - 0.10 * std_eval (same criterion as Stage 2)
    results.sort(
        key=lambda r: r["score"],
        reverse=True
    )
    _print_and_save_table(results, args.save_dir, "Stage 1", time.time() - t0)
    print(f"\n  ➜  Run Stage 2 on the top-{args.top_k} configs:")
    print(f"     python tune_lunarlander.py stage2 "
          f"--stage1_json {args.save_dir}/tuning_summary.json "
          f"--top_k {args.top_k} --seeds 42 1 2 3 4 --episodes {args.confirm_episodes} --n_jobs 6")


# ── Stage 2: confirmation run ──────────────────────────────────────────────────

def cmd_stage2(args):
    with open(args.stage1_json) as f:
        stage1_results = json.load(f)

    top_configs = stage1_results[: args.top_k]
    print(f"\n{'='*78}")
    print(f"  Stage-2 Confirmation: top-{len(top_configs)} configs x "
          f"{len(args.seeds)} seeds x {args.episodes} episodes")
    print(f"  Parallel workers: {args.n_jobs}")
    print(f"{'='*78}\n")

    config_records = [(i, cfg_rec["params"]) for i, cfg_rec in enumerate(top_configs, 1)]

    t0 = time.time()
    results = _run_configs_parallel(
        config_records, args.episodes, args.seeds,
        args.save_dir, args.device, n_jobs=args.n_jobs,
    )

    results.sort(
        key=lambda r: r["score"],
        reverse=True
    )
    _print_and_save_table(results, args.save_dir, "Stage 2 (confirmed)", time.time() - t0)


# ── Confirmation experiment ────────────────────────────────────────────────────

def cmd_confirm(args):
    """
    Final single-config confirmation run before freezing hyperparameters.
    Runs a fixed configuration over N seeds × E episodes and emits a
    detailed publication-ready report.
    """
    params = {
        "tau":         args.tau,
        "beta_max":    args.beta_max,
        "d_threshold": args.d_threshold,
        "lambda_coef": args.lambda_coef,
        "k_slope":     args.k_slope,
    }

    print("\n" + "═" * 78)
    print("  AC-PPO — Final Confirmation Experiment (LunarLander-v3)")
    print("═" * 78)
    print(f"  Config  : {params}")
    print(f"  Seeds   : {args.seeds}  ({len(args.seeds)} seeds)")
    print(f"  Episodes: {args.episodes}")
    print(f"  Parallel workers: {args.n_jobs}")
    print(f"  Base PPO: rollout={BASE_PPO['rollout_steps']}  "
          f"epochs={BASE_PPO['ppo_epochs']}  lr={BASE_PPO['lr']}")
    print("═" * 78 + "\n")

    t0 = time.time()
    results = _run_configs_parallel(
        [(1, params)], args.episodes, args.seeds,
        args.save_dir, args.device, n_jobs=args.n_jobs,
    )
    rec = results[0] if results else None
    elapsed = time.time() - t0

    if rec is None:
        print("  ERROR: no evaluation data returned — check training logs.")
        return

    os.makedirs(args.save_dir, exist_ok=True)

    lines = []
    lines.append("\n" + "═" * 78)
    lines.append("  AC-PPO FINAL HYPERPARAMETER CONFIRMATION REPORT")
    lines.append(f"  Environment : LunarLander-v3")
    lines.append(f"  Seeds       : {args.seeds}")
    lines.append(f"  Episodes    : {args.episodes}")
    lines.append("═" * 78)
    lines.append("")
    lines.append("  ── Hyperparameters (to be frozen) ──")
    lines.append(f"      τ  (conflict threshold)      = {params['tau']}")
    lines.append(f"      β_max (max cog influence)    = {params['beta_max']}")
    lines.append(f"      d  (affective cost tol.)     = {params['d_threshold']}")
    lines.append(f"      λ  (constraint weight)       = {params['lambda_coef']}")
    lines.append(f"      k  (sigmoid slope)           = {params['k_slope']}")
    lines.append("")
    lines.append("  ── Base PPO (locked, SB3 defaults) ──")
    lines.append(f"      lr={BASE_PPO['lr']}  γ={BASE_PPO['gamma']}  "
                 f"λ_GAE={BASE_PPO['gae_lambda']}  ε_clip={BASE_PPO['clip_eps']}")
    lines.append(f"      rollout_steps={BASE_PPO['rollout_steps']}  "
                 f"ppo_epochs={BASE_PPO['ppo_epochs']}  mini_batch={BASE_PPO['mini_batch']}")
    lines.append("")
    lines.append("  ── Performance Metrics ──")
    lines.append(f"      Final Mean Eval  : {rec['final_mean_eval']:+.3f}")
    lines.append(f"      Overall Mean Eval: {rec['mean_eval']:+.3f}  ± {rec['std_eval']:.3f}")
    lines.append(f"      Best Eval        : {rec['best_eval']:+.3f}")
    lines.append(f"      Success Rate     : {rec['succ_rate']:.3f}")
    lines.append("")
    lines.append("  ── Cognitive-Affective Regulation Metrics ──")
    lines.append(f"      Avg JS Divergence  : {rec['avg_divergence']:.4f}")
    lines.append(f"      Avg β_t            : {rec['avg_beta']:.4f}")
    lines.append(f"      Avg Violation Rate : {rec['avg_violation']:.4f}")
    lines.append("")

    # Freeze recommendation
    reward_ok = rec["final_mean_eval"] > -120
    succ_ok   = rec["succ_rate"]       > 0.95
    viol_ok   = rec["avg_violation"]   < 0.05
    all_ok    = reward_ok and succ_ok and viol_ok

    lines.append("  ── Freeze Recommendation ──")
    lines.append(f"      Reward threshold (> -120)  : {'✓ PASS' if reward_ok else '✗ FAIL'}  "
                 f"({rec['final_mean_eval']:+.2f})")
    lines.append(f"      Success rate   (> 0.95)    : {'✓ PASS' if succ_ok else '✗ FAIL'}  "
                 f"({rec['succ_rate']:.3f})")
    lines.append(f"      Avg violation  (< 0.05)    : {'✓ PASS' if viol_ok else '✗ FAIL'}  "
                 f"({rec['avg_violation']:.4f})")
    lines.append("")
    if all_ok:
        lines.append("  ✅  ALL CHECKS PASSED — Safe to freeze these hyperparameters.")
        lines.append("      Update FullConfig defaults in train_full_model.py:")
        lines.append(f"          tau          = {params['tau']}")
        lines.append(f"          beta_max     = {params['beta_max']}")
        lines.append(f"          d_threshold  = {params['d_threshold']}")
        lines.append(f"          lambda_coef  = {params['lambda_coef']}")
        lines.append(f"          k_slope      = {params['k_slope']}")
    else:
        lines.append("  ⚠️  ONE OR MORE CHECKS FAILED — Review results before freezing.")
        lines.append("      Consider re-running Stage 2 with more seeds or a different config.")

    lines.append("")
    lines.append(f"  Wall time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    lines.append("═" * 78 + "\n")

    text = "\n".join(lines)
    print(text)

    txt_path  = os.path.join(args.save_dir, "confirmation_report.txt")
    json_path = os.path.join(args.save_dir, "confirmation_report.json")
    with open(txt_path,  "w") as f: f.write(text)
    with open(json_path, "w") as f: json.dump(rec, f, indent=2)
    print(f"  Report saved → {txt_path}")
    print(f"  JSON saved   → {json_path}")


# ── Sensitivity analysis ───────────────────────────────────────────────────────

def cmd_sensitivity(args):
    """
    For each key hyperparameter, sweep its values while holding the others
    at the best configuration found in Stage 1. Produces one subplot per param.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("  matplotlib not available — cannot produce sensitivity figure.")
        return

    with open(args.stage1_json) as f:
        stage1_results = json.load(f)

    if not stage1_results:
        print("  No Stage-1 results found.")
        return

    best_params = stage1_results[0]["params"]
    sweep_params = ["tau", "beta_max", "lambda_coef", "d_threshold"]

    print(f"\n  Sensitivity analysis — sweeping each parameter around best config:")
    print(f"  Best config: {best_params}\n")

    sweep_data = {}   # param_name → {value: record}

    for param in sweep_params:
        sweep_data[param] = {}
        for val in PARAM_GRID[param]:
            trial_params = dict(best_params, **{param: val})
            # Skip if it matches best (already have it from stage1)
            matching = [r for r in stage1_results
                        if r["params"].get(param) == val and
                        all(r["params"].get(k) == best_params[k]
                            for k in best_params if k != param)]
            if matching:
                sweep_data[param][val] = matching[0]
                continue

            trial_id = 1000 + len(sweep_data[param])
            print(f"  Sweep {param}={val} ...")
            rec = _run_one(trial_params, args.episodes, [args.seed],
                           args.save_dir, args.device, trial_id=trial_id)
            if rec:
                sweep_data[param][val] = rec

    # Plot
    n = len(sweep_params)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5), sharey=False)
    if n == 1:
        axes = [axes]

    fig.suptitle("AC-PPO Sensitivity Analysis (LunarLander-v3)",
                 fontsize=13, fontweight="bold", y=1.03)

    for ax, param in zip(axes, sweep_params):
        data = sweep_data[param]
        xs   = sorted(data.keys())
        ys_mean   = [data[x]["final_mean_eval"] if x in data else float("nan") for x in xs]
        ys_std    = [data[x]["std_eval"]         if x in data else 0.0          for x in xs]
        ys_viol   = [data[x]["avg_violation"]    if x in data else float("nan") for x in xs]

        ax2 = ax.twinx()
        ax.errorbar(xs, ys_mean, yerr=ys_std,
                    fmt="o-", color="#2ca02c", linewidth=2, markersize=7,
                    capsize=5, label="Final Mean Eval ± Std", zorder=3)
        ax2.plot(xs, ys_viol, "s--", color="#e07b39", linewidth=1.5,
                 markersize=6, alpha=0.8, label="Avg Violation Rate")

        # Mark best value
        best_val = best_params.get(param)
        if best_val in xs:
            idx  = xs.index(best_val)
            ax.axvline(best_val, color="#2ca02c", linestyle=":", alpha=0.5)
            ax.scatter([best_val], [ys_mean[idx]], s=120, zorder=5,
                       color="#2ca02c", marker="*", label=f"Best (={best_val})")

        ax.set_title(PARAM_DISPLAY[param], fontsize=10)
        ax.set_xlabel("Parameter value", fontsize=9)
        ax.set_ylabel("Final Mean Eval Reward", fontsize=9, color="#2ca02c")
        ax2.set_ylabel("Avg Violation Rate", fontsize=9, color="#e07b39", rotation=-90, labelpad=12)
        ax.tick_params(axis="y", colors="#2ca02c")
        ax2.tick_params(axis="y", colors="#e07b39")

        lines1, labs1 = ax.get_legend_handles_labels()
        lines2, labs2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labs1 + labs2, fontsize=7, loc="lower right")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(args.save_dir, exist_ok=True)
    out = os.path.join(args.save_dir, "sensitivity_analysis.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Sensitivity figure saved → {out}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    root = argparse.ArgumentParser(
        description="AC-PPO two-stage hyperparameter tuning on LunarLander-v3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = root.add_subparsers(dest="command", required=True)

    # ── stage1
    s1 = sub.add_parser("stage1", help="Random/grid search (short runs)")
    s1.add_argument("--method",           choices=["random","grid"], default="random")
    s1.add_argument("--num_samples",      type=int,   default=100,
                    help="Number of hyperparameter configs to try")
    s1.add_argument("--episodes",         type=int,   default=400,
                    help="Training episodes per trial (default 400)")
    s1.add_argument("--seed",             type=int,   default=42,
                    help="Single seed for stage-1 exploration (default 42)")
    s1.add_argument("--top_k",            type=int,   default=3,
                    help="Number of top configs to confirm in stage-2")
    s1.add_argument("--confirm_episodes", type=int,   default=2000,
                    help="Episodes hint to print for stage-2 command")
    s1.add_argument("--save_dir",         default="tuning_results/stage1")
    s1.add_argument("--device",           default="cpu")

    # ── stage2
    s2 = sub.add_parser("stage2", help="Confirm top configs with multi-seed long runs")
    s2.add_argument("--stage1_json", required=True,
                    help="Path to tuning_summary.json from stage1")
    s2.add_argument("--top_k",       type=int,          default=3)
    s2.add_argument("--seeds",       type=int, nargs="+", default=[42, 1, 2, 3, 4])
    s2.add_argument("--episodes",    type=int,          default=2000)
    s2.add_argument("--save_dir",    default="tuning_results/stage2")
    s2.add_argument("--device",      default="cpu")
    s2.add_argument("--n_jobs",      type=int,          default=6,
                    help="Number of parallel worker processes (default: 6, matches 6-core CPU)")

    # ── sensitivity
    sv = sub.add_parser("sensitivity", help="Produce sensitivity analysis figure")
    sv.add_argument("--stage1_json", required=True,
                    help="Path to tuning_summary.json from stage1")
    sv.add_argument("--episodes",    type=int, default=300,
                    help="Episodes for new sweep points not already in stage1")
    sv.add_argument("--seed",        type=int, default=42)
    sv.add_argument("--save_dir",    default="tuning_results/sensitivity")

    # ── confirm
    cf = sub.add_parser(
        "confirm",
        help="Final single-config confirmation run before freezing (N seeds × E episodes)",
    )
    cf.add_argument("--tau",         type=float, default=0.10,
                    help="τ: conflict threshold (default: 0.10)")
    cf.add_argument("--beta_max",    type=float, default=0.30,
                    help="β_max: max cognitive influence (default: 0.30)")
    cf.add_argument("--d_threshold", type=float, default=0.20,
                    help="d: affective cost tolerance (default: 0.20)")
    cf.add_argument("--lambda_coef", type=float, default=0.05,
                    help="λ: constraint weight (default: 0.05)")
    cf.add_argument("--k_slope",     type=float, default=10.0,
                    help="k: sigmoid slope (default: 10.0)")
    cf.add_argument("--seeds",    type=int, nargs="+", default=[42, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    help="Seeds to run (default: 10 seeds [42 1 2 3 4 5 6 7 8 9])")
    cf.add_argument("--episodes", type=int, default=2000,
                    help="Training episodes per seed (default: 2000)")
    cf.add_argument("--save_dir", default="tuning_results/confirm")
    cf.add_argument("--device",   default="cpu")
    cf.add_argument("--n_jobs",   type=int, default=6,
                    help="Number of parallel worker processes (default: 6, matches 6-core CPU)")

    return root.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.command == "stage1":
        cmd_stage1(args)
    elif args.command == "stage2":
        cmd_stage2(args)
    elif args.command == "sensitivity":
        cmd_sensitivity(args)
    elif args.command == "confirm":
        cmd_confirm(args)
