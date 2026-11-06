import argparse
import os
import sys
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))

from envs.gym_env               import GymEnvWrapper, make_env, SUPPORTED_ENVS
from agents.ppo_agent          import PPOEnvAgent, PPOCogAgent
from modules.appraisal         import AppraisalModule
from modules.cognitive_state   import CognitiveAffectiveState
from modules.cognitive_reward  import CognitiveReward
from modules.divergence        import PolicyDivergence
from modules.behaviour_influence import BehaviourInfluence
from modules.constraints       import ConstraintModule, AdaptiveLambda


# ============================================================================ #
#  Config                                                                       #
# ============================================================================ #

@dataclass
class FullConfig:
    # Environment
    env_name:        str   = "CartPole-v1"   # CartPole-v1 | Acrobot-v1 | LunarLander-v3 | MountainCar-v0
    seed:            int   = 42

    # Training  (2000 eps × 10 seeds is the final experimental protocol)
    total_episodes:  int   = 2_000
    rollout_steps:   int   = 1_024  # SB3 default for LunarLander / general
    ppo_epochs:      int   = 4
    mini_batch:      int   = 64
    eval_interval:   int   = 100
    eval_episodes:   int   = 10
    log_interval:    int   = 50

    # PPO shared  (locked to SB3 / Schulman-2017 defaults — same for B1/B2/B3/A1)
    lr:              float = 3e-4
    gamma:           float = 0.99
    gae_lambda:      float = 0.95
    clip_eps:        float = 0.2
    value_coef:      float = 0.5
    entropy_coef:    float = 0.01
    max_grad_norm:   float = 0.5
    hidden_size:     int   = 128

    # Appraisal
    mental_dim:      int   = 64
    latent_dim:      int   = 64    # appraisal encoder z_t dimension (unchanged)
    # NOTE: c_t (cognitive-affective state) is now always 3-dim [ĝ, n̂, p̂]
    #       CognitiveAffectiveState.APPRAISAL_DIM = 3 — fixed, not configurable
    appraisal_lr:    float = 3e-4

    # Cognitive reward weights
    beta_r:          float = 0.5    # weight for environmental reward in R_cog
    alpha_g:         float = 1.0
    alpha_p:         float = 1.0
    # NOTE: alpha_n removed — novelty now amplifies β_t, not R_cog

    # Cognitive-affective state
    alpha_ema:       float = 0.1    # emotional memory factor

    # Divergence + behaviour influence  (FROZEN from two-stage tuning on LunarLander-v3)
    tau:             float = 0.10   # conflict threshold          [tuned]
    k_slope:         float = 10.0   # sigmoid slope               [tuned]
    beta_max:        float = 0.30   # max cognitive influence     [tuned]

    # Constraints  (FROZEN from two-stage tuning)
    d_threshold:     float = 0.20   # acceptable affective cost   [tuned]
    lambda_coef:     float = 0.05   # constraint weight           [tuned]
    adaptive_lambda: bool  = False  # use dual ascent instead of fixed λ
    lambda_lr:       float = 0.05   # dual ascent step size

    # Mode flags
    mode:            str   = "full"  # baseline (B1) | affect_reward (B2) | full (B3) | appraisal_only (A1-ablation)

    # Saving / logging
    save_dir:        str   = "checkpoints"
    run_name:        str   = "ac_ppo"
    results_dir:     str   = ""     # if non-empty, write per-run CSVs here
    device:          str   = "cpu"
    render:          bool  = False


# ============================================================================ #
#  Mode helpers                                                                 #
# ============================================================================ #

def mode_flags(mode: str) -> dict:
    """Return boolean flags controlling which components are active."""
    return {
        "baseline":      dict(use_appraisal=False, use_cog_ppo=False, use_constraint=False, affect_as_reward=False),
        "affect_reward": dict(use_appraisal=True,  use_cog_ppo=False, use_constraint=False, affect_as_reward=True),
        "appraisal_only":dict(use_appraisal=True,  use_cog_ppo=True,  use_constraint=False, affect_as_reward=False),
        "full":          dict(use_appraisal=True,  use_cog_ppo=True,  use_constraint=True,  affect_as_reward=False),
    }[mode]


# ============================================================================ #
#  Evaluation                                                                   #
# ============================================================================ #

def is_success(env_name: str, ep_reward: float, ep_length: int, terminated: bool, truncated: bool) -> float:
    """
    Determine episode success based on environment-specific definitions:
    - CartPole-v1: Surviving 500 steps (or >= 475) without falling (falling early = terminated).
    - Acrobot-v1: Swinging tip above target line (goal reached = terminated; timeout = truncated).
    - LunarLander-v3: Landing safely inside target flags (ep_reward >= 200.0 or (terminated and ep_reward > 0)).
    """
    name = env_name.split("/")[-1] if "/" in env_name else env_name
    if "CartPole" in name:
        return 1.0 if (truncated or ep_length >= 475 or ep_reward >= 475) else 0.0
    elif "Acrobot" in name:
        return 1.0 if terminated else 0.0
    elif "LunarLander" in name:
        return 1.0 if (ep_reward >= 200.0 or (terminated and ep_reward > 0)) else 0.0
    else:
        return 1.0 if terminated else 0.0


def evaluate(
    env_agent:     PPOEnvAgent,
    cog_agent:     PPOCogAgent | None,
    appraisal_mod: AppraisalModule | None,
    cas:           CognitiveAffectiveState | None,
    influence:     BehaviourInfluence | None,
    cfg:           FullConfig,
    flags:         dict,
    n_episodes:    int = 10,
) -> dict:
    env     = make_env(cfg.env_name, seed=cfg.seed + 2000)
    rewards, lengths, successes, div_vals, cost_vals = [], [], [], [], []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=cfg.seed + 2000 + ep)
        if cas is not None:
            cas.reset()
        c_t       = torch.zeros(3) if flags["use_cog_ppo"] else None
        ep_r      = 0.0
        ep_length = 0
        done      = False

        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32)

            with torch.no_grad():
                # Env policy
                action_e, _, val_e = env_agent.select_action(obs)
                p_env = env_agent.get_probs(obs_t)

                if flags["use_cog_ppo"] and cog_agent is not None:
                    p_cog = cog_agent.get_probs(obs_t, c_t)
                    D_t   = PolicyDivergence(normalise=True).compute_scalar(p_env, p_cog)
                    beta_t = influence.compute_beta(D_t) if influence else 0.0
                    p_final = (1 - beta_t) * p_env + beta_t * p_cog
                    from torch.distributions import Categorical
                    action = Categorical(probs=p_final).sample().item()
                    div_vals.append(D_t)
                    cost_vals.append(abs(beta_t * D_t * ep_r) if ep_r != 0 else 0.0)
                else:
                    action = action_e

            obs, reward, terminated, truncated, info = env.step(action)
            ep_r      += reward
            ep_length += 1
            done       = terminated or truncated

            if flags["use_appraisal"] and appraisal_mod is not None and cas is not None:
                with torch.no_grad():
                    appr = appraisal_mod.infer(
                        torch.as_tensor(obs, dtype=torch.float32),
                        reward, action, val_e
                    )
                    a_hat = torch.stack([
                        torch.tensor(appr["g"]),
                        torch.tensor(appr["n"]),
                        torch.tensor(appr["p"]),
                    ])
                    cas.update(a_hat)
                    if c_t is not None:
                        c_t = cas.get_flat()

        rewards.append(ep_r)
        lengths.append(ep_length)
        successes.append(is_success(cfg.env_name, ep_r, ep_length, terminated, truncated))

    result = {
        "mean_reward":  float(np.mean(rewards)),
        "std_reward":   float(np.std(rewards)),
        "success_rate": float(np.mean(successes)),
        "mean_length":  float(np.mean(lengths)),
    }
    if div_vals:
        result["mean_divergence"] = float(np.mean(div_vals))
        result["mean_cost"]       = float(np.mean(cost_vals))
    return result


# ============================================================================ #
#  Main Training Loop                                                           #
# ============================================================================ #

def train(cfg: FullConfig) -> dict:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    flags = mode_flags(cfg.mode)

    # ── environment ──────────────────────────────────────────────────────────
    env = make_env(cfg.env_name, seed=cfg.seed)
    OBS_DIM = env.obs_dim
    ACT_DIM = env.act_dim

    # ── PPO-Env ───────────────────────────────────────────────────────────────
    ppo_kwargs = dict(
        act_dim=ACT_DIM, hidden=cfg.hidden_size, lr=cfg.lr,
        rollout_steps=cfg.rollout_steps, ppo_epochs=cfg.ppo_epochs,
        mini_batch=cfg.mini_batch, gamma=cfg.gamma, gae_lambda=cfg.gae_lambda,
        clip_eps=cfg.clip_eps, value_coef=cfg.value_coef,
        entropy_coef=cfg.entropy_coef, max_grad_norm=cfg.max_grad_norm,
        device=cfg.device,
    )
    env_agent = PPOEnvAgent(obs_dim=OBS_DIM, **ppo_kwargs)

    # ── PPO-Cog ───────────────────────────────────────────────────────────────
    cog_agent = None
    if flags["use_cog_ppo"]:
        cog_agent = PPOCogAgent(obs_dim=OBS_DIM, **ppo_kwargs)
        # PPO-Cog input: obs_dim + cog_dim(3)

    # ── Appraisal module ──────────────────────────────────────────────────────
    appraisal_mod  = None
    appraisal_optim = None
    if flags["use_appraisal"]:
        appraisal_mod   = AppraisalModule(OBS_DIM, ACT_DIM, cfg.mental_dim, cfg.latent_dim)
        appraisal_optim = torch.optim.Adam(appraisal_mod.parameters(), lr=cfg.appraisal_lr)

    # ── Cognitive-affective state ─────────────────────────────────────────────
    cas = CognitiveAffectiveState(alpha=cfg.alpha_ema) \
          if flags["use_appraisal"] else None

    # ── Cognitive reward ──────────────────────────────────────────────────────
    cog_reward_fn = CognitiveReward(cfg.beta_r, cfg.alpha_g, cfg.alpha_p) \
                    if flags["use_appraisal"] else None

    # ── Divergence + influence ────────────────────────────────────────────────
    divergence = PolicyDivergence(normalise=True)  if flags["use_cog_ppo"] else None
    influence  = BehaviourInfluence(cfg.tau, cfg.k_slope, cfg.beta_max) \
                 if flags["use_cog_ppo"] else None

    # ── Constraint module ─────────────────────────────────────────────────────
    cm = ConstraintModule(cfg.d_threshold, cfg.lambda_coef) if flags["use_constraint"] else None
    adaptive_lam = AdaptiveLambda(cfg.d_threshold, cfg.lambda_lr, cfg.lambda_coef) \
                   if (flags["use_constraint"] and cfg.adaptive_lambda) else None

    # ── Logging ───────────────────────────────────────────────────────────────
    os.makedirs(cfg.save_dir, exist_ok=True)
    history = {k: [] for k in [
        "ep_reward", "ep_length", "goal_rate",
        "eval_reward", "eval_success",
        "env_loss", "cog_loss", "appraisal_loss",
        "divergence", "beta", "affective_cost", "violation_rate",
        "constraint_loss", "lambda_val",
    ]}

    recent = {k: deque(maxlen=cfg.log_interval)
              for k in ["reward", "length", "success", "D", "beta", "cost"]}

    print(f"\n{'='*65}")
    print(f"  AC-PPO  mode={cfg.mode}  episodes={cfg.total_episodes}  seed={cfg.seed}")
    print(f"  Components: appraisal={flags['use_appraisal']}  "
          f"cog_ppo={flags['use_cog_ppo']}  "
          f"constraint={flags['use_constraint']}  "
          f"affect_as_reward={flags['affect_as_reward']}")
    print(f"{'='*65}\n")

    # ── Rollout state ─────────────────────────────────────────────────────────
    obs, _     = env.reset()
    if cas:
        cas.reset()
    c_t        = torch.zeros(3)   # [ĝ_memory, n̂_memory, p̂_memory]
    ep_reward  = 0.0
    ep_length  = 0
    ep_count   = 0
    global_step = 0
    best_eval  = -np.inf
    start_time = time.time()

    # Per-rollout cost buffer (for constraint loss — logging only now)
    step_costs: list[float] = []

    # Rollout tensor buffers for differentiable constraint (B4 mode)
    _obs_rollout: list[torch.Tensor]  = []
    _c_t_rollout: list[torch.Tensor]  = []
    _rew_rollout: list[float]         = []

    # ── Main loop ─────────────────────────────────────────────────────────────
    while ep_count < cfg.total_episodes:

        obs_t = torch.as_tensor(obs, dtype=torch.float32)

        # ── 1. PPO-Env: select action ─────────────────────────────────────────
        action_env, logp_env, val_env = env_agent.select_action(obs)
        p_env = env_agent.get_probs(obs_t).detach()

        # ── 2. PPO-Cog: select action and get probs ───────────────────────────
        if flags["use_cog_ppo"] and cog_agent is not None:
            action_cog, logp_cog, val_cog = cog_agent.select_action(obs, c_t)
            p_cog = cog_agent.get_probs(obs_t, c_t).detach()

            # JS divergence
            D_t = divergence.compute_scalar(p_env, p_cog)
            divergence.record(D_t)
            recent["D"].append(D_t)

            # Behaviour influence coefficient
            beta_t = influence.compute_beta(D_t)
            recent["beta"].append(beta_t)

            # Soft blend → final action
            p_final   = influence.blend(p_env, p_cog, beta_t)
            from torch.distributions import Categorical
            dist_final = Categorical(probs=p_final)
            action     = dist_final.sample().item()
            logp_final = dist_final.log_prob(torch.tensor(action)).item()
        else:
            action     = action_env
            logp_final = logp_env
            D_t        = 0.0
            beta_t     = 0.0
            val_cog    = 0.0
            logp_cog   = logp_env

        # ── 3. Environment step ───────────────────────────────────────────────
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # ── 4. Appraisal pipeline ─────────────────────────────────────────────
        r_cog = 0.0
        if flags["use_appraisal"] and appraisal_mod is not None:
            with torch.no_grad():
                next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32)
                appr = appraisal_mod.infer(obs_t, reward, action, val_env)

            # Cognitive reward
            r_cog = cog_reward_fn.compute_single(reward, appr["g"], appr["p"])

            # Affect-as-reward baseline (B2): add r_cog to env reward
            if flags["affect_as_reward"]:
                reward = reward + r_cog

            # Update cognitive-affective state with â_t = [ĝ, n̂, p̂]
            a_hat = torch.tensor([appr["g"], appr["n"], appr["p"]])
            cas.update(a_hat)
            c_t = cas.get_flat()   # (3,) — [ĝ_memory, n̂_memory, p̂_memory]

        # ── 5. Affective cost (logging / violation tracking only) ─────────────
        C_t = 0.0
        if flags["use_constraint"] and cm is not None:
            C_t = cm.compute_cost(beta_t, D_t, reward)
            cm.record_cost(C_t)
            step_costs.append(C_t)
            recent["cost"].append(C_t)

        # ── 6. Store transitions ──────────────────────────────────────────────
        # PPO-Env sees the (possibly affect-augmented) reward
        env_agent.store(obs, action, logp_final, reward, val_env, float(done))

        # PPO-Cog sees the cognitive reward
        if flags["use_cog_ppo"] and cog_agent is not None:
            cog_agent.store_cog(obs, c_t, action, logp_cog, r_cog, val_cog, float(done))

        # Accumulate tensor buffers for differentiable constraint
        if flags["use_constraint"]:
            _obs_rollout.append(torch.as_tensor(obs, dtype=torch.float32))
            _c_t_rollout.append(c_t.detach().clone())
            _rew_rollout.append(float(reward))

        obs       = next_obs
        ep_reward += reward
        ep_length += 1
        global_step += 1

        # ── 7. Episode end ────────────────────────────────────────────────────
        if done:
            ep_count += 1
            succ_val = is_success(cfg.env_name, ep_reward, ep_length, terminated, truncated)
            recent["reward"].append(ep_reward)
            recent["length"].append(ep_length)
            recent["success"].append(succ_val)

            history["ep_reward"].append(ep_reward)
            history["ep_length"].append(ep_length)
            history["goal_rate"].append(succ_val)
            history["divergence"].append(D_t)
            history["beta"].append(beta_t)
            history["affective_cost"].append(C_t)

            obs, _ = env.reset()
            if cas:
                cas.reset()
            c_t       = torch.zeros(3)   # reset emotional memory
            ep_reward = 0.0
            ep_length = 0

            # ── 8. PPO updates (when rollout buffer is full) ──────────────────
            if env_agent.buffer.ready():
                obs_last = obs

                # 8a. Appraisal update
                if flags["use_appraisal"] and appraisal_mod is not None:
                    _run_appraisal_update(
                        appraisal_mod, appraisal_optim,
                        env_agent.buffer, OBS_DIM, ACT_DIM
                    )

                # 8b. PPO-Env update — with or without differentiable constraint
                effective_lambda = cfg.lambda_coef
                if flags["use_constraint"] and cog_agent is not None and _obs_rollout:
                    # Adaptive lambda (dual ascent) step, if enabled
                    if adaptive_lam is not None and step_costs:
                        effective_lambda = adaptive_lam.step(
                            sum(step_costs) / len(step_costs)
                        )
                        step_costs.clear()

                    # Build rollout tensors from this rollout window
                    obs_buf = torch.stack(_obs_rollout)
                    c_t_buf = torch.stack(_c_t_rollout)
                    rew_buf = torch.tensor(_rew_rollout, dtype=torch.float32)

                    env_info = env_agent.update_env_with_constraint(
                        last_obs    = obs_last,
                        cog_network = cog_agent.network,
                        obs_buf     = obs_buf,
                        c_t_buf     = c_t_buf,
                        rewards_buf = rew_buf,
                        lambda_coef = effective_lambda,
                        d_threshold = cfg.d_threshold,
                        tau         = cfg.tau,
                        k_slope     = cfg.k_slope,
                        beta_max    = cfg.beta_max,
                    )
                    _obs_rollout.clear()
                    _c_t_rollout.clear()
                    _rew_rollout.clear()
                else:
                    env_info = env_agent.update_env(obs_last)

                # 8d. PPO-Cog update
                cog_info = {}
                if flags["use_cog_ppo"] and cog_agent is not None:
                    cog_info = cog_agent.update_cog(
                        obs_last, torch.zeros(3)   # bootstrap with zero c_t
                    )

                # Record losses
                history["env_loss"].append(env_info.get("env_loss", 0.0))
                if cog_info:
                    history["cog_loss"].append(cog_info.get("cog_loss", 0.0))
                if flags["use_constraint"] and cm is not None:
                    # Use the real differentiable loss from env_info when available,
                    # fall back to cm.summary() for B3-style logging.
                    con_loss_val = env_info.get("env_constraint_loss",
                                                cm.summary()["constraint_loss"])
                    history["constraint_loss"].append(con_loss_val)
                    history["violation_rate"].append(cm.violation_rate())
                lam_val = adaptive_lam.value if adaptive_lam else cfg.lambda_coef
                history["lambda_val"].append(lam_val)

            # ── 9. Console log ────────────────────────────────────────────────
            if ep_count % cfg.log_interval == 0:
                _log(ep_count, global_step, recent, divergence, influence,
                     cm, adaptive_lam, cfg, start_time)

            # ── 10. Evaluation ────────────────────────────────────────────────
            if ep_count % cfg.eval_interval == 0:
                eval_m = evaluate(
                    env_agent, cog_agent, appraisal_mod, cas,
                    influence, cfg, flags, cfg.eval_episodes
                )
                history["eval_reward"].append(eval_m["mean_reward"])
                history["eval_success"].append(eval_m["success_rate"])
                _print_eval(ep_count, eval_m, divergence, cm)

                if eval_m["mean_reward"] > best_eval:
                    best_eval = eval_m["mean_reward"]
                    _save(env_agent, cog_agent, appraisal_mod, cfg, "best")

    # ── Final save ────────────────────────────────────────────────────────────
    _save(env_agent, cog_agent, appraisal_mod, cfg, "final")

    elapsed = time.time() - start_time
    print(f"\n{'='*65}")
    print(f"  Training complete  {elapsed:.0f}s  best_eval={best_eval:+.3f}")
    print(f"{'='*65}\n")

    _plot(history, cfg)
    if cfg.results_dir:
        _save_csvs(history, cfg)
    return history



# ============================================================================ #
#  CSV export                                                                   #
# ============================================================================ #

def _save_csvs(history: dict, cfg) -> None:
    """
    Write two CSV files per training run for later analysis:

    1. {run_name}_train.csv — one row per training episode:
       episode, train_reward, divergence, beta, affective_cost

    2. {run_name}_eval.csv — one row per evaluation checkpoint:
       eval_episode, eval_reward, eval_success,
       avg_divergence, avg_beta, avg_violation, avg_constraint_loss
    """
    import csv as _csv
    os.makedirs(cfg.results_dir, exist_ok=True)

    n_ep = len(history.get("ep_reward", []))

    # ── Training CSV ────────────────────────────────────────────────────────
    train_path = os.path.join(cfg.results_dir, f"{cfg.run_name}_train.csv")
    train_fields = ["episode", "train_reward", "episode_length", "divergence", "beta", "affective_cost"]
    with open(train_path, "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=train_fields)
        w.writeheader()
        for i in range(n_ep):
            def _get(key):
                lst = history.get(key, [])
                return lst[i] if i < len(lst) else ""
            w.writerow({
                "episode":        i + 1,
                "train_reward":   _get("ep_reward"),
                "episode_length": _get("ep_length"),
                "divergence":     _get("divergence"),
                "beta":           _get("beta"),
                "affective_cost": _get("affective_cost"),
            })

    # ── Eval CSV ────────────────────────────────────────────────────────────
    eval_rewards = history.get("eval_reward", [])
    eval_succs   = history.get("eval_success", [])
    n_eval = len(eval_rewards)
    if n_eval == 0:
        return   # nothing to write

    eval_path = os.path.join(cfg.results_dir, f"{cfg.run_name}_eval.csv")
    eval_fields = [
        "eval_episode", "eval_reward", "eval_success",
        "avg_divergence", "avg_beta", "avg_violation", "avg_constraint_loss",
    ]

    # Per-update lists (not per-episode) — map proportionally to eval checkpoints
    viol_list   = history.get("violation_rate",  [])
    con_list    = history.get("constraint_loss", [])

    with open(eval_path, "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=eval_fields)
        w.writeheader()
        for i in range(n_eval):
            ep = (i + 1) * cfg.eval_interval
            # Window of training episodes for this checkpoint
            lo = i * cfg.eval_interval
            hi = min((i + 1) * cfg.eval_interval, n_ep)

            def _win(key):
                vals = history.get(key, [])
                win  = vals[lo:hi]
                return float(np.mean(win)) if win else ""

            # Per-update metrics: map uniformly to eval checkpoints
            def _upd(lst):
                if not lst:
                    return ""
                n_upd = len(lst)
                lo_u  = int(i / n_eval * n_upd)
                hi_u  = int((i + 1) / n_eval * n_upd)
                seg   = lst[lo_u:hi_u]
                return float(np.mean(seg)) if seg else ""

            w.writerow({
                "eval_episode":        ep,
                "eval_reward":         eval_rewards[i],
                "eval_success":        eval_succs[i] if i < len(eval_succs) else "",
                "avg_divergence":      _win("divergence"),
                "avg_beta":            _win("beta"),
                "avg_violation":       _upd(viol_list),
                "avg_constraint_loss": _upd(con_list),
            })

    print(f"  CSVs saved → {cfg.results_dir}/{cfg.run_name}_{{train,eval}}.csv")


# ============================================================================ #
#  Internal helpers                                                             #
# ============================================================================ #


def _run_appraisal_update(appraisal_mod, optim, env_buffer, obs_dim, act_dim):
    """Run one gradient step on the appraisal module using stored observations."""
    n   = env_buffer._n()
    idx = torch.randperm(n)[:64]   # mini-batch
    obs_b      = env_buffer.obs[idx]
    actions_b  = env_buffer.actions[idx]
    rewards_b  = env_buffer.rewards[idx]
    values_b   = env_buffer.values[idx]

    # Approximate next_obs: shift by 1 (last step wraps — acceptable approximation)
    next_idx   = (idx + 1) % n
    next_obs_b = env_buffer.obs[next_idx]

    optim.zero_grad()
    loss, _ = appraisal_mod.compute_loss(
        obs_b, rewards_b, actions_b, next_obs_b, values_b
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(appraisal_mod.parameters(), 0.5)
    optim.step()
    return loss.item()






def _log(ep, step, recent, divergence, influence, cm, adaptive_lam, cfg, start_time):
    elapsed = time.time() - start_time
    r   = f"{np.mean(recent['reward']):+7.3f}" if recent["reward"] else "    —  "
    suc = f"{np.mean(recent['success']):.2f}"  if recent["success"] else " — "
    D   = f"{np.mean(recent['D']):.3f}"        if recent["D"]       else " — "
    b   = f"{np.mean(recent['beta']):.3f}"     if recent["beta"]    else " — "
    vr  = f"{cm.violation_rate():.2f}"         if cm else " — "
    lam = f"{adaptive_lam.value:.3f}"          if adaptive_lam else f"{cfg.lambda_coef:.3f}"

    print(
        f"ep {ep:>5} | rew {r} | succ {suc} | "
        f"D {D} | β {b} | viol {vr} | λ {lam} | "
        f"steps {step:>7} | {elapsed:.0f}s"
    )


def _print_eval(ep, m, divergence, cm):
    div_str  = f"  D={m.get('mean_divergence', 0):.3f}" if "mean_divergence" in m else ""
    cost_str = f"  C={m.get('mean_cost', 0):.3f}"       if "mean_cost" in m else ""
    vr_str   = f"  viol={cm.violation_rate():.2f}"       if cm else ""
    print(
        f"\n  ── EVAL @ ep {ep} {'─'*40}\n"
        f"  reward={m['mean_reward']:+.3f} ± {m['std_reward']:.3f}  "
        f"success={m['success_rate']:.2f}{div_str}{cost_str}{vr_str}\n"
    )


def _save(env_agent, cog_agent, appraisal_mod, cfg, tag):
    base = os.path.join(cfg.save_dir, f"{cfg.run_name}_{cfg.mode}_{tag}")
    env_agent.save(f"{base}_env.pt")
    if cog_agent:
        cog_agent.save(f"{base}_cog.pt")
    if appraisal_mod:
        torch.save(appraisal_mod.state_dict(), f"{base}_appraisal.pt")


def _plot(history, cfg):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        fig = plt.figure(figsize=(16, 10))
        fig.suptitle(f"AC-PPO — mode={cfg.mode}  seed={cfg.seed}", fontsize=13)
        gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

        def smooth(vals, w=20):
            if len(vals) < w:
                return vals
            k = np.ones(w) / w
            return np.convolve(vals, k, mode="valid").tolist()

        plots = [
            ("ep_reward",       "Episode Reward",       0, 0, "steelblue"),
            ("goal_rate",       "Goal Success Rate",    0, 1, "green"),
            ("divergence",      "JS Divergence D_t",    0, 2, "darkorange"),
            ("beta",            "Influence β_t",        1, 0, "purple"),
            ("affective_cost",  "Affective Cost C_t",   1, 1, "crimson"),
            ("violation_rate",  "Violation Rate",       1, 2, "firebrick"),
            ("env_loss",        "PPO-Env Loss",         2, 0, "navy"),
            ("cog_loss",        "PPO-Cog Loss",         2, 1, "teal"),
            ("constraint_loss", "Constraint Loss",      2, 2, "darkred"),
        ]

        for key, title, row, col, colour in plots:
            ax   = fig.add_subplot(gs[row, col])
            vals = history.get(key, [])
            if not vals:
                ax.set_title(f"{title} (no data)")
                continue
            ax.plot(vals, alpha=0.25, color=colour)
            s = smooth(vals)
            if s:
                ax.plot(range(len(vals) - len(s), len(vals)), s, color=colour, linewidth=1.5)
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("Episode", fontsize=7)
            ax.tick_params(labelsize=7)

        # Overlay eval rewards on episode reward plot
        if history["eval_reward"]:
            ax_r = fig.add_subplot(gs[0, 0])
            x_ev = [i * cfg.eval_interval for i in range(len(history["eval_reward"]))]
            ax_r.plot(x_ev, history["eval_reward"], "o--", color="gold",
                      markersize=4, linewidth=1.2, label="Eval")
            ax_r.legend(fontsize=7)

        path = os.path.join(cfg.save_dir, f"{cfg.run_name}_{cfg.mode}_curves.png")
        plt.savefig(path, dpi=130, bbox_inches="tight")
        plt.close()
        print(f"  Training curves → {path}")
    except ImportError:
        print("  (matplotlib not found — skipping plots)")


# ============================================================================ #
#  CLI                                                                          #
# ============================================================================ #

def parse_args() -> FullConfig:
    p = argparse.ArgumentParser(description="AC-PPO Full Training")
    p.add_argument("--env",        type=str,   default="CartPole-v1",
                   choices=["CartPole-v1", "LunarLander-v3", "MountainCar-v0"],
                   help="Gymnasium environment to train on")
    p.add_argument("--mode",       type=str,   default="full",
                   choices=["baseline","affect_reward","appraisal_only","full"])
    p.add_argument("--episodes",   type=int,   default=1_000)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--lambda_coef",type=float, default=0.1)
    p.add_argument("--tau",        type=float, default=0.2)
    p.add_argument("--beta_max",   type=float, default=0.5)
    p.add_argument("--beta_r",     type=float, default=0.5,  help="env reward weight in R_cog")
    p.add_argument("--d",          type=float, default=0.3)
    p.add_argument("--adaptive_lambda", action="store_true")
    p.add_argument("--device",     type=str,   default="cpu")
    p.add_argument("--run_name",   type=str,   default="ac_ppo")
    p.add_argument("--render",     action="store_true")
    a = p.parse_args()

    return FullConfig(
        env_name=a.env,
        mode=a.mode, total_episodes=a.episodes, seed=a.seed, lr=a.lr,
        lambda_coef=a.lambda_coef, tau=a.tau, beta_max=a.beta_max,
        beta_r=a.beta_r,
        d_threshold=a.d, adaptive_lambda=a.adaptive_lambda,
        device=a.device, run_name=a.run_name, render=a.render,
    )


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
