"""
Run:
    python demo.py                          # full mode, CartPole-v1, seed 42
    python demo.py --mode baseline          # plain PPO-Env (no appraisal)
    python demo.py --mode affect_reward     # appraisal reward shaping
    python demo.py --mode appraisal_only    # appraisal + cog PPO, no constraint
    python demo.py --mode full              # complete AC-PPO (default)
    python demo.py --env LunarLander-v3
    python demo.py --env Acrobot-v1 --seed 0
    python demo.py --no-render              # no Gymnasium window
    python demo.py --delay 0.15            # slower dashboard
"""

# stdlib
import argparse
import os
import sys
import time

# path setup
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# project imports (reusing the full existing implementation)
import torch
import numpy as np
from torch.distributions import Categorical

from train_full_model            import FullConfig, mode_flags
from agents.ppo_agent            import PPOEnvAgent, PPOCogAgent
from modules.appraisal           import AppraisalModule
from modules.cognitive_state     import CognitiveAffectiveState
from modules.cognitive_reward    import CognitiveReward
from modules.divergence          import PolicyDivergence
from modules.behaviour_influence import BehaviourInfluence
from modules.constraints         import ConstraintModule
from envs.gym_env                import make_env, SUPPORTED_ENVS

DEMO_ENVS = [env for env in SUPPORTED_ENVS if env != "MountainCar-v0"]


# ==============================================================================
#  ANSI colour helpers
# ==============================================================================

def _supports_color():
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


_USE_COLOR = _supports_color()


class C:
    RESET   = "\033[0m"   if _USE_COLOR else ""
    BOLD    = "\033[1m"   if _USE_COLOR else ""
    DIM     = "\033[2m"   if _USE_COLOR else ""
    CYAN    = "\033[96m"  if _USE_COLOR else ""
    GREEN   = "\033[92m"  if _USE_COLOR else ""
    YELLOW  = "\033[93m"  if _USE_COLOR else ""
    RED     = "\033[91m"  if _USE_COLOR else ""
    MAGENTA = "\033[95m"  if _USE_COLOR else ""
    BLUE    = "\033[94m"  if _USE_COLOR else ""
    WHITE   = "\033[97m"  if _USE_COLOR else ""


# ==============================================================================
#  Checkpoint path resolution
# ==============================================================================

_ENV_FOLDER = {
    "CartPole-v1":    "CartPole_v1",
    "LunarLander-v3": "LunarLander_v3",
    "Acrobot-v1":     "Acrobot_v1",
    "MountainCar-v0": "MountainCar_v0",
}


# Which checkpoint suffix each mode uses in results/<env_folder>/
# Pattern: <mode>_s<seed>_<mode>_best_<suffix>.pt
_MODE_TAG = {
    "baseline":      "baseline",
    "affect_reward": "affect_reward",
    "appraisal_only":"appraisal_only",
    "full":          "full",
}

# Which checkpoints each mode actually needs
_MODE_NEEDS = {
    "baseline":       {"env"},
    "affect_reward":  {"env", "appraisal"},
    "appraisal_only": {"env", "cog", "appraisal"},
    "full":           {"env", "cog", "appraisal"},
}


def resolve_checkpoints(env_name, seed, mode, results_dir):
    folder   = _ENV_FOLDER.get(env_name, env_name.replace("-", "_").replace(".", "_"))
    res_dir  = os.path.join(results_dir, folder)
    ckpt_dir = os.path.join(os.path.dirname(results_dir), "checkpoints")
    tag      = _MODE_TAG[mode]
    needs    = _MODE_NEEDS[mode]

    def _pick(suffix):
        # Priority 1: results subfolder
        p = os.path.join(res_dir,
                         "{tag}_s{seed}_{tag}_best_{suf}.pt".format(
                             tag=tag, seed=seed, suf=suffix))
        if os.path.isfile(p):
            return p
        # Priority 2: generic checkpoints folder
        p2 = os.path.join(ckpt_dir,
                          "ac_ppo_{tag}_best_{suf}.pt".format(tag=tag, suf=suffix))
        if os.path.isfile(p2):
            return p2
        return p   # return primary path so the error message is useful

    return {k: _pick(k) for k in needs}


# ==============================================================================
#  Dashboard helpers
# ==============================================================================

_W = 66   # total dashboard width


def _bar(value, lo, hi, width=20, char="="):
    frac   = max(0.0, min(1.0, (value - lo) / (hi - lo + 1e-9)))
    filled = int(round(frac * width))
    empty  = width - filled
    return "[{}{}]".format(char * filled, "." * empty)


def _header(title):
    pad = max(0, _W - len(title) - 4)
    return "╔══ {}{}{} {}╗".format(C.BOLD + C.CYAN, title, C.RESET, "=" * pad)


def _footer():
    return "╚" + "=" * (_W - 2) + "╝"


def _row(label, value, colour=""):
    label_w = 26
    val_w   = _W - label_w - 5
    lbl     = "{:<{}}".format(label, label_w)
    val_str = "{}{}{}".format(colour, value, C.RESET) if colour else value
    raw_len = len(value)
    pad     = max(0, val_w - raw_len)
    return "║  {}{}{}  {}{}  ║".format(C.BOLD, lbl, C.RESET, val_str, " " * pad)


def _section(title):
    dash = "-" * ((_W - len(title) - 4) // 2)
    return "╟{}  {}{}{}  {}╢".format(dash, C.DIM, title, C.RESET, dash)


def _clear():
    if _USE_COLOR:
        print("\033[2J\033[H", end="", flush=True)


# ==============================================================================
#  Main dashboard print
# ==============================================================================

def print_dashboard(step, action, reward, ep_reward, g, n, p,
                    cas_state, D_t, beta_t, C_t, env_name, act_labels,
                    mode="full", flags=None, clear=True):
    """Live per-step dashboard.  Sections are shown / greyed depending on mode."""
    if flags is None:
        flags = {"use_appraisal": True, "use_cog_ppo": True, "use_constraint": True}
    if clear:
        _clear()

    try:
        act_str = act_labels[action]
    except (IndexError, KeyError):
        act_str = str(action)

    def _g_col(v):  return C.GREEN if v > 0.3  else (C.RED if v < -0.3 else C.YELLOW)
    def _D_col(v):  return C.RED   if v > 0.5  else (C.YELLOW if v > 0.2 else C.GREEN)
    def _b_col(v):  return C.MAGENTA if v > 0.15 else C.CYAN
    def _C_col(v):  return C.RED   if v > 0.20 else (C.YELLOW if v > 0.10 else C.GREEN)
    _NA = "{}n/a  (not active in '{}' mode){}".format(C.DIM, mode, C.RESET)

    lines = [
        _header("  AC-PPO Live Demo  |  {}  [{}]  ".format(env_name, mode)),
        _row("Environment",    env_name,                                  colour=C.CYAN),
        _row("Mode",           mode,                                      colour=C.MAGENTA),
        _row("Step",           "{:>6}".format(step),                      colour=C.WHITE),
        _row("Action",         "{}  ({})".format(action, act_str),        colour=C.YELLOW),
        _row("Step Reward",    "{:>+8.4f}".format(reward),                colour=C.GREEN if reward >= 0 else C.RED),
        _row("Episode Reward", "{:>+8.4f}".format(ep_reward),             colour=C.GREEN if ep_reward >= 0 else C.RED),
    ]

    # Appraisal section
    lines.append(_section("Appraisal  (current step)"))
    if flags["use_appraisal"]:
        lines += [
            _row("Goal Congruence  (g_hat)",
                 "{:>+6.4f}  {}".format(g, _bar(g, -1, 1)),              colour=_g_col(g)),
            _row("Novelty          (n_hat)",
                 "{:>6.4f}  {}".format(n, _bar(n, 0, 2)),                colour=C.YELLOW),
            _row("Coping Potential (p_hat)",
                 "{:>6.4f}  {}".format(p, _bar(p, 0, 1)),                colour=C.CYAN),
        ]
    else:
        lines.append(_row("Appraisal", _NA))

    # Cognitive-Affective State section
    lines.append(_section("Cognitive-Affective State  c_t  (EMA)"))
    if flags["use_appraisal"]:
        lines += [
            _row("c[0]  goal memory",
                 "{:>+6.4f}  {}".format(cas_state[0], _bar(cas_state[0], -1, 1)),
                 colour=_g_col(cas_state[0])),
            _row("c[1]  novelty   ",
                 "{:>6.4f}  {}".format(cas_state[1], _bar(cas_state[1], 0, 2)),
                 colour=C.YELLOW),
            _row("c[2]  coping    ",
                 "{:>6.4f}  {}".format(cas_state[2], _bar(cas_state[2], 0, 1)),
                 colour=C.CYAN),
        ]
    else:
        lines.append(_row("CAS", _NA))

    # Divergence & influence section
    lines.append(_section("Policy Divergence & Behaviour Influence"))
    if flags["use_cog_ppo"]:
        lines += [
            _row("JS Divergence    (D_t)",
                 "{:>6.4f}  {}".format(D_t, _bar(D_t, 0, 1)),            colour=_D_col(D_t)),
            _row("Influence Coeff  (beta_t)",
                 "{:>6.4f}  {}".format(beta_t, _bar(beta_t, 0, 0.3)),    colour=_b_col(beta_t)),
        ]
    else:
        lines.append(_row("JS Divergence / beta", _NA))

    # Constraint section
    lines.append(_section("Constraint Module"))
    if flags["use_constraint"]:
        lines += [
            _row("Constraint Cost  (C_t)",
                 "{:>6.4f}  {}".format(C_t, _bar(C_t, 0, 0.4)),          colour=_C_col(C_t)),
            _row("Threshold  (d)",  "0.20"),
            _row("Violation?",
                 "YES  cost > threshold" if C_t > 0.20 else "no",
                 colour=C.RED if C_t > 0.20 else C.GREEN),
        ]
    else:
        lines.append(_row("Constraint", _NA))

    lines.append(_footer())
    print("\n".join(lines), flush=True)


# ==============================================================================
#  Final summary
# ==============================================================================

def print_summary(ep_reward, steps, success, g_vals, n_vals, p_vals,
                  D_vals, beta_vals, C_vals, env_name):
    _clear()
    w = _W

    def mean(lst):
        return float(np.mean(lst)) if lst else 0.0

    n_viol    = sum(1 for c in C_vals if c > 0.20)
    viol_rate = n_viol / max(len(C_vals), 1) * 100

    print("╔" + "=" * (w - 2) + "╗")
    title = "  EPISODE COMPLETE  --  FINAL SUMMARY"
    print("║{}{}{}{:^{}}{}║".format(
        C.BOLD, C.CYAN, title, w - 2 - len(title) + len(title), w - 2, C.RESET))
    print("╠" + "=" * (w - 2) + "╣")

    def prow(label, val, col=""):
        lbl = "{:<32}".format(label)
        v   = "{}{}{}".format(col, val, C.RESET) if col else val
        pad = max(0, w - 35 - len(val))
        print("║  {}{}{}  {}{}  ║".format(C.BOLD, lbl, C.RESET, v, " " * pad))

    def divider():
        print("╟" + "-" * (w - 2) + "╢")

    prow("Environment",              env_name,             C.CYAN)
    prow("Total Steps",              str(steps),           C.WHITE)
    prow("Episode Reward",           "{:+.4f}".format(ep_reward),
                                                           C.GREEN if ep_reward >= 0 else C.RED)
    prow("Outcome",
         "SUCCESS" if success else "FAILED",
         C.GREEN if success else C.RED)
    divider()
    prow("Avg Goal Congruence  (g)", "{:+.4f}".format(mean(g_vals)),
                                                           C.GREEN if mean(g_vals) > 0 else C.RED)
    prow("Avg Novelty          (n)", "{:.4f}".format(mean(n_vals)),    C.YELLOW)
    prow("Avg Coping Potential (p)", "{:.4f}".format(mean(p_vals)),    C.CYAN)
    prow("Avg JS Divergence  (D_t)", "{:.4f}".format(mean(D_vals)),    C.YELLOW)
    prow("Avg Influence Coeff (bt)", "{:.4f}".format(mean(beta_vals)), C.MAGENTA)
    prow("Avg Constraint Cost (C_t)", "{:.4f}".format(mean(C_vals)),
                                                           C.RED if mean(C_vals) > 0.20 else C.GREEN)
    prow("Constraint Violations",
         "{}/{} ({:.1f}%)".format(n_viol, steps, viol_rate),
         C.RED if viol_rate > 20 else C.GREEN)
    divider()
    prow("Max JS Divergence",  "{:.4f}".format(max(D_vals,     default=0)), C.WHITE)
    prow("Max Influence beta", "{:.4f}".format(max(beta_vals,  default=0)), C.WHITE)
    prow("Max Constraint Cost","{:.4f}".format(max(C_vals,     default=0)), C.WHITE)
    print("╚" + "=" * (w - 2) + "╝")
    print()


# ==============================================================================
#  Action label tables
# ==============================================================================

_ACTION_LABELS = {
    "CartPole-v1":    ["Push Left", "Push Right"],
    "Acrobot-v1":     ["Torque -1", "Torque 0", "Torque +1"],
    "LunarLander-v3": ["Do Nothing", "Fire Left", "Fire Main", "Fire Right"],
    "MountainCar-v0": ["Push Left", "Do Nothing", "Push Right"],
}


# ==============================================================================
#  Component loading
# ==============================================================================

def _ok(msg):
    print("  {}✓{}  {}".format(C.GREEN, C.RESET, msg))

def _fail(msg):
    print("  {}✗{}  {}".format(C.RED, C.RESET, msg))


def load_framework(env_name, seed, mode, results_dir, render):
    """
    Instantiate and load framework components from trained checkpoints.
    Only the components required by `mode` are loaded and returned.
    Returns a dict containing all live objects needed for the episode loop.
    """
    print()
    print("{}{}{}".format(C.BOLD + C.CYAN, "=" * _W, C.RESET))
    title = "  AC-PPO  Cognitive-Affective Constraint Framework"
    print("{}{:^{}}{}".format(C.BOLD, title, _W, C.RESET))
    sub   = "  Final Year Project -- Evaluation Demo  [mode: {}]".format(mode)
    print("{}{:^{}}{}".format(C.DIM, sub, _W, C.RESET))
    print("{}{}{}".format(C.BOLD + C.CYAN, "=" * _W, C.RESET))
    print()

    # Resolve checkpoint paths for this mode
    paths = resolve_checkpoints(env_name, seed, mode, results_dir)
    flags = mode_flags(mode)

    print("{}[ Checkpoint Verification  (mode={}) ]{}".format(C.BOLD, mode, C.RESET))
    for name, path in paths.items():
        exists = os.path.isfile(path)
        if exists:
            _ok("{:<12}  {}".format(name, path))
        else:
            _fail("{:<12}  {}".format(name, path))
            raise FileNotFoundError(
                "\nCheckpoint not found: {}\n"
                "Ensure results/ is present in {} and seed={} was trained "
                "with mode='{}'.\n".format(path, results_dir, seed, mode)
            )
    print()

    # Configuration
    cfg = FullConfig(
        env_name      = env_name,
        seed          = seed,
        mode          = mode,
        tau           = 0.10,
        k_slope       = 10.0,
        beta_max      = 0.30,
        d_threshold   = 0.20,
        lambda_coef   = 0.05,
        alpha_ema     = 0.10,
        device        = "cpu",
        render        = render,
    )

    # Environment
    print("{}[ Building Components  (mode={}) ]{}".format(C.BOLD, mode, C.RESET))
    env = make_env(env_name, seed=seed, render=render)
    _ok("Gymnasium env  '{}'  obs_dim={}  act_dim={}".format(
        env_name, env.obs_dim, env.act_dim))

    OBS_DIM = env.obs_dim
    ACT_DIM = env.act_dim

    ppo_kw = dict(
        act_dim       = ACT_DIM,
        hidden        = cfg.hidden_size,
        lr            = cfg.lr,
        rollout_steps = cfg.rollout_steps,
        ppo_epochs    = cfg.ppo_epochs,
        mini_batch    = cfg.mini_batch,
        gamma         = cfg.gamma,
        gae_lambda    = cfg.gae_lambda,
        clip_eps      = cfg.clip_eps,
        value_coef    = cfg.value_coef,
        entropy_coef  = cfg.entropy_coef,
        max_grad_norm = cfg.max_grad_norm,
        device        = cfg.device,
    )

    # PPO-Env  (always required)
    env_agent = PPOEnvAgent(obs_dim=OBS_DIM, **ppo_kw)
    env_agent.load(paths["env"])
    _ok("PPO-Env agent   loaded  ({:,} params)".format(
        sum(p.numel() for p in env_agent.network.parameters())))

    # PPO-Cog  (appraisal_only and full only)
    cog_agent = None
    if flags["use_cog_ppo"]:
        cog_agent = PPOCogAgent(obs_dim=OBS_DIM, **ppo_kw)
        cog_agent.load(paths["cog"])
        _ok("PPO-Cog agent   loaded  ({:,} params)".format(
            sum(p.numel() for p in cog_agent.network.parameters())))
    else:
        _ok("PPO-Cog agent   SKIPPED  (not used in '{}' mode)".format(mode))

    # Appraisal module  (all modes except baseline)
    appraisal_mod = None
    cas           = None
    cog_reward_fn = None
    if flags["use_appraisal"]:
        appraisal_mod = AppraisalModule(OBS_DIM, ACT_DIM, cfg.mental_dim, cfg.latent_dim)
        try:
            state_dict = torch.load(paths["appraisal"], map_location="cpu",
                                    weights_only=True)
        except TypeError:
            state_dict = torch.load(paths["appraisal"], map_location="cpu")
        appraisal_mod.load_state_dict(state_dict)
        appraisal_mod.eval()
        _ok("Appraisal module loaded  ({:,} params)".format(
            sum(p.numel() for p in appraisal_mod.parameters())))

        cas = CognitiveAffectiveState(alpha=cfg.alpha_ema)
        _ok("CognitiveAffectiveState  alpha={}".format(cfg.alpha_ema))

        cog_reward_fn = CognitiveReward(cfg.beta_r, cfg.alpha_g, cfg.alpha_p)
        _ok("CognitiveReward          beta_r={}  alpha_g={}  alpha_p={}".format(
            cfg.beta_r, cfg.alpha_g, cfg.alpha_p))
    else:
        _ok("Appraisal / CAS / CogReward  SKIPPED  (baseline mode)")

    # Policy divergence + behaviour influence  (cog_ppo modes only)
    divergence = None
    influence  = None
    if flags["use_cog_ppo"]:
        divergence = PolicyDivergence(normalise=True)
        _ok("PolicyDivergence         normalised JS in [0,1]")
        influence = BehaviourInfluence(cfg.tau, cfg.k_slope, cfg.beta_max)
        _ok("BehaviourInfluence       tau={}  k={}  beta_max={}".format(
            cfg.tau, cfg.k_slope, cfg.beta_max))
    else:
        _ok("PolicyDivergence / BehaviourInfluence  SKIPPED  (mode='{}')".format(mode))

    # Constraint module  (full mode only)
    cm = None
    if flags["use_constraint"]:
        cm = ConstraintModule(cfg.d_threshold, cfg.lambda_coef)
        _ok("ConstraintModule         d={}  lambda={}".format(
            cfg.d_threshold, cfg.lambda_coef))
    else:
        _ok("ConstraintModule  SKIPPED  (mode='{}')".format(mode))

    print()
    print("{}{}  Framework loaded  [mode={}].{}".format(
        C.BOLD, C.GREEN, mode, C.RESET))
    print()

    return dict(
        cfg=cfg, flags=flags, mode=mode, env=env,
        env_agent=env_agent, cog_agent=cog_agent,
        appraisal_mod=appraisal_mod, cas=cas,
        cog_reward_fn=cog_reward_fn, divergence=divergence,
        influence=influence, cm=cm,
        obs_dim=OBS_DIM, act_dim=ACT_DIM,
    )


# ==============================================================================
#  Success criterion  (mirrors train_full_model.is_success)
# ==============================================================================

def _is_success(env_name, ep_reward, ep_length, terminated, truncated):
    if "CartPole" in env_name:
        return truncated or ep_length >= 475 or ep_reward >= 475
    elif "Acrobot" in env_name:
        return bool(terminated)
    elif "LunarLander" in env_name:
        return ep_reward >= 200.0 or (terminated and ep_reward > 0)
    return bool(terminated)


# ==============================================================================
#  Evaluation episode loop
# ==============================================================================

def run_episode(fw, clear_screen=True, step_delay=0.04, max_steps=2000):
    """
    Run one complete evaluation episode using the loaded framework.
    Automatically adapts to the active mode (baseline / affect_reward /
    appraisal_only / full) — only calls the components that exist.
    Returns a dict of episode statistics.
    """
    env        = fw["env"]
    env_agent  = fw["env_agent"]
    cog_agent  = fw["cog_agent"]       # None for baseline / affect_reward
    appr       = fw["appraisal_mod"]   # None for baseline
    cas        = fw["cas"]             # None for baseline
    divergence = fw["divergence"]      # None for baseline / affect_reward
    influence  = fw["influence"]       # None for baseline / affect_reward
    cm         = fw["cm"]             # None unless full mode
    flags      = fw["flags"]
    mode       = fw["mode"]
    env_name   = fw["cfg"].env_name
    act_labels = _ACTION_LABELS.get(env_name, [str(i) for i in range(fw["act_dim"])])

    # Reset
    obs, _    = env.reset(seed=fw["cfg"].seed + 9999)
    if cas is not None:
        cas.reset()
    c_t       = torch.zeros(3)
    ep_reward = 0.0
    done      = False
    step      = 0

    # History collectors
    g_vals, n_vals, p_vals = [], [], []
    D_vals, beta_vals, C_vals = [], [], []

    # Track termination flags for success check
    terminated = False
    truncated  = False

    print("{}{}  Starting episode  [mode={}]  ...  Ctrl-C to abort.{}".format(
        C.BOLD, C.CYAN, mode, C.RESET))
    print()
    time.sleep(0.8)

    while not done and step < max_steps:

        obs_t = torch.as_tensor(obs, dtype=torch.float32)

        with torch.no_grad():
            # PPO-Env: always active
            action_env, _, val_env = env_agent.select_action(obs)
            p_env = env_agent.get_probs(obs_t)

            if flags["use_cog_ppo"] and cog_agent is not None:
                # PPO-Cog + behaviour blending
                p_cog   = cog_agent.get_probs(obs_t, c_t)
                D_t     = divergence.compute_scalar(p_env, p_cog)
                divergence.record(D_t)
                beta_t  = influence.compute_beta(D_t)
                p_final = influence.blend(p_env, p_cog, beta_t)
                dist    = Categorical(probs=p_final)
                action  = dist.sample().item()
            else:
                # Pure PPO-Env action
                action  = action_env
                D_t     = 0.0
                beta_t  = 0.0

        # Environment step
        obs, reward, terminated, truncated, info = env.step(action)
        done       = terminated or truncated
        ep_reward += reward
        step      += 1

        # Appraisal pipeline
        g = n = p = 0.0
        cas_state = [0.0, 0.0, 0.0]
        if flags["use_appraisal"] and appr is not None:
            with torch.no_grad():
                appr_out = appr.infer(obs_t, reward, action, val_env)
            g = appr_out["g"]
            n = appr_out["n"]
            p = appr_out["p"]

            if cas is not None:
                a_hat = torch.tensor([g, n, p])
                cas.update(a_hat)
                c_t       = cas.get_flat()
                cas_state = c_t.tolist()

        # Constraint cost
        C_t = 0.0
        if flags["use_constraint"] and cm is not None:
            C_t = cm.compute_cost(beta_t, D_t, reward)
            cm.record_cost(C_t)

        # Collect history
        g_vals.append(g)
        n_vals.append(n)
        p_vals.append(p)
        D_vals.append(D_t)
        beta_vals.append(beta_t)
        C_vals.append(C_t)

        # Live terminal dashboard
        print_dashboard(
            step       = step,
            action     = action,
            reward     = reward,
            ep_reward  = ep_reward,
            g          = g,
            n          = n,
            p          = p,
            cas_state  = cas_state,
            D_t        = D_t,
            beta_t     = beta_t,
            C_t        = C_t,
            env_name   = env_name,
            act_labels = act_labels,
            mode       = mode,
            flags      = flags,
            clear      = clear_screen,
        )

        if step_delay > 0:
            time.sleep(step_delay)

    success = _is_success(env_name, ep_reward, step, terminated, truncated)

    return dict(
        ep_reward  = ep_reward,
        steps      = step,
        success    = success,
        g_vals     = g_vals,
        n_vals     = n_vals,
        p_vals     = p_vals,
        D_vals     = D_vals,
        beta_vals  = beta_vals,
        C_vals     = C_vals,
        terminated = terminated,
        truncated  = truncated,
    )


# ==============================================================================
#  CLI
# ==============================================================================

def _parse_args():
    p = argparse.ArgumentParser(
        description="Demonstration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="",
    )
    p.add_argument("--env",         type=str,   default="CartPole-v1",
                   choices=DEMO_ENVS,
                   help="Gymnasium environment (default: CartPole-v1)")
    p.add_argument("--mode",        type=str,   default="full",
                   choices=["baseline", "affect_reward", "appraisal_only", "full"],
                   help="Which trained variant to run (default: full)")
    p.add_argument("--seed",        type=int,   default=42,
                   help="Random seed -- must match a trained checkpoint (default: 42)")
    p.add_argument("--results-dir", type=str,   default=None,
                   help="Path to results/ directory (default: auto-detect)")
    p.add_argument("--no-render",   action="store_true",
                   help="Disable Gymnasium window (headless mode)")
    p.add_argument("--no-clear",    action="store_true",
                   help="Do not clear the terminal between steps")
    p.add_argument("--delay",       type=float, default=0.04,
                   help="Seconds to sleep after each step (default: 0.04)")
    p.add_argument("--max-steps",   type=int,   default=2000,
                   help="Hard cap on episode length (default: 2000)")
    return p.parse_args()


# ==============================================================================
#  Entry point
# ==============================================================================

def main():
    args = _parse_args()

    # Results directory auto-detection
    if args.results_dir is None:
        default_results = os.path.join(_HERE, "results")
        if not os.path.isdir(default_results):
            print("{}ERROR:{} Could not find results/ directory at:".format(C.RED, C.RESET))
            print("  {}".format(default_results))
            print("Use --results-dir to specify the path explicitly.")
            sys.exit(1)
        results_dir = default_results
    else:
        results_dir = os.path.abspath(args.results_dir)
        if not os.path.isdir(results_dir):
            print("{}ERROR:{} Directory does not exist: {}".format(
                C.RED, C.RESET, results_dir))
            sys.exit(1)

    render = not args.no_render
    fw     = {}

    try:
        fw = load_framework(
            env_name    = args.env,
            seed        = args.seed,
            mode        = args.mode,
            results_dir = results_dir,
            render      = render,
        )

        input("{}  Press ENTER to begin the evaluation episode ...{}".format(
            C.DIM, C.RESET))

        result = run_episode(
            fw,
            clear_screen = not args.no_clear,
            step_delay   = args.delay,
            max_steps    = args.max_steps,
        )

        print_summary(
            ep_reward  = result["ep_reward"],
            steps      = result["steps"],
            success    = result["success"],
            g_vals     = result["g_vals"],
            n_vals     = result["n_vals"],
            p_vals     = result["p_vals"],
            D_vals     = result["D_vals"],
            beta_vals  = result["beta_vals"],
            C_vals     = result["C_vals"],
            env_name   = args.env,
        )

    except KeyboardInterrupt:
        print("\n\n{}  Demo interrupted by user.{}\n".format(C.YELLOW, C.RESET))
        sys.exit(0)
    except FileNotFoundError as e:
        print("\n{}  {}{}\n".format(C.RED, e, C.RESET))
        sys.exit(1)
    finally:
        try:
            fw["env"].close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
