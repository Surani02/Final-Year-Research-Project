import os
import sys

# Add the project root (Research_Implementation/) to the Python path
_ROOT = getattr(sys, "_MEIPASS", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import torch
import gymnasium as gym
from torch.distributions import Categorical

from train_full_model            import FullConfig, mode_flags
from agents.ppo_agent            import PPOEnvAgent, PPOCogAgent
from modules.appraisal           import AppraisalModule
from modules.cognitive_state     import CognitiveAffectiveState
from modules.cognitive_reward    import CognitiveReward
from modules.divergence          import PolicyDivergence
from modules.behaviour_influence import BehaviourInfluence
from modules.constraints         import ConstraintModule

# Reuse checkpoint resolution from demo.py
from demo import resolve_checkpoints

# ── constants ──────────────────────────────────────────────────────────────── #

RESULTS_DIR = os.path.join(_ROOT, "results")

ACTION_LABELS = {
    "CartPole-v1":    ["Push Left", "Push Right"],
    "Acrobot-v1":     ["Torque −1", "Torque 0", "Torque +1"],
    "LunarLander-v3": ["Do Nothing", "Fire Left", "Fire Main", "Fire Right"],
    "MountainCar-v0": ["Push Left", "Do Nothing", "Push Right"],
}

MODE_DISPLAY = {
    "baseline":       "Baseline PPO",
    "affect_reward":  "Affect Reward",
    "appraisal_only": "Appraisal Only",
    "full":           "Full AC-PPO",
}

# ── EpisodeRunner ──────────────────────────────────────────────────────────── #

class EpisodeRunner:
    """
    Step-by-step episode runner.  Call `step()` repeatedly until `done` is True.

    Parameters
    ----------
    env_name    : one of the SUPPORTED_ENVS keys
    mode        : "baseline" | "affect_reward" | "appraisal_only" | "full"
    seed        : integer seed — must match a trained checkpoint
    results_dir : path to the results/ folder (auto-detected if None)
    """

    def __init__(self, env_name: str, mode: str, seed: int, results_dir: str = None):
        self.env_name    = env_name
        self.mode        = mode
        self.seed        = seed
        self.results_dir = results_dir or RESULTS_DIR
        self.flags       = mode_flags(mode)

        self._obs        = None
        self._c_t        = torch.zeros(3)
        self.ep_reward   = 0.0
        self.step_count  = 0
        self._done       = False
        self._terminated = False
        self._truncated  = False
        self._last_frame = None
        self._last_metrics = None

        self._load()
        self.reset()

    # ── initialise all framework components ───────────────────────────────── #

    def _load(self):
        paths = resolve_checkpoints(
            self.env_name, self.seed, self.mode, self.results_dir
        )

        cfg = FullConfig(
            env_name      = self.env_name,
            seed          = self.seed,
            mode          = self.mode,
            tau           = 0.10,
            k_slope       = 10.0,
            beta_max      = 0.30,
            d_threshold   = 0.20,
            lambda_coef   = 0.05,
            alpha_ema     = 0.10,
            device        = "cpu",
            render        = False,
        )
        self.cfg = cfg

        # Gym env with rgb_array rendering (for web frame capture)
        self._gym_env = gym.make(self.env_name, render_mode="rgb_array")
        self.obs_dim  = self._gym_env.observation_space.shape[0]
        self.act_dim  = int(self._gym_env.action_space.n)

        ppo_kw = dict(
            act_dim       = self.act_dim,
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

        # PPO-Env (always required)
        self.env_agent = PPOEnvAgent(obs_dim=self.obs_dim, **ppo_kw)
        self.env_agent.load(paths["env"])

        # PPO-Cog (appraisal_only / full)
        self.cog_agent = None
        if self.flags["use_cog_ppo"]:
            self.cog_agent = PPOCogAgent(obs_dim=self.obs_dim, **ppo_kw)
            self.cog_agent.load(paths["cog"])

        # Appraisal module + CAS + CogReward
        self.appr         = None
        self.cas          = None
        self.cog_reward_fn = None
        if self.flags["use_appraisal"]:
            self.appr = AppraisalModule(
                self.obs_dim, self.act_dim,
                cfg.mental_dim, cfg.latent_dim
            )
            try:
                sd = torch.load(
                    paths["appraisal"], map_location="cpu", weights_only=True
                )
            except TypeError:
                sd = torch.load(paths["appraisal"], map_location="cpu")
            self.appr.load_state_dict(sd)
            self.appr.eval()
            self.cas           = CognitiveAffectiveState(alpha=cfg.alpha_ema)
            self.cog_reward_fn = CognitiveReward(cfg.beta_r, cfg.alpha_g, cfg.alpha_p)

        # Policy divergence + behaviour influence
        self.divergence = None
        self.influence  = None
        if self.flags["use_cog_ppo"]:
            self.divergence = PolicyDivergence(normalise=True)
            self.influence  = BehaviourInfluence(cfg.tau, cfg.k_slope, cfg.beta_max)

        # Constraint module
        self.cm = None
        if self.flags["use_constraint"]:
            self.cm = ConstraintModule(cfg.d_threshold, cfg.lambda_coef)

    # ── public API ────────────────────────────────────────────────────────── #

    def reset(self):
        """Reset the episode.  Returns the initial frame as a numpy array."""
        obs, _ = self._gym_env.reset(seed=self.seed + 9999)
        self._obs        = obs.astype(np.float32)
        self._c_t        = torch.zeros(3)
        if self.cas is not None:
            self.cas.reset()
        self.ep_reward   = 0.0
        self.step_count  = 0
        self._done       = False
        self._terminated = False
        self._truncated  = False
        frame = self._gym_env.render()
        self._last_frame = frame
        self._last_metrics = self._make_metrics(0.0, 0, 0.0, 0.0, 0.0,
                                                 [0., 0., 0.], 0.0, 0.0, 0.0)
        return frame

    def step(self):
        """
        Execute one environment step.

        Returns
        -------
        frame   : np.ndarray (H, W, 3) — the rendered frame after the step
        metrics : dict — all cognitive + reward metrics for this step
        """
        if self._done:
            # Episode already finished — return last frame/metrics frozen
            return self._last_frame, self._last_metrics

        obs_t = torch.as_tensor(self._obs, dtype=torch.float32)

        with torch.no_grad():
            action_env, _, val_env = self.env_agent.select_action(self._obs)
            p_env = self.env_agent.get_probs(obs_t)

            if self.flags["use_cog_ppo"] and self.cog_agent is not None:
                p_cog   = self.cog_agent.get_probs(obs_t, self._c_t)
                D_t     = self.divergence.compute_scalar(p_env, p_cog)
                self.divergence.record(D_t)
                beta_t  = self.influence.compute_beta(D_t)
                p_final = self.influence.blend(p_env, p_cog, beta_t)
                dist    = Categorical(probs=p_final)
                action  = dist.sample().item()
            else:
                action = action_env
                D_t    = 0.0
                beta_t = 0.0

        # Step environment
        obs, reward, self._terminated, self._truncated, _ = self._gym_env.step(action)
        self._obs      = obs.astype(np.float32)
        self._done     = self._terminated or self._truncated
        self.ep_reward += reward
        self.step_count += 1

        # Appraisal pipeline
        g = n = p = 0.0
        cas_state = [0.0, 0.0, 0.0]
        if self.flags["use_appraisal"] and self.appr is not None:
            with torch.no_grad():
                appr_out = self.appr.infer(obs_t, reward, action, val_env)
            g, n, p = appr_out["g"], appr_out["n"], appr_out["p"]
            if self.cas is not None:
                self.cas.update(torch.tensor([g, n, p]))
                self._c_t = self.cas.get_flat()
                cas_state = self._c_t.tolist()

        # Constraint cost
        C_t = 0.0
        if self.flags["use_constraint"] and self.cm is not None:
            C_t = self.cm.compute_cost(beta_t, D_t, reward)
            self.cm.record_cost(C_t)

        frame   = self._gym_env.render()
        metrics = self._make_metrics(reward, action, g, n, p,
                                     cas_state, D_t, beta_t, C_t)
        self._last_frame   = frame
        self._last_metrics = metrics
        return frame, metrics

    def close(self):
        try:
            self._gym_env.close()
        except Exception:
            pass

    @property
    def done(self) -> bool:
        return self._done

    # ── internal helpers ──────────────────────────────────────────────────── #

    def _make_metrics(self, reward, action, g, n, p,
                      cas_state, D_t, beta_t, C_t) -> dict:
        labels = ACTION_LABELS.get(self.env_name,
                                   [str(i) for i in range(self.act_dim)])
        act_label = labels[action] if 0 <= action < len(labels) else str(action)
        return {
            "step":        self.step_count,
            "action":      action,
            "action_label": act_label,
            "reward":      float(reward),
            "ep_reward":   float(self.ep_reward),
            "g":           float(g),
            "n":           float(n),
            "p":           float(p),
            "cas":         [float(x) for x in cas_state],
            "D_t":         float(D_t),
            "beta_t":      float(beta_t),
            "C_t":         float(C_t),
            "done":        self._done,
            "terminated":  self._terminated,
            "truncated":   self._truncated,
            "mode":        self.mode,
            "mode_display": MODE_DISPLAY[self.mode],
            "flags":       self.flags,
        }
