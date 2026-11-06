"""
train_baseline.py

Phase 1 — Baseline PPO Training

Validates that:
  1. DecisionEnv runs correctly
  2. Standard PPO learns a stable policy
  3. Evaluation metrics are collected properly

This is the foundation every later phase (appraisal, cognitive PPO,
constraints) will be benchmarked against.

Usage:
    python train_baseline.py                 # default config
    python train_baseline.py --episodes 500  # quick smoke test
    python train_baseline.py --render        # visual output

Dependencies:
    pip install torch gymnasium numpy matplotlib
"""

import argparse
import time
import os
import sys
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# ── local import ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from envs.decision_env import DecisionEnv


# ============================================================================ #
#  Config                                                                       #
# ============================================================================ #

@dataclass
class Config:
    # Environment
    max_steps:        int   = 200
    risk_threshold:   float = 0.75
    seed:             int   = 42

    # Training
    total_episodes:   int   = 1_000
    rollout_steps:    int   = 512      # steps collected before each PPO update
    ppo_epochs:       int   = 4        # passes over the rollout buffer
    mini_batch_size:  int   = 64

    # PPO hyperparameters
    lr:               float = 3e-4
    gamma:            float = 0.99
    gae_lambda:       float = 0.95
    clip_eps:         float = 0.2
    entropy_coef:     float = 0.01
    value_coef:       float = 0.5
    max_grad_norm:    float = 0.5

    # Network
    hidden_size:      int   = 128

    # Logging
    log_interval:     int   = 20       # episodes between console prints
    eval_interval:    int   = 100      # episodes between evaluation runs
    eval_episodes:    int   = 10

    # Saving
    save_dir:         str   = "checkpoints"
    run_name:         str   = "baseline_ppo"

    # Misc
    render:           bool  = False
    device:           str   = "cpu"    # "cuda" if GPU available


# ============================================================================ #
#  Networks                                                                     #
# ============================================================================ #

class ActorCritic(nn.Module):
    """
    Shared-trunk actor-critic network for PPO.

    Architecture:
        obs (7) → Linear → ReLU → Linear → ReLU
                  ↓                          ↓
              actor head               critic head
         (action logits)            (state value)
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()

        # Shared feature extractor
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        # Actor head — outputs logits over discrete actions
        self.actor_head  = nn.Linear(hidden, act_dim)

        # Critic head — outputs scalar value estimate
        self.critic_head = nn.Linear(hidden, 1)

        self._init_weights()

    def _init_weights(self):
        """Orthogonal initialisation as recommended for PPO."""
        for layer in self.trunk:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor_head.weight,  gain=0.01)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
        nn.init.zeros_(self.actor_head.bias)
        nn.init.zeros_(self.critic_head.bias)

    def forward(self, obs: torch.Tensor):
        features = self.trunk(obs)
        logits   = self.actor_head(features)
        value    = self.critic_head(features).squeeze(-1)
        return logits, value

    def get_action_and_value(self, obs: torch.Tensor, action=None):
        logits, value = self.forward(obs)
        dist          = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy  = dist.entropy()
        return action, log_prob, entropy, value


# ============================================================================ #
#  Rollout Buffer                                                               #
# ============================================================================ #

class RolloutBuffer:
    """Stores a fixed-length rollout and computes GAE returns."""

    def __init__(self, capacity: int, obs_dim: int, device: str):
        self.capacity  = capacity
        self.obs_dim   = obs_dim
        self.device    = device
        self.ptr       = 0
        self.full      = False

        self.obs       = torch.zeros(capacity, obs_dim)
        self.actions   = torch.zeros(capacity, dtype=torch.long)
        self.log_probs = torch.zeros(capacity)
        self.rewards   = torch.zeros(capacity)
        self.values    = torch.zeros(capacity)
        self.dones     = torch.zeros(capacity)

    def store(self, obs, action, log_prob, reward, value, done):
        i = self.ptr % self.capacity
        self.obs[i]       = torch.as_tensor(obs,      dtype=torch.float32)
        self.actions[i]   = torch.as_tensor(action,   dtype=torch.long)
        self.log_probs[i] = torch.as_tensor(log_prob, dtype=torch.float32)
        self.rewards[i]   = torch.as_tensor(reward,   dtype=torch.float32)
        self.values[i]    = torch.as_tensor(value,    dtype=torch.float32)
        self.dones[i]     = torch.as_tensor(done,     dtype=torch.float32)
        self.ptr += 1
        if self.ptr >= self.capacity:
            self.full = True

    def ready(self) -> bool:
        return self.full or self.ptr >= self.capacity

    def compute_returns_and_advantages(
        self,
        last_value: float,
        gamma: float,
        gae_lambda: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generalised Advantage Estimation (GAE)."""
        n          = self.ptr if not self.full else self.capacity
        returns    = torch.zeros(n)
        advantages = torch.zeros(n)

        gae = 0.0
        for t in reversed(range(n)):
            next_value = last_value if t == n - 1 else self.values[t + 1].item()
            next_done  = self.dones[t].item()
            delta      = (
                self.rewards[t].item()
                + gamma * next_value * (1.0 - next_done)
                - self.values[t].item()
            )
            gae = delta + gamma * gae_lambda * (1.0 - next_done) * gae
            advantages[t] = gae
            returns[t]    = gae + self.values[t]

        return returns, advantages

    def get_batches(
        self,
        returns: torch.Tensor,
        advantages: torch.Tensor,
        mini_batch_size: int,
    ):
        """Yield randomly shuffled mini-batches."""
        n       = self.ptr if not self.full else self.capacity
        indices = torch.randperm(n)
        for start in range(0, n, mini_batch_size):
            idx = indices[start : start + mini_batch_size]
            yield (
                self.obs[idx],
                self.actions[idx],
                self.log_probs[idx],
                returns[idx],
                advantages[idx],
            )

    def reset(self):
        self.ptr  = 0
        self.full = False


# ============================================================================ #
#  PPO Agent                                                                    #
# ============================================================================ #

class PPOAgent:
    def __init__(self, obs_dim: int, act_dim: int, cfg: Config):
        self.cfg     = cfg
        self.device  = torch.device(cfg.device)
        self.network = ActorCritic(obs_dim, act_dim, cfg.hidden_size).to(self.device)
        self.optim   = optim.Adam(self.network.parameters(), lr=cfg.lr, eps=1e-5)
        self.buffer  = RolloutBuffer(cfg.rollout_steps, obs_dim, cfg.device)

    @torch.no_grad()
    def select_action(self, obs: np.ndarray):
        obs_t = torch.as_tensor(obs, dtype=torch.float32).to(self.device)
        action, log_prob, _, value = self.network.get_action_and_value(obs_t)
        return (
            action.item(),
            log_prob.item(),
            value.item(),
        )

    def store(self, obs, action, log_prob, reward, value, done):
        self.buffer.store(obs, action, log_prob, reward, value, done)

    def update(self, last_obs: np.ndarray) -> dict:
        """Run PPO update over the collected rollout."""
        cfg = self.cfg

        # Bootstrap value for incomplete trajectory
        with torch.no_grad():
            obs_t      = torch.as_tensor(last_obs, dtype=torch.float32).to(self.device)
            _, last_val = self.network.forward(obs_t)
            last_value  = last_val.item()

        returns, advantages = self.buffer.compute_returns_and_advantages(
            last_value, cfg.gamma, cfg.gae_lambda
        )
        # Normalise advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO epochs
        total_loss_val    = 0.0
        total_policy_loss = 0.0
        total_value_loss  = 0.0
        total_entropy     = 0.0
        n_updates         = 0

        for _ in range(cfg.ppo_epochs):
            for obs_b, act_b, logp_old_b, ret_b, adv_b in self.buffer.get_batches(
                returns, advantages, cfg.mini_batch_size
            ):
                obs_b      = obs_b.to(self.device)
                act_b      = act_b.to(self.device)
                logp_old_b = logp_old_b.to(self.device)
                ret_b      = ret_b.to(self.device)
                adv_b      = adv_b.to(self.device)

                _, logp_new, entropy, value_pred = self.network.get_action_and_value(
                    obs_b, act_b
                )

                # Policy loss (clipped surrogate)
                ratio       = torch.exp(logp_new - logp_old_b)
                surr1       = ratio * adv_b
                surr2       = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (clipped)
                value_loss  = 0.5 * (value_pred - ret_b).pow(2).mean()

                # Entropy bonus
                entropy_loss = -entropy.mean()

                loss = (
                    policy_loss
                    + cfg.value_coef  * value_loss
                    + cfg.entropy_coef * entropy_loss
                )

                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), cfg.max_grad_norm)
                self.optim.step()

                total_loss_val    += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss  += value_loss.item()
                total_entropy     += entropy.mean().item()
                n_updates         += 1

        self.buffer.reset()

        return {
            "loss":         total_loss_val    / max(n_updates, 1),
            "policy_loss":  total_policy_loss / max(n_updates, 1),
            "value_loss":   total_value_loss  / max(n_updates, 1),
            "entropy":      total_entropy     / max(n_updates, 1),
        }

    def save(self, path: str):
        torch.save(
            {
                "network": self.network.state_dict(),
                "optim":   self.optim.state_dict(),
            },
            path,
        )

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.network.load_state_dict(ckpt["network"])
        self.optim.load_state_dict(ckpt["optim"])


# ============================================================================ #
#  Evaluation                                                                   #
# ============================================================================ #

def evaluate(agent: PPOAgent, cfg: Config, n_episodes: int = 10) -> dict:
    """Run greedy evaluation and return summary metrics."""
    env      = DecisionEnv(max_steps=cfg.max_steps, risk_threshold=cfg.risk_threshold)
    rewards  = []
    lengths  = []
    successes = []
    risk_avg  = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=cfg.seed + 1000 + ep)
        ep_reward  = 0.0
        ep_length  = 0
        ep_risk    = []
        done       = False

        while not done:
            with torch.no_grad():
                obs_t  = torch.as_tensor(obs, dtype=torch.float32)
                logits, _ = agent.network.forward(obs_t)
                action = logits.argmax().item()   # greedy

            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_length += 1
            ep_risk.append(info["risk_level"])
            done = terminated or truncated

        rewards.append(ep_reward)
        lengths.append(ep_length)
        successes.append(float(info.get("goal_reached", False)))
        risk_avg.append(float(np.mean(ep_risk)))

    return {
        "mean_reward":   float(np.mean(rewards)),
        "std_reward":    float(np.std(rewards)),
        "mean_length":   float(np.mean(lengths)),
        "success_rate":  float(np.mean(successes)),
        "mean_risk":     float(np.mean(risk_avg)),
    }


# ============================================================================ #
#  Training loop                                                                #
# ============================================================================ #

def train(cfg: Config):
    # ---- reproducibility -------------------------------------------------- #
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # ---- environment ------------------------------------------------------- #
    env = DecisionEnv(
        max_steps=cfg.max_steps,
        risk_threshold=cfg.risk_threshold,
        seed=cfg.seed,
    )
    obs_dim = env.observation_space.shape[0]
    act_dim = int(env.action_space.n)

    # ---- agent ------------------------------------------------------------- #
    agent = PPOAgent(obs_dim, act_dim, cfg)
    print(f"\n{'='*60}")
    print(f"  Baseline PPO  |  obs={obs_dim}  act={act_dim}")
    print(f"  Device: {cfg.device}  |  Total episodes: {cfg.total_episodes}")
    print(f"{'='*60}\n")

    # ---- logging ----------------------------------------------------------- #
    os.makedirs(cfg.save_dir, exist_ok=True)
    history = {
        "episode_rewards":  [],
        "episode_lengths":  [],
        "goal_rates":       [],
        "eval_rewards":     [],
        "losses":           [],
    }

    recent_rewards  = deque(maxlen=cfg.log_interval)
    recent_lengths  = deque(maxlen=cfg.log_interval)
    recent_successes= deque(maxlen=cfg.log_interval)
    global_step     = 0
    start_time      = time.time()

    obs, _     = env.reset()
    ep_reward  = 0.0
    ep_length  = 0
    ep_count   = 0
    best_eval  = -np.inf

    # ---- main loop --------------------------------------------------------- #
    while ep_count < cfg.total_episodes:
        action, log_prob, value = agent.select_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        agent.store(obs, action, log_prob, reward, value, float(done))

        obs         = next_obs
        ep_reward  += reward
        ep_length  += 1
        global_step += 1

        if done:
            ep_count += 1
            recent_rewards.append(ep_reward)
            recent_lengths.append(ep_length)
            recent_successes.append(float(info.get("goal_reached", False)))
            history["episode_rewards"].append(ep_reward)
            history["episode_lengths"].append(ep_length)
            history["goal_rates"].append(float(info.get("goal_reached", False)))

            if cfg.render:
                env.render()

            obs, _    = env.reset()
            ep_reward = 0.0
            ep_length = 0

            # ---- PPO update (when buffer full) ----------------------------- #
            if agent.buffer.ready():
                loss_info = agent.update(obs)
                history["losses"].append(loss_info["loss"])
            else:
                loss_info = {}

            # ---- console log ----------------------------------------------- #
            if ep_count % cfg.log_interval == 0:
                elapsed = time.time() - start_time
                mean_r  = np.mean(recent_rewards)  if recent_rewards  else 0.0
                mean_l  = np.mean(recent_lengths)  if recent_lengths  else 0.0
                succ    = np.mean(recent_successes) if recent_successes else 0.0
                loss_s  = f"{loss_info.get('loss', 0.0):.4f}" if loss_info else "—"

                print(
                    f"ep {ep_count:>5} | "
                    f"rew {mean_r:>+7.3f} | "
                    f"len {mean_l:>5.1f} | "
                    f"succ {succ:.2f} | "
                    f"loss {loss_s} | "
                    f"steps {global_step:>7} | "
                    f"time {elapsed:.0f}s"
                )

            # ---- evaluation ------------------------------------------------ #
            if ep_count % cfg.eval_interval == 0:
                eval_metrics = evaluate(agent, cfg, cfg.eval_episodes)
                history["eval_rewards"].append(eval_metrics["mean_reward"])
                print(
                    f"\n  ── EVAL @ ep {ep_count} ──────────────────────────────\n"
                    f"  mean_reward={eval_metrics['mean_reward']:+.3f}  "
                    f"std={eval_metrics['std_reward']:.3f}  "
                    f"success={eval_metrics['success_rate']:.2f}  "
                    f"risk={eval_metrics['mean_risk']:.3f}\n"
                )

                # Save best model
                if eval_metrics["mean_reward"] > best_eval:
                    best_eval = eval_metrics["mean_reward"]
                    save_path = os.path.join(cfg.save_dir, f"{cfg.run_name}_best.pt")
                    agent.save(save_path)
                    print(f"  ✓ Best model saved ({best_eval:+.3f}) → {save_path}\n")

    # ---- final save -------------------------------------------------------- #
    final_path = os.path.join(cfg.save_dir, f"{cfg.run_name}_final.pt")
    agent.save(final_path)

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  Training complete in {total_time:.1f}s")
    print(f"  Best eval reward: {best_eval:+.3f}")
    print(f"  Model saved → {final_path}")
    print(f"{'='*60}\n")

    _plot_training_curves(history, cfg)

    return agent, history


# ============================================================================ #
#  Plotting                                                                    #
# ============================================================================ #

def _plot_training_curves(history: dict, cfg: Config):
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f"Baseline PPO — {cfg.run_name}", fontsize=14)

        # Episode rewards
        ax = axes[0, 0]
        rewards = history["episode_rewards"]
        ax.plot(rewards, alpha=0.3, color="steelblue", label="Episode reward")
        # Smoothed
        if len(rewards) >= 20:
            kernel = np.ones(20) / 20
            smooth = np.convolve(rewards, kernel, mode="valid")
            ax.plot(range(19, len(rewards)), smooth, color="steelblue", label="Smoothed")
        ax.set_title("Episode Reward")
        ax.set_xlabel("Episode")
        ax.legend()

        # Success rate
        ax = axes[0, 1]
        goal_rates = history["goal_rates"]
        if len(goal_rates) >= 20:
            kernel     = np.ones(20) / 20
            succ_smooth = np.convolve(goal_rates, kernel, mode="valid")
            ax.plot(range(19, len(goal_rates)), succ_smooth, color="green")
        ax.set_title("Goal Success Rate (smoothed)")
        ax.set_xlabel("Episode")
        ax.set_ylim(0, 1)

        # Eval rewards
        ax = axes[1, 0]
        eval_r = history["eval_rewards"]
        ax.plot(
            [i * cfg.eval_interval for i in range(len(eval_r))],
            eval_r,
            marker="o",
            color="darkorange",
        )
        ax.set_title("Eval Mean Reward")
        ax.set_xlabel("Episode")

        # Loss
        ax = axes[1, 1]
        losses = history["losses"]
        ax.plot(losses, color="crimson", alpha=0.7)
        ax.set_title("PPO Loss")
        ax.set_xlabel("Update step")

        plt.tight_layout()
        plot_path = os.path.join(cfg.save_dir, f"{cfg.run_name}_curves.png")
        plt.savefig(plot_path, dpi=120)
        print(f"  Training curves saved → {plot_path}")
        plt.close()

    except ImportError:
        print("  (matplotlib not installed — skipping plots)")


# ============================================================================ #
#  CLI                                                                         #
# ============================================================================ #

def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Baseline PPO for DecisionEnv")
    parser.add_argument("--episodes",    type=int,   default=1_000)
    parser.add_argument("--rollout",     type=int,   default=512)
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--render",      action="store_true")
    parser.add_argument("--device",      type=str,   default="cpu")
    parser.add_argument("--run_name",    type=str,   default="baseline_ppo")
    args = parser.parse_args()

    return Config(
        total_episodes=args.episodes,
        rollout_steps=args.rollout,
        lr=args.lr,
        seed=args.seed,
        render=args.render,
        device=args.device,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
