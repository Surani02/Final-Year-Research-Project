import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.networks import ActorCritic


# ============================================================================ #
#  Rollout Buffer                                                               #
# ============================================================================ #

class RolloutBuffer:
    def __init__(self, capacity: int, input_dim: int, device: str = "cpu"):
        self.capacity  = capacity
        self.input_dim = input_dim
        self.device    = device
        self.ptr       = 0
        self.full      = False

        self.obs       = torch.zeros(capacity, input_dim)
        self.actions   = torch.zeros(capacity, dtype=torch.long)
        self.log_probs = torch.zeros(capacity)
        self.rewards   = torch.zeros(capacity)
        self.values    = torch.zeros(capacity)
        self.dones     = torch.zeros(capacity)

    def store(
        self,
        obs:      np.ndarray | torch.Tensor,
        action:   int,
        log_prob: float,
        reward:   float,
        value:    float,
        done:     bool,
    ) -> None:
        i = self.ptr % self.capacity
        self.obs[i]       = torch.as_tensor(obs,      dtype=torch.float32)
        self.actions[i]   = int(action)
        self.log_probs[i] = float(log_prob)
        self.rewards[i]   = float(reward)
        self.values[i]    = float(value)
        self.dones[i]     = float(done)
        self.ptr += 1
        if self.ptr >= self.capacity:
            self.full = True

    def ready(self) -> bool:
        return self.full or self.ptr >= self.capacity

    def _n(self) -> int:
        return self.capacity if self.full else self.ptr

    def compute_gae(
        self,
        last_value: float,
        gamma:      float = 0.99,
        gae_lambda: float = 0.95,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n          = self._n()
        returns    = torch.zeros(n)
        advantages = torch.zeros(n)
        gae        = 0.0

        for t in reversed(range(n)):
            next_val  = last_value if t == n - 1 else self.values[t + 1].item()
            next_done = self.dones[t].item()
            delta     = (
                self.rewards[t].item()
                + gamma * next_val * (1.0 - next_done)
                - self.values[t].item()
            )
            gae          = delta + gamma * gae_lambda * (1.0 - next_done) * gae
            advantages[t] = gae
            returns[t]    = gae + self.values[t]

        return returns, advantages

    def get_batches(
        self,
        returns:        torch.Tensor,
        advantages:     torch.Tensor,
        mini_batch_size: int,
    ):
        n       = self._n()
        indices = torch.randperm(n)
        for start in range(0, n, mini_batch_size):
            idx = indices[start : start + mini_batch_size]
            yield (
                idx,                    # raw indices — callers may use to align aux buffers
                self.obs[idx],
                self.actions[idx],
                self.log_probs[idx],
                returns[idx],
                advantages[idx],
            )

    def reset(self) -> None:
        self.ptr  = 0
        self.full = False


# ============================================================================ #
#  Base PPO Agent                                                               #
# ============================================================================ #

class _BasePPOAgent:
    def __init__(
        self,
        input_dim:      int,
        act_dim:        int,
        hidden:         int   = 128,
        lr:             float = 3e-4,
        rollout_steps:  int   = 512,
        ppo_epochs:     int   = 4,
        mini_batch:     int   = 64,
        gamma:          float = 0.99,
        gae_lambda:     float = 0.95,
        clip_eps:       float = 0.2,
        value_coef:     float = 0.5,
        entropy_coef:   float = 0.01,
        max_grad_norm:  float = 0.5,
        device:         str   = "cpu",
    ):
        self.act_dim       = act_dim
        self.ppo_epochs    = ppo_epochs
        self.mini_batch    = mini_batch
        self.gamma         = gamma
        self.gae_lambda    = gae_lambda
        self.clip_eps      = clip_eps
        self.value_coef    = value_coef
        self.entropy_coef  = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.device        = torch.device(device)

        self.network = ActorCritic(input_dim, act_dim, hidden).to(self.device)
        self.optim   = torch.optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)
        self.buffer  = RolloutBuffer(rollout_steps, input_dim, device)

    @torch.no_grad()
    def get_value(self, x: torch.Tensor) -> float:
        _, value = self.network.forward(x.to(self.device))
        return value.item()

    def store(self, obs, action, log_prob, reward, value, done):
        self.buffer.store(obs, action, log_prob, reward, value, done)

    def update(self, last_input: torch.Tensor, label: str = "") -> dict:
        """
        Args:
            last_input: the input tensor for the final state (for bootstrapping)
            label:      string tag for logging ("env" or "cog")
        """
        with torch.no_grad():
            last_val = self.get_value(last_input)

        returns, advantages = self.buffer.compute_gae(last_val, self.gamma, self.gae_lambda)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total = dict(loss=0.0, policy_loss=0.0, value_loss=0.0, entropy=0.0)
        n_up  = 0

        for _ in range(self.ppo_epochs):
            for _, obs_b, act_b, logp_old, ret_b, adv_b in self.buffer.get_batches(
                returns, advantages, self.mini_batch
            ):
                obs_b    = obs_b.to(self.device)
                act_b    = act_b.to(self.device)
                logp_old = logp_old.to(self.device)
                ret_b    = ret_b.to(self.device)
                adv_b    = adv_b.to(self.device)

                _, logp_new, entropy, value_pred = self.network.get_action_and_value(obs_b, act_b)

                ratio  = torch.exp(logp_new - logp_old)
                s1     = ratio * adv_b
                s2     = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_b
                p_loss = -torch.min(s1, s2).mean()
                v_loss = 0.5 * (value_pred - ret_b).pow(2).mean()
                e_loss = -entropy.mean()

                loss = p_loss + self.value_coef * v_loss + self.entropy_coef * e_loss

                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optim.step()

                total["loss"]         += loss.item()
                total["policy_loss"]  += p_loss.item()
                total["value_loss"]   += v_loss.item()
                total["entropy"]      += entropy.mean().item()
                n_up += 1

        self.buffer.reset()
        n_up = max(n_up, 1)
        return {f"{label}_{k}": v / n_up for k, v in total.items()}

    def save(self, path: str) -> None:
        torch.save({"network": self.network.state_dict(),
                    "optim":   self.optim.state_dict()}, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.network.load_state_dict(ckpt["network"])
        self.optim.load_state_dict(ckpt["optim"])


# ============================================================================ #
#  PPO-Env                                                                      #
# ============================================================================ #

class PPOEnvAgent(_BasePPOAgent):
    def __init__(self, obs_dim: int, act_dim: int, **kwargs):
        super().__init__(input_dim=obs_dim, act_dim=act_dim, **kwargs)
        self.obs_dim = obs_dim

    @torch.no_grad()
    def select_action(self, obs: np.ndarray) -> tuple[int, float, float]:
        """
        Returns:
            action:   int
            log_prob: float
            value:    float  (used for GAE and for coping pseudo-label)
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32).to(self.device)
        action, log_prob, _, value = self.network.get_action_and_value(obs_t)
        return action.item(), log_prob.item(), value.item()

    @torch.no_grad()
    def get_probs(self, obs: torch.Tensor) -> torch.Tensor:
        """p_base: action probability vector for JS divergence"""
        return self.network.get_probs(obs.to(self.device))

    def update_env(self, last_obs: np.ndarray) -> dict:
        last_t = torch.as_tensor(last_obs, dtype=torch.float32)
        return self.update(last_t, label="env")

    def update_env_with_constraint(
        self,
        last_obs:    np.ndarray,
        cog_network: nn.Module,
        obs_buf:     torch.Tensor,   # (N, obs_dim) — stored observations from rollout
        c_t_buf:     torch.Tensor,   # (N, cog_dim) — cognitive states stored at each step
        rewards_buf: torch.Tensor,   # (N,)          — stored rewards
        lambda_coef: float = 0.1,
        d_threshold: float = 0.3,
        tau:         float = 0.2,
        k_slope:     float = 10.0,
        beta_max:    float = 0.5,
    ) -> dict:
        """
        PPO update for PPO-Env with constraint loss fused into a single backward pass.

        During each mini-batch, this method:
          1. Re-computes p_env and p_cog from the current networks (with gradients).
          2. Computes D_t = JS(p_env, p_cog) as a differentiable tensor.
          3. Computes β_t = β_max · σ(k · (D_t − τ)) — still a tensor.
          4. Computes affective cost C_t = β_t · D_t · |R_t|.
          5. Computes L_constraint = mean(ReLU(C_t − d)).
          6. Fuses: L_total = L_PPO + λ · L_constraint.
          7. Calls a single loss.backward() + optimizer.step().

        This is the PPO-Lagrangian pattern: fully connected computation graph,
        one backward pass, no severed gradients.
        """
        last_t = torch.as_tensor(last_obs, dtype=torch.float32)

        with torch.no_grad():
            last_val = self.get_value(last_t)

        returns, advantages = self.buffer.compute_gae(last_val, self.gamma, self.gae_lambda)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total = dict(loss=0.0, policy_loss=0.0, value_loss=0.0,
                     entropy=0.0, constraint_loss=0.0)
        n_up  = 0
        ln2   = torch.log(torch.tensor(2.0))

        for _ in range(self.ppo_epochs):
            for idx, obs_b, act_b, logp_old, ret_b, adv_b in self.buffer.get_batches(
                returns, advantages, self.mini_batch
            ):
                obs_b    = obs_b.to(self.device)
                act_b    = act_b.to(self.device)
                logp_old = logp_old.to(self.device)
                ret_b    = ret_b.to(self.device)
                adv_b    = adv_b.to(self.device)

                # ── Standard PPO loss ────────────────────────────────────────
                _, logp_new, entropy, value_pred = self.network.get_action_and_value(
                    obs_b, act_b
                )
                ratio  = torch.exp(logp_new - logp_old)
                s1     = ratio * adv_b
                s2     = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_b
                p_loss = -torch.min(s1, s2).mean()
                v_loss = 0.5 * (value_pred - ret_b).pow(2).mean()
                e_loss = -entropy.mean()
                ppo_loss = p_loss + self.value_coef * v_loss + self.entropy_coef * e_loss

                # ── Differentiable constraint loss ───────────────────────────
                # Re-use the same rollout indices as the PPO mini-batch so both
                # objectives see identical data — eliminates cross-batch gradient noise.
                obs_c = obs_buf[idx].to(self.device)       # (B, obs_dim)
                c_t_c = c_t_buf[idx].to(self.device)       # (B, cog_dim)
                r_c   = rewards_buf[idx].to(self.device)   # (B,)

                # p_env: re-run env network WITH gradients
                p_env = self.network.get_probs(obs_c)                            # (B, A)

                # p_cog: run cog network WITHOUT gradients.
                # The constraint penalises the env policy for diverging from the
                # cognitive policy; p_cog is a fixed reference target so that
                # only the env policy is updated by this loss.
                with torch.no_grad():
                    cog_input = torch.cat([obs_c, c_t_c], dim=-1)
                    p_cog     = cog_network.get_probs(cog_input)                 # (B, A)

                # JS divergence — stays as tensor (B,)
                eps = 1e-8
                p_env_c = p_env.clamp(min=eps)
                p_cog_c = p_cog.clamp(min=eps)
                m       = 0.5 * (p_env_c + p_cog_c)
                kl_pm   = F.kl_div(m.log(), p_env_c, reduction="none").sum(-1)
                kl_qm   = F.kl_div(m.log(), p_cog_c, reduction="none").sum(-1)
                D_t     = 0.5 * (kl_pm + kl_qm) / ln2                           # (B,) ∈ [0,1]

                # β_t — scalar multiply avoids repeated torch.tensor allocation
                beta_t = beta_max * torch.sigmoid(k_slope * (D_t - tau))         # (B,)

                # Affective cost C_t = β_t · D_t · |R_t|
                C_t = beta_t * D_t * r_c.abs()                                   # (B,)

                # Hinge constraint loss
                L_con = F.relu(C_t - d_threshold).mean()                         # scalar

                # ── Fused total loss — single backward pass ──────────────────
                total_loss = ppo_loss + lambda_coef * L_con

                self.optim.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optim.step()

                total["loss"]            += total_loss.item()
                total["policy_loss"]     += p_loss.item()
                total["value_loss"]      += v_loss.item()
                total["entropy"]         += entropy.mean().item()
                total["constraint_loss"] += L_con.item()
                n_up += 1

        self.buffer.reset()
        n_up = max(n_up, 1)
        return {f"env_{k}": v / n_up for k, v in total.items()}


# ============================================================================ #
#  PPO-Cog                                                                      #
# ============================================================================ #

class PPOCogAgent(_BasePPOAgent):

    COG_DIM = 3   # fixed: c_t always has one slot per appraisal variable

    def __init__(self, obs_dim: int, act_dim: int, **kwargs):
        super().__init__(input_dim=obs_dim + self.COG_DIM, act_dim=act_dim, **kwargs)
        self.obs_dim = obs_dim

    def _build_input(
        self,
        obs: np.ndarray | torch.Tensor,
        c_t: torch.Tensor,
    ) -> torch.Tensor:
        """Concatenate s_t and c_t into the cognitive input vector."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)
        if c_t.dim() == 1:
            c_t = c_t.unsqueeze(0)
        return torch.cat([obs_t, c_t], dim=-1).squeeze(0)

    @torch.no_grad()
    def select_action(
        self,
        obs: np.ndarray,
        c_t: torch.Tensor,
    ) -> tuple[int, float, float]:
        """
        Args:
            obs: s_t — current environmental observation
            c_t: cognitive-affective state from CognitiveAffectiveState

        Returns:
            action:   int
            log_prob: float
            value:    float
        """
        x = self._build_input(obs, c_t).to(self.device)
        action, log_prob, _, value = self.network.get_action_and_value(x)
        return action.item(), log_prob.item(), value.item()

    @torch.no_grad()
    def get_probs(self, obs: torch.Tensor, c_t: torch.Tensor) -> torch.Tensor:
        """p_cog: action probability vector for JS divergence"""
        x = torch.cat([obs, c_t], dim=-1).to(self.device)
        return self.network.get_probs(x)

    def store_cog(
        self,
        obs:      np.ndarray,
        c_t:      torch.Tensor,
        action:   int,
        log_prob: float,
        r_cog:    float,
        value:    float,
        done:     bool,
    ) -> None:
        """Store a cognitive transition using R_cog."""
        x = self._build_input(obs, c_t).cpu().numpy()
        self.buffer.store(x, action, log_prob, r_cog, value, done)

    def update_cog(self, last_obs: np.ndarray, last_c_t: torch.Tensor) -> dict:
        last_x = self._build_input(last_obs, last_c_t)
        return self.update(last_x, label="cog")


# ============================================================================ #
#  Sanity check                                                                 #
# ============================================================================ #

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from envs.decision_env import DecisionEnv

    torch.manual_seed(0)
    np.random.seed(0)

    OBS_DIM = 7
    ACT_DIM = 3
    COG_DIM = PPOCogAgent.COG_DIM   # 3 — one slot per appraisal variable

    env_agent = PPOEnvAgent(obs_dim=OBS_DIM, act_dim=ACT_DIM, rollout_steps=64)
    cog_agent = PPOCogAgent(obs_dim=OBS_DIM, act_dim=ACT_DIM, rollout_steps=64)

    print(f"PPO-Env params: {sum(p.numel() for p in env_agent.network.parameters()):,}")
    print(f"PPO-Cog params: {sum(p.numel() for p in cog_agent.network.parameters()):,}")
    print(f"PPO-Cog input:  obs_dim({OBS_DIM}) + cog_dim({COG_DIM}) = {OBS_DIM + COG_DIM}")

    env   = DecisionEnv(seed=0)
    obs, _ = env.reset()
    c_t   = torch.zeros(COG_DIM)   # [g_memory, n_memory, p_memory]

    # Collect a short rollout
    for step in range(70):
        # PPO-Env action
        action, logp_env, val_env = env_agent.select_action(obs)
        # PPO-Cog action (same obs + affective state)
        _, logp_cog, val_cog = cog_agent.select_action(obs, c_t)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        env_agent.store(obs, action, logp_env, reward,  val_env, done)
        cog_agent.store_cog(obs, c_t, action, logp_cog, reward * 0.5, val_cog, done)

        obs = next_obs if not done else env.reset()[0]

    # Update both
    env_info = env_agent.update_env(obs)
    cog_info = cog_agent.update_cog(obs, c_t)

    print("\nPPO-Env update:")
    for k, v in env_info.items():
        print(f"  {k:<25} {v:.4f}")

    print("\nPPO-Cog update:")
    for k, v in cog_info.items():
        print(f"  {k:<25} {v:.4f}")

    # Verify probability outputs (needed for Phase 7)
    obs_t = torch.as_tensor(obs, dtype=torch.float32)
    p_env = env_agent.get_probs(obs_t)
    p_cog = cog_agent.get_probs(obs_t, c_t)
    print(f"\np_env: {p_env.tolist()}")
    print(f"p_cog: {p_cog.tolist()}")
    print(f"Both sum to 1: {p_env.sum():.4f}, {p_cog.sum():.4f}")
