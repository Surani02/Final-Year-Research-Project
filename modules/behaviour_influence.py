import torch
import torch.nn.functional as F
from torch.distributions import Categorical


# ============================================================================ #
#  Influence Coefficient                                                        #
# ============================================================================ #

def compute_beta(
    D_t:      float | torch.Tensor,
    tau:      float = 0.2,
    k:        float = 10.0,
    beta_max: float = 0.5,
) -> float | torch.Tensor:
    """
    β_t = β_max · σ(k · (D_t − τ))

    Args:
        D_t:      JS divergence ∈ [0, 1]
        tau:      conflict threshold (default: 0.2)
        k:        sigmoid slope (default: 10)
        beta_max: maximum influence strength (default: 0.5)
    """
    import math
    if isinstance(D_t, torch.Tensor):
        return beta_max * torch.sigmoid(
            torch.tensor(k, dtype=D_t.dtype) * (D_t - tau)
        )
    else:
        return beta_max / (1.0 + math.exp(-k * (D_t - tau)))


# ============================================================================ #
#  Soft Policy Blending                                                         #
# ============================================================================ #

def blend_policies(
    p_env:  torch.Tensor,
    p_cog:  torch.Tensor,
    beta_t: float | torch.Tensor,
) -> torch.Tensor:
    """
    π_final = (1 − β_t) · π_env  +  β_t · π_cog

    Convex combination — result is always a valid probability distribution.
    """
    if isinstance(beta_t, torch.Tensor):
        if beta_t.dim() < p_env.dim():
            beta_t = beta_t.unsqueeze(-1)
    return (1.0 - beta_t) * p_env + beta_t * p_cog


def sample_from_blended(
    p_env:  torch.Tensor,
    p_cog:  torch.Tensor,
    beta_t: float,
) -> tuple[int, float]:
    """Sample action and log_prob from the blended distribution."""
    p_final = blend_policies(p_env, p_cog, beta_t)
    dist    = Categorical(probs=p_final)
    action  = dist.sample()
    return action.item(), dist.log_prob(action).item()


# ============================================================================ #
#  BehaviourInfluence — stateful wrapper                                        #
# ============================================================================ #

class BehaviourInfluence:
    """
    Manages the full influence pipeline: D_t → β_t → π_final

    Usage:
        influence = BehaviourInfluence(tau=0.2, k=10.0, beta_max=0.5)

        beta_t  = influence.compute_beta(D_t)
        p_final = influence.blend(p_env, p_cog, beta_t)
        action, logp = influence.sample(p_env, p_cog, beta_t)
    """

    def __init__(
        self,
        tau:         float = 0.2,
        k:           float = 10.0,
        beta_max:    float = 0.5,
        history_len: int   = 1000,
    ):
        self.tau         = tau
        self.k           = k
        self.beta_max    = beta_max
        self._history:   list[float] = []
        self.history_len = history_len

    def compute_beta(self, D_t: float) -> float:
        beta = compute_beta(D_t, self.tau, self.k, self.beta_max)
        self._history.append(float(beta))
        if len(self._history) > self.history_len:
            self._history.pop(0)
        return float(beta)

    def blend(self, p_env, p_cog, beta_t):
        return blend_policies(p_env, p_cog, beta_t)

    def sample(self, p_env, p_cog, beta_t):
        return sample_from_blended(p_env, p_cog, beta_t)

    def summary(self) -> dict:
        if not self._history:
            return {"beta_mean": 0.0, "beta_std": 0.0, "beta_last": 0.0, "beta_max_seen": 0.0}
        h    = self._history
        mean = sum(h) / len(h)
        std  = (sum((x - mean)**2 for x in h) / max(len(h)-1, 1)) ** 0.5
        return {
            "beta_mean":     mean,
            "beta_std":      std,
            "beta_last":     h[-1],
            "beta_max_seen": max(h),
        }

    def is_regulating(self, threshold: float = 0.1) -> bool:
        return bool(self._history) and self._history[-1] > threshold

    def __repr__(self) -> str:
        return f"BehaviourInfluence(τ={self.tau}, k={self.k}, β_max={self.beta_max})"


# ============================================================================ #
#  Sanity check                                                                 #
# ============================================================================ #

if __name__ == "__main__":
    import math
    torch.manual_seed(0)
    ACT_DIM = 3

    print("=== β_t vs D_t (tau=0.2, k=10, beta_max=0.5) ===")
    influence = BehaviourInfluence(tau=0.2, k=10.0, beta_max=0.5)
    for D in [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]:
        beta = influence.compute_beta(D)
        bar  = "█" * int(beta * 40)
        print(f"  D={D:.1f}  β={beta:.4f}  {bar}")
    print(f"\n  Summary: {influence.summary()}")

    print("\n=== Policy blending ===")
    p_env = torch.tensor([0.6, 0.3, 0.1])
    p_cog = torch.tensor([0.1, 0.2, 0.7])
    for beta_t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        p_f = blend_policies(p_env, p_cog, beta_t)
        print(f"  β={beta_t:.2f}  π_final={[round(x,3) for x in p_f.tolist()]}  sum={p_f.sum():.4f}")

    print("\n=== Edge: β=0 → pure PPO-Env ===")
    print(f"  match: {torch.allclose(blend_policies(p_env, p_cog, 0.0), p_env)}")
    print("=== Edge: β=1 → pure PPO-Cog ===")
    print(f"  match: {torch.allclose(blend_policies(p_env, p_cog, 1.0), p_cog)}")
