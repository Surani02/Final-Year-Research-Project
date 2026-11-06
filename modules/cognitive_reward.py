import torch
import torch.nn as nn


# ============================================================================ #
#  CognitiveReward                                                              #
# ============================================================================ #

class CognitiveReward:
    """
    Args:
        beta_r:  weight for environmental reward  (default 0.5)
        alpha_g: weight for goal congruence       (default 1.0)
        alpha_p: weight for coping potential      (default 1.0)
        clip:    if True, clip R_cog to [-2, 2] for stability
    """

    def __init__(
        self,
        beta_r:  float = 0.5,
        alpha_g: float = 1.0,
        alpha_p: float = 1.0,
        clip:    bool  = True,
    ):
        self.beta_r  = beta_r
        self.alpha_g = alpha_g
        self.alpha_p = alpha_p
        self.clip    = clip

    def compute(
        self,
        reward: torch.Tensor,
        g_hat:  torch.Tensor,
        p_hat:  torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            reward: (B,) environmental reward R_t
            g_hat:  (B,) goal congruence predictions ∈ [-1, 1]
            p_hat:  (B,) coping potential predictions ∈ [0, 1]

        Returns:
            r_cog: (B,) cognitive reward
        """
        r_cog = (
            self.beta_r  * reward
            + self.alpha_g * g_hat
            + self.alpha_p * p_hat
        )
        if self.clip:
            r_cog = torch.clamp(r_cog, -2.0, 2.0)
        return r_cog

    def compute_single(
        self,
        reward: float,
        g:      float,
        p:      float,
    ) -> float:
        """
        Args:
            reward: R_t from the environment
            g:      ĝ_t from the appraisal predictor
            p:      p̂_t from the appraisal predictor
        """
        r = self.beta_r * reward + self.alpha_g * g + self.alpha_p * p
        if self.clip:
            r = max(-2.0, min(2.0, r))
        return r

    def decompose(
        self,
        reward: torch.Tensor,
        g_hat:  torch.Tensor,
        p_hat:  torch.Tensor,
    ) -> dict:
        env_contrib  = self.beta_r  * reward
        g_contrib    = self.alpha_g * g_hat
        p_contrib    = self.alpha_p * p_hat
        r_cog        = self.compute(reward, g_hat, p_hat)

        return {
            "r_cog":        r_cog.mean().item(),
            "env_contrib":  env_contrib.mean().item(),
            "g_contrib":    g_contrib.mean().item(),
            "p_contrib":    p_contrib.mean().item(),
            "mean_reward":  reward.mean().item(),
            "mean_g":       g_hat.mean().item(),
            "mean_p":       p_hat.mean().item(),
        }

    def __repr__(self) -> str:
        return (
            f"CognitiveReward("
            f"β_r={self.beta_r}, α_g={self.alpha_g}, α_p={self.alpha_p}, "
            f"clip={self.clip})"
        )


# ============================================================================ #
#  Running-normalised variant                                                   #
# ============================================================================ #

class RunningNormCognitiveReward(CognitiveReward):
    """
    Extends CognitiveReward with online Welford running-mean normalisation.

    Prevents R_cog from dominating the environmental reward early in training
    when appraisal predictions are still noisy.
    """

    def __init__(self, epsilon: float = 1e-8, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self._count  = 0
        self._mean   = 0.0
        self._M2     = 0.0

    def _welford_update(self, x: float) -> None:
        self._count += 1
        delta        = x - self._mean
        self._mean  += delta / self._count
        self._M2    += delta * (x - self._mean)

    @property
    def std(self) -> float:
        if self._count < 2:
            return 1.0
        return max((self._M2 / (self._count - 1)) ** 0.5, self.epsilon)

    def compute(
        self,
        reward: torch.Tensor,
        g_hat:  torch.Tensor,
        p_hat:  torch.Tensor,
    ) -> torch.Tensor:
        r_raw = (
            self.beta_r  * reward
            + self.alpha_g * g_hat
            + self.alpha_p * p_hat
        )
        for val in r_raw.detach().cpu().numpy().flatten():
            self._welford_update(float(val))
        r_norm = (r_raw - self._mean) / self.std
        if self.clip:
            r_norm = torch.clamp(r_norm, -2.0, 2.0)
        return r_norm


# ============================================================================ #
#  Sanity check                                                                 #
# ============================================================================ #

if __name__ == "__main__":
    torch.manual_seed(0)

    cr = CognitiveReward(beta_r=0.5, alpha_g=1.0, alpha_p=1.0)
    print(cr)

    BATCH = 8
    reward = torch.randn(BATCH) * 0.5       # typical env rewards
    g      = torch.rand(BATCH) * 2 - 1     # ∈ [-1, 1]
    p      = torch.rand(BATCH)              # ∈ [0, 1]

    r_cog = cr.compute(reward, g, p)
    print(f"\nBatch cognitive rewards: {[round(x,3) for x in r_cog.tolist()]}")
    print(f"Mean: {r_cog.mean():.4f}  Std: {r_cog.std():.4f}")

    print("\nDecomposition:")
    for k, v in cr.decompose(reward, g, p).items():
        print(f"  {k:<15} {v:+.4f}")

    print("\n=== Edge cases ===")
    cases = [
        ("perfect:  R=1, g=1, p=1",   1.0,  1.0,  1.0),
        ("neutral:  R=0, g=0, p=0.5",  0.0,  0.0,  0.5),
        ("bad task: R=-1, g=-1, p=0", -1.0, -1.0,  0.0),
        ("mixed:    R=1, g=-0.5, p=0.3", 1.0, -0.5, 0.3),
    ]
    for label, r_val, g_val, p_val in cases:
        r_t = torch.tensor([r_val])
        g_t = torch.tensor([g_val])
        p_t = torch.tensor([p_val])
        out = cr.compute(r_t, g_t, p_t).item()
        scalar = cr.compute_single(r_val, g_val, p_val)
        print(f"  {label:<30} R_cog={out:+.4f}  (scalar={scalar:+.4f})")

    print("\n=== Running-normalised variant ===")
    cr_norm = RunningNormCognitiveReward(beta_r=0.5, alpha_g=1.0, alpha_p=1.0)
    for step in range(5):
        r_t = torch.randn(16) * 0.5
        g_t = torch.rand(16) * 2 - 1
        p_t = torch.rand(16)
        out = cr_norm.compute(r_t, g_t, p_t)
        print(f"  step {step}  r_cog mean={out.mean():+.4f}  running_std={cr_norm.std:.4f}")
