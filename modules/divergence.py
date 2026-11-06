import torch
import torch.nn.functional as F


# ============================================================================ #
#  Core JS Divergence                                                           #
# ============================================================================ #

def js_divergence(
    p: torch.Tensor,
    q: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    JS(P‖Q) = ½ KL(P‖M) + ½ KL(Q‖M)   where M = ½(P + Q)

    Args:
        p:   (B, act_dim) or (act_dim,) — distribution 1 (e.g. π_env)
        q:   (B, act_dim) or (act_dim,) — distribution 2 (e.g. π_cog)
        eps: small value for numerical safety

    Returns:
        js:  (B,) or scalar — JS divergence in [0, ln2] ≈ [0, 0.693]
             Divide by ln(2) if [0, 1] range is preferred.
    """
    # Ensure valid probability distributions
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)
    p = p / p.sum(dim=-1, keepdim=True)
    q = q / q.sum(dim=-1, keepdim=True)

    m = 0.5 * (p + q)

    # KL(P‖M) = Σ p * log(p/m)
    # F.kl_div expects log-input: kl_div(log_q, p) = Σ p*(log_p - log_q)
    kl_pm = F.kl_div(m.log(), p, reduction="none").sum(dim=-1)
    kl_qm = F.kl_div(m.log(), q, reduction="none").sum(dim=-1)

    return 0.5 * (kl_pm + kl_qm)


def js_divergence_normalised(
    p: torch.Tensor,
    q: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    JS divergence normalised to [0, 1] by dividing by ln(2).
    """
    return js_divergence(p, q, eps) / torch.log(torch.tensor(2.0))


# ============================================================================ #
#  PolicyDivergence — stateful wrapper with history tracking                   #
# ============================================================================ #

class PolicyDivergence:

    def __init__(self, normalise: bool = True, history_len: int = 1000):
        """
        Args:
            normalise:   if True, D_t ∈ [0, 1]; else D_t ∈ [0, ln2]
            history_len: rolling window length for summary stats
        """
        self.normalise   = normalise
        self.history_len = history_len
        self._history:   list[float] = []

    def compute(
        self,
        p_env: torch.Tensor,
        p_cog: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            p_env: (..., act_dim) probability vector from PPO-Env
            p_cog: (..., act_dim) probability vector from PPO-Cog

        Returns:
            D_t: scalar or (B,) tensor
        """
        if self.normalise:
            return js_divergence_normalised(p_env, p_cog)
        return js_divergence(p_env, p_cog)

    def compute_scalar(
        self,
        p_env: torch.Tensor,
        p_cog: torch.Tensor,
    ) -> float:
        with torch.no_grad():
            return float(self.compute(p_env, p_cog).mean().item())

    def record(self, d_t: float | torch.Tensor) -> None:
        """Store a divergence value in the rolling history."""
        val = float(d_t) if not isinstance(d_t, torch.Tensor) else float(d_t.mean().item())
        self._history.append(val)
        if len(self._history) > self.history_len:
            self._history.pop(0)

    def summary(self) -> dict:
        """Diagnostic statistics for logging and plotting."""
        if not self._history:
            return {"d_mean": 0.0, "d_std": 0.0, "d_min": 0.0, "d_max": 0.0, "d_last": 0.0}
        h = self._history
        return {
            "d_mean": float(sum(h) / len(h)),
            "d_std":  float((sum((x - sum(h)/len(h))**2 for x in h) / max(len(h)-1, 1)) ** 0.5),
            "d_min":  float(min(h)),
            "d_max":  float(max(h)),
            "d_last": float(h[-1]),
        }

    def is_converging(self, window: int = 100, threshold: float = 0.05) -> bool:
        if len(self._history) < window:
            return False
        recent = self._history[-window:]
        return (sum(recent) / window) < threshold


# ============================================================================ #
#  Sanity check                                                                 #
# ============================================================================ #

if __name__ == "__main__":
    torch.manual_seed(0)

    ACT_DIM = 3

    print("=== Edge case: identical distributions ===")
    p = torch.tensor([0.2, 0.5, 0.3])
    q = torch.tensor([0.2, 0.5, 0.3])
    d = js_divergence_normalised(p, q)
    print(f"  JS(P‖P) = {d.item():.6f}  (should be 0.0)")

    print("\n=== Edge case: maximally different distributions ===")
    p_max = torch.tensor([1.0, 0.0, 0.0])
    q_max = torch.tensor([0.0, 0.0, 1.0])
    d_max = js_divergence_normalised(p_max, q_max)
    print(f"  JS(P‖Q) = {d_max.item():.6f}  (should be 1.0)")

    print("\n=== Batch computation ===")
    BATCH = 8
    p_batch = torch.softmax(torch.randn(BATCH, ACT_DIM), dim=-1)
    q_batch = torch.softmax(torch.randn(BATCH, ACT_DIM), dim=-1)
    d_batch = js_divergence_normalised(p_batch, q_batch)
    print(f"  Batch D_t: {d_batch.tolist()}")
    print(f"  All in [0,1]: {bool((d_batch >= 0).all() and (d_batch <= 1).all())}")

    print("\n=== PolicyDivergence tracker ===")
    tracker = PolicyDivergence(normalise=True)

    # Simulate training: divergence starts high, reduces over time
    import math
    for step in range(200):
        # Simulated convergence: policies start far apart, drift together
        noise = torch.randn(ACT_DIM) * max(0.1, 1.0 - step / 150)
        p_sim = torch.softmax(torch.zeros(ACT_DIM), dim=-1)
        q_sim = torch.softmax(noise, dim=-1)
        D_t   = tracker.compute_scalar(p_sim, q_sim)
        tracker.record(D_t)

    print(f"  Summary: {tracker.summary()}")
    print(f"  Converging (window=100, τ=0.2): {tracker.is_converging(window=100, threshold=0.2)}")

    print("\n=== Symmetry check ===")
    p_s = torch.softmax(torch.randn(ACT_DIM), dim=-1)
    q_s = torch.softmax(torch.randn(ACT_DIM), dim=-1)
    d_pq = js_divergence(p_s, q_s).item()
    d_qp = js_divergence(q_s, p_s).item()
    print(f"  JS(P‖Q) = {d_pq:.6f}")
    print(f"  JS(Q‖P) = {d_qp:.6f}")
    print(f"  Symmetric: {abs(d_pq - d_qp) < 1e-5}")
