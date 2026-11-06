import torch


# ============================================================================ #
#  CognitiveAffectiveState                                                      #
# ============================================================================ #

class CognitiveAffectiveState:
    """
    c_{t+1} = (1 - α) · c_t  +  α · â_t

    where:
    â_t  = [ĝ_t, n̂_t, p̂_t]  — predicted appraisal vector (3-dim)
    α    — emotional sensitivity
    c_t  — cognitive-affective state (3-dim, same space as â_t)
    """

    # Fixed: c_t always lives in the same space as â_t
    APPRAISAL_DIM = 3

    def __init__(
        self,
        alpha:   float = 0.1,
        n_envs:  int   = 1,
        device:  str   = "cpu",
    ):
        """
        Args:
            alpha:   emotional sensitivity ∈ (0, 1]
                       0.05 → very slow emotional changes (long memory)
                       0.1  → recommended default
                       0.5  → high reactivity, fast forgetting
            n_envs:  number of parallel environments
            device:  torch device string
        """
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")

        self.alpha      = alpha
        self.n_envs     = n_envs
        self.device     = torch.device(device)

        # c_t shape: (n_envs, 3)
        self._state: torch.Tensor = torch.zeros(
            n_envs, self.APPRAISAL_DIM, device=self.device
        )

    # ------------------------------------------------------------------ #
    #  Core API                                                            #
    # ------------------------------------------------------------------ #

    def reset(self, env_idx: int | None = None) -> None:
        """
        Reset c_t to zero.

        Args:
            env_idx: reset only this environment; None resets all.
        """
        if env_idx is None:
            self._state = torch.zeros(
                self.n_envs, self.APPRAISAL_DIM, device=self.device
            )
        else:
            self._state[env_idx] = 0.0

    def update(
        self,
        a_hat: torch.Tensor,
        env_idx: int | None = None,
    ) -> torch.Tensor:
        """
        Args:
            a_hat:   (3,) tensor — [ĝ_t, n̂_t, p̂_t] predicted appraisal values
                     OR (n_envs, 3) for vectorised update
            env_idx: update only this environment; None updates all.

        Returns:
            c_{t+1}: updated state (3,) or (n_envs, 3)
        """
        a_hat = a_hat.to(self.device).detach()

        if env_idx is not None:
            if a_hat.dim() != 1 or a_hat.shape[0] != self.APPRAISAL_DIM:
                raise ValueError(
                    f"a_hat must be (3,) when env_idx is specified, got {a_hat.shape}"
                )
            self._state[env_idx] = (
                (1.0 - self.alpha) * self._state[env_idx]
                + self.alpha * a_hat
            )
            return self._state[env_idx].clone()
        else:
            if a_hat.dim() == 1:
                # Single env: broadcast across n_envs if needed
                a_hat = a_hat.unsqueeze(0).expand(self.n_envs, -1)
            self._state = (1.0 - self.alpha) * self._state + self.alpha * a_hat
            return self._state.clone()

    def get(self, env_idx: int | None = None) -> torch.Tensor:
        """
        Args:
            env_idx: return only this env's state (3,); None returns (n_envs, 3)
        """
        if env_idx is not None:
            return self._state[env_idx].clone()
        return self._state.clone()

    def get_flat(self) -> torch.Tensor:
        """
        Return c_t as a flat (3,) tensor — for single-env use as PPO-Cog input.
        """
        return self._state.squeeze(0).clone()

    # ------------------------------------------------------------------ #
    #  Interpretation helpers                                              #
    # ------------------------------------------------------------------ #

    def goal_congruence_memory(self, env_idx: int = 0) -> float:
        """Running average of ĝ_t — is the agent trending toward its goal?"""
        return float(self._state[env_idx, 0].item())

    def novelty_memory(self, env_idx: int = 0) -> float:
        """Running average of n̂_t — how novel has the environment been recently?"""
        return float(self._state[env_idx, 1].item())

    def coping_memory(self, env_idx: int = 0) -> float:
        """Running average of p̂_t — how in-control has the agent felt recently?"""
        return float(self._state[env_idx, 2].item())

    def summary(self, env_idx: int = 0) -> dict:
        """Diagnostic dictionary for logging — directly interpretable."""
        return {
            "c_goal_congruence": self.goal_congruence_memory(env_idx),
            "c_novelty":         self.novelty_memory(env_idx),
            "c_coping":          self.coping_memory(env_idx),
            "c_norm":            float(self._state[env_idx].norm().item()),
        }

    def __repr__(self) -> str:
        s = self._state[0]
        return (
            f"CognitiveAffectiveState("
            f"α={self.alpha}, "
            f"c=[g={s[0]:.3f}, n={s[1]:.3f}, p={s[2]:.3f}])"
        )


# ============================================================================ #
#  Alpha Scheduler (for sensitivity analysis)                                   #
# ============================================================================ #

class AlphaScheduler:
    """
    Linearly anneals α from alpha_start → alpha_end over total_steps.

    Early training:  high α (reactive — explore emotional landscape)
    Late training:   low α  (stable — settled regulated behaviour)
    """

    def __init__(
        self,
        cas:         CognitiveAffectiveState,
        alpha_start: float = 0.3,
        alpha_end:   float = 0.05,
        total_steps: int   = 100_000,
    ):
        self.cas         = cas
        self.alpha_start = alpha_start
        self.alpha_end   = alpha_end
        self.total_steps = total_steps
        self._step       = 0

    def step(self) -> float:
        progress        = min(self._step / self.total_steps, 1.0)
        new_alpha       = self.alpha_start + progress * (self.alpha_end - self.alpha_start)
        self.cas.alpha  = new_alpha
        self._step     += 1
        return new_alpha

    @property
    def current_alpha(self) -> float:
        return self.cas.alpha


# ============================================================================ #
#  Sanity check                                                                 #
# ============================================================================ #

if __name__ == "__main__":
    torch.manual_seed(0)

    print("=== Basic update with â_t = [ĝ, n̂, p̂] ===")
    cas = CognitiveAffectiveState(alpha=0.1)
    cas.reset()

    print(f"Initial: {cas}")

    # Simulate 10 steps: high goal congruence, low novelty, high coping
    for t in range(10):
        g_hat = torch.tensor( 0.8)
        n_hat = torch.tensor( 0.1)
        p_hat = torch.tensor( 0.9)
        a_hat = torch.stack([g_hat, n_hat, p_hat])
        cas.update(a_hat)

    print(f"After 10 positive steps: {cas}")
    print(f"Summary: {cas.summary()}")

    # Simulate 10 steps: negative — low congruence, high novelty, low coping
    for t in range(10):
        a_hat = torch.tensor([-0.8, 1.5, 0.1])
        cas.update(a_hat)

    print(f"\nAfter 10 stressful steps: {cas}")
    print(f"Summary: {cas.summary()}")

    print("\n=== Alpha sensitivity ===")
    for alpha in [0.05, 0.1, 0.3, 0.5]:
        cas2 = CognitiveAffectiveState(alpha=alpha)
        cas2.reset()
        spike = torch.tensor([1.0, 1.0, 1.0])
        for _ in range(10):
            cas2.update(spike)
        silence = torch.zeros(3)
        for _ in range(10):
            cas2.update(silence)
        print(f"  α={alpha:.2f}  residual after silence: {cas2.get_flat().tolist()}")

    print("\n=== Interpretability: each dimension is readable ===")
    cas3 = CognitiveAffectiveState(alpha=0.2)
    cas3.reset()
    for _ in range(20):
        a_hat = torch.tensor([0.6, 0.3, 0.7])
        cas3.update(a_hat)
    print(f"  goal_congruence_memory: {cas3.goal_congruence_memory():.4f}  (expect ≈0.6)")
    print(f"  novelty_memory:         {cas3.novelty_memory():.4f}          (expect ≈0.3)")
    print(f"  coping_memory:          {cas3.coping_memory():.4f}           (expect ≈0.7)")
