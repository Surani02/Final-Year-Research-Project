import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================ #
#  Mental Model Network                                                         #
# ============================================================================ #

class MentalModel(nn.Module):
    """
    Constructs m_t — the agent's internal interpretation of the environment.

    Inputs:  s_t (obs_dim) + r_t (scalar) → concatenated
    Output:  m_t (mental_dim)

    Architecture:
        Input → 128 → ReLU → 128 → ReLU → m_t
    """

    def __init__(self, obs_dim: int, mental_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + 1, 128),   # +1 for scalar reward
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, mental_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)

    def forward(self, obs: torch.Tensor, reward: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs:    (B, obs_dim)  — current environmental state
            reward: (B,) or (B,1) — last environmental reward
        Returns:
            m_t: (B, mental_dim)
        """
        if reward.dim() == 1:
            reward = reward.unsqueeze(-1)
        x = torch.cat([obs, reward], dim=-1)
        return self.net(x)


# ============================================================================ #
#  Appraisal Encoder                                                            #
# ============================================================================ #

class AppraisalEncoder(nn.Module):
    """
    Extracts latent cognitive features z_t from s_t and m_t.

    Inputs:  s_t (obs_dim) + m_t (mental_dim) → concatenated
    Output:  z_t (latent_dim)

    Architecture:
        Input → 256 → ReLU → 128 → ReLU → z_t
    """

    def __init__(self, obs_dim: int, mental_dim: int = 64, latent_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + mental_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)

    def forward(self, obs: torch.Tensor, mental: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs:    (B, obs_dim)
            mental: (B, mental_dim)
        Returns:
            z_t: (B, latent_dim)
        """
        x = torch.cat([obs, mental], dim=-1)
        return self.net(x)


# ============================================================================ #
#  Appraisal Predictor                                                          #
# ============================================================================ #

class AppraisalPredictor(nn.Module):
    """
    Predicts the three appraisal variables from z_t.

    Input:  z_t (latent_dim)
    Output: (ĝ_t, n̂_t, p̂_t) — goal congruence, novelty, coping potential

    Outputs are activated to constrain ranges:
        ĝ_t ∈ [-1, 1]  via tanh  (mirrors tanh pseudo-label)
        n̂_t ∈ [0, ∞)   via softplus (prediction error is non-negative)
        p̂_t ∈ [0, 1]   via sigmoid (probability-like coping)

    Architecture:
        z_t → 128 → ReLU → 64 → ReLU → 3 outputs
    """

    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        # Separate heads for each appraisal variable
        self.head_goal_congruence  = nn.Linear(64, 1)   # ĝ
        self.head_novelty          = nn.Linear(64, 1)   # n̂
        self.head_coping_potential = nn.Linear(64, 1)   # p̂

        self._init_weights()

    def _init_weights(self):
        for layer in self.trunk:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)
        for head in [self.head_goal_congruence, self.head_novelty, self.head_coping_potential]:
            nn.init.orthogonal_(head.weight, gain=0.01)
            nn.init.zeros_(head.bias)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z: (B, latent_dim)
        Returns:
            g_hat: (B,)  goal congruence  ∈ [-1, 1]
            n_hat: (B,)  novelty          ∈ [0, ∞)
            p_hat: (B,)  coping potential ∈ [0, 1]
        """
        h = self.trunk(z)
        g_hat = torch.tanh(self.head_goal_congruence(h)).squeeze(-1)
        n_hat = F.softplus(self.head_novelty(h)).squeeze(-1)
        p_hat = torch.sigmoid(self.head_coping_potential(h)).squeeze(-1)
        return g_hat, n_hat, p_hat


# ============================================================================ #
#  Next-State Predictor (for novelty pseudo-label)                             #
# ============================================================================ #

class NextStatePredictor(nn.Module):
    """
    Predicts ŝ_{t+1} from s_t and action a_t.

    Used to compute novelty pseudo-label:
        n_t = ||s_{t+1} - ŝ_{t+1}||

    Large prediction error → high novelty (Scherer's novelty appraisal).
    """

    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, obs_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)

    def forward(
        self, obs: torch.Tensor, action_onehot: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            obs:           (B, obs_dim)
            action_onehot: (B, act_dim)
        Returns:
            next_obs_pred: (B, obs_dim)
        """
        x = torch.cat([obs, action_onehot], dim=-1)
        return self.net(x)


# ============================================================================ #
#  Pseudo-Label Generation                                                      #
# ============================================================================ #

def generate_pseudo_labels(
    reward: torch.Tensor,
    obs: torch.Tensor,
    next_obs: torch.Tensor,
    action: torch.Tensor,
    value: torch.Tensor,
    next_state_predictor: NextStatePredictor,
    act_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generates (g, n, p) pseudo-labels without any labelled affective dataset.

    Goal Congruence:
        g_t = tanh(r_t)

    Novelty:
        n_t = ||s_{t+1} - ŝ_{t+1}||₂

    Coping Potential:
        p_t = σ(V(s_t))

    Args:
        reward:               (B,)
        obs:                  (B, obs_dim)
        next_obs:             (B, obs_dim)
        action:               (B,) long — discrete action indices
        value:                (B,) — critic value estimates
        next_state_predictor: NextStatePredictor
        act_dim:              int

    Returns:
        g_pseudo: (B,)  goal congruence  ∈ [-1, 1]
        n_pseudo: (B,)  novelty          ∈ [0, ∞)
        p_pseudo: (B,)  coping potential ∈ [0, 1]
    """
    # Goal congruence: tanh-normalised reward
    g_pseudo = torch.tanh(reward)

    # Novelty: prediction error of next state
    action_onehot = F.one_hot(action, num_classes=act_dim).float()
    with torch.no_grad():
        next_obs_pred = next_state_predictor(obs, action_onehot)
    n_pseudo = torch.norm(next_obs - next_obs_pred, dim=-1)

    # Coping potential: sigmoid of critic value
    p_pseudo = torch.sigmoid(value)

    return g_pseudo, n_pseudo, p_pseudo


# ============================================================================ #
#  Appraisal Loss                                                               #
# ============================================================================ #

def appraisal_loss(
    g_hat: torch.Tensor,
    n_hat: torch.Tensor,
    p_hat: torch.Tensor,
    g_pseudo: torch.Tensor,
    n_pseudo: torch.Tensor,
    p_pseudo: torch.Tensor,
) -> torch.Tensor:
    """
    MSE regression loss over all three appraisal variables.

    L_appraisal = (g - ĝ)² + (n - n̂)² + (p - p̂)²
    """
    return (
        F.mse_loss(g_hat, g_pseudo)
        + F.mse_loss(n_hat, n_pseudo)
        + F.mse_loss(p_hat, p_pseudo)
    )


# ============================================================================ #
#  Appraisal Module (combined interface)                                        #
# ============================================================================ #

class AppraisalModule(nn.Module):
    """
    Owned networks:
        MentalModel          → m_t
        AppraisalEncoder     → z_t
        AppraisalPredictor   → (ĝ, n̂, p̂)
        NextStatePredictor   → ŝ_{t+1}  (for novelty pseudo-label)

    Usage:
        module = AppraisalModule(obs_dim=7, act_dim=3)
        optim  = Adam(module.parameters(), lr=3e-4)

        # During training step:
        g_hat, n_hat, p_hat, z_t = module(obs, reward, action, next_obs, value)
        loss = module.loss(g_hat, n_hat, p_hat, obs, reward, action, next_obs, value)
        loss.backward()
        optim.step()
    """

    def __init__(
        self,
        obs_dim:    int,
        act_dim:    int,
        mental_dim: int = 64,
        latent_dim: int = 64,
    ):
        super().__init__()
        self.obs_dim    = obs_dim
        self.act_dim    = act_dim
        self.mental_dim = mental_dim
        self.latent_dim = latent_dim

        self.mental_model         = MentalModel(obs_dim, mental_dim)
        self.encoder              = AppraisalEncoder(obs_dim, mental_dim, latent_dim)
        self.predictor            = AppraisalPredictor(latent_dim)
        self.next_state_predictor = NextStatePredictor(obs_dim, act_dim)

    def forward(
        self,
        obs:      torch.Tensor,
        reward:   torch.Tensor,
        action:   torch.Tensor,
        next_obs: torch.Tensor,
        value:    torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            g_hat:  (B,)  predicted goal congruence
            n_hat:  (B,)  predicted novelty
            p_hat:  (B,)  predicted coping potential
            z_t:    (B, latent_dim)  latent cognitive features
        """
        m_t   = self.mental_model(obs, reward)
        z_t   = self.encoder(obs, m_t)
        g_hat, n_hat, p_hat = self.predictor(z_t)
        return g_hat, n_hat, p_hat, z_t

    def compute_loss(
        self,
        obs:      torch.Tensor,
        reward:   torch.Tensor,
        action:   torch.Tensor,
        next_obs: torch.Tensor,
        value:    torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Returns:
            total_loss: scalar
            info: dict with individual loss components and appraisal values
        """
        # Forward
        g_hat, n_hat, p_hat, z_t = self.forward(obs, reward, action, next_obs, value)

        # Pseudo-labels
        g_pseudo, n_pseudo, p_pseudo = generate_pseudo_labels(
            reward, obs, next_obs, action, value,
            self.next_state_predictor, self.act_dim,
        )

        # Appraisal regression loss
        loss_appraisal = appraisal_loss(g_hat, n_hat, p_hat, g_pseudo, n_pseudo, p_pseudo)

        # Next-state predictor loss (auxiliary — improves novelty signal quality)
        action_onehot = F.one_hot(action, num_classes=self.act_dim).float()
        next_obs_pred = self.next_state_predictor(obs, action_onehot)
        loss_dynamics = F.mse_loss(next_obs_pred, next_obs)

        total_loss = loss_appraisal + 0.5 * loss_dynamics

        info = {
            "loss_appraisal":   loss_appraisal.item(),
            "loss_dynamics":    loss_dynamics.item(),
            "loss_total":       total_loss.item(),
            "mean_g":           g_hat.mean().item(),
            "mean_n":           n_hat.mean().item(),
            "mean_p":           p_hat.mean().item(),
            "mean_g_pseudo":    g_pseudo.mean().item(),
            "mean_n_pseudo":    n_pseudo.mean().item(),
            "mean_p_pseudo":    p_pseudo.mean().item(),
        }

        return total_loss, info

    @torch.no_grad()
    def infer(
        self,
        obs:    torch.Tensor,
        reward: float | torch.Tensor,
        action: int | torch.Tensor,
        value:  float | torch.Tensor,
    ) -> dict:
        """
        Single-step inference (no gradients) — used during rollout collection.

        Returns dict with g, n, p predictions and z_t.
        """
        if not isinstance(reward, torch.Tensor):
            reward = torch.tensor([reward], dtype=torch.float32)
        if not isinstance(action, torch.Tensor):
            action = torch.tensor([action], dtype=torch.long)
        if not isinstance(value, torch.Tensor):
            value = torch.tensor([value], dtype=torch.float32)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        m_t = self.mental_model(obs, reward.unsqueeze(-1) if reward.dim() == 0 else reward.unsqueeze(-1) if reward.dim() == 1 else reward)
        z_t = self.encoder(obs, m_t)
        g_hat, n_hat, p_hat = self.predictor(z_t)

        return {
            "g":   g_hat.squeeze().item(),
            "n":   n_hat.squeeze().item(),
            "p":   p_hat.squeeze().item(),
            "z_t": z_t.squeeze(),
        }


# ============================================================================ #
#  Sanity check                                                                 #
# ============================================================================ #

if __name__ == "__main__":
    torch.manual_seed(0)

    OBS_DIM = 7
    ACT_DIM = 3
    BATCH   = 16

    module = AppraisalModule(obs_dim=OBS_DIM, act_dim=ACT_DIM)
    optim  = torch.optim.Adam(module.parameters(), lr=3e-4)

    print("AppraisalModule parameter count:",
          sum(p.numel() for p in module.parameters()))

    # Simulate a training mini-batch
    obs      = torch.rand(BATCH, OBS_DIM)
    next_obs = torch.rand(BATCH, OBS_DIM)
    reward   = torch.randn(BATCH)
    action   = torch.randint(0, ACT_DIM, (BATCH,))
    value    = torch.randn(BATCH)

    loss, info = module.compute_loss(obs, reward, action, next_obs, value)
    loss.backward()
    optim.step()

    print("\nTraining step output:")
    for k, v in info.items():
        print(f"  {k:<22} {v:.4f}")

    # Single-step inference
    single_obs = torch.rand(OBS_DIM)
    result = module.infer(single_obs, reward=0.5, action=1, value=0.8)
    print("\nSingle-step inference:")
    for k, v in result.items():
        if k != "z_t":
            print(f"  {k}: {v:.4f}")
    print(f"  z_t shape: {result['z_t'].shape}")
