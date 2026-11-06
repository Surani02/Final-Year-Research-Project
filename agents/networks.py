import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    """
    Shared-trunk Actor-Critic used by both PPO agents.

    Architecture:
        input → Linear(hidden) → ReLU → Linear(hidden) → ReLU
                     ↓ actor_head (logits)
                     ↓ critic_head (value)
    """

    def __init__(self, input_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.actor_head  = nn.Linear(hidden, act_dim)
        self.critic_head = nn.Linear(hidden, 1)
        self._init_weights()

    def _init_weights(self):
        for layer in self.trunk:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor_head.weight,  gain=0.01)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
        nn.init.zeros_(self.actor_head.bias)
        nn.init.zeros_(self.critic_head.bias)

    def forward(self, x: torch.Tensor):
        """Returns (logits, value)."""
        h     = self.trunk(x)
        return self.actor_head(h), self.critic_head(h).squeeze(-1)

    def get_action_and_value(self, x: torch.Tensor, action=None):
        logits, value = self.forward(x)
        dist          = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    def get_probs(self, x: torch.Tensor) -> torch.Tensor:
        """Return action probability distribution (used for JS divergence)."""
        logits, _ = self.forward(x)
        return torch.softmax(logits, dim=-1)
