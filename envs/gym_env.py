"""
Supported environments (discrete action space):
    CartPole-v1      obs_dim=4   act_dim=2
    Acrobot-v1       obs_dim=6   act_dim=3
    MountainCar-v0   obs_dim=2   act_dim=3
    LunarLander-v3   obs_dim=8   act_dim=4

Usage:
    env = GymEnvWrapper("CartPole-v1", seed=42)
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)
"""

import numpy as np
import gymnasium as gym


# ============================================================================ #
#  Supported environments                                                       #
# ============================================================================ #

SUPPORTED_ENVS = {
    "CartPole-v1": {
        "obs_dim": 4,
        "act_dim": 2,
        "obs_labels": [
            "cart_position",
            "cart_velocity",
            "pole_angle",
            "pole_angular_velocity",
        ],
        "max_steps": 500,
        "description": "Balance a pole on a cart. Goal = keep pole upright.",
    },
    "Acrobot-v1": {
        "obs_dim": 6,
        "act_dim": 3,
        "obs_labels": [
            "cos_theta1",
            "sin_theta1",
            "cos_theta2",
            "sin_theta2",
            "angular_velocity1",
            "angular_velocity2",
        ],
        "max_steps": 500,
        "description": "Swing a two-link pendulum up. Goal = lift end-effector above threshold.",
    },
    "LunarLander-v3": {
        "obs_dim": 8,
        "act_dim": 4,
        "obs_labels": [
            "x_position",
            "y_position",
            "x_velocity",
            "y_velocity",
            "angle",
            "angular_velocity",
            "left_leg_contact",
            "right_leg_contact",
        ],
        "max_steps": 1000,
        "description": "Land a spacecraft between two flags. Goal = safe landing.",
    },
    "MountainCar-v0": {
        "obs_dim": 2,
        "act_dim": 3,
        "obs_labels": [
            "position",
            "velocity",
        ],
        "max_steps": 200,
        "description": "Drive a car up a hill using momentum. Goal = reach the flag.",
    },
}


# ============================================================================ #
#  Wrapper                                                                      #
# ============================================================================ #

class GymEnvWrapper:
    def __init__(
        self,
        env_name: str = "CartPole-v1",
        seed:     int = 42,
        render:   bool = False,
    ):
        if env_name not in SUPPORTED_ENVS:
            raise ValueError(
                f"Unsupported environment: '{env_name}'. "
                f"Choose from: {list(SUPPORTED_ENVS.keys())}"
            )

        self.env_name = env_name
        self.seed     = seed
        self._meta    = SUPPORTED_ENVS[env_name]

        render_mode = "human" if render else None
        self._env   = gym.make(env_name, render_mode=render_mode)

        # Validate discrete action space
        if not isinstance(self._env.action_space, gym.spaces.Discrete):
            raise ValueError(
                f"{env_name} does not have a discrete action space. "
                f"AC-PPO currently supports discrete actions only."
            )

        # Public attributes — used by agents and appraisal module
        self.obs_dim         = self._env.observation_space.shape[0]
        self.act_dim         = int(self._env.action_space.n)
        self.observation_space = self._env.observation_space
        self.action_space      = self._env.action_space

    # ------------------------------------------------------------------ #
    #  Gymnasium-compatible API                                            #
    # ------------------------------------------------------------------ #

    def reset(
        self,
        seed: int | None = None,
    ) -> tuple[np.ndarray, dict]:
        obs, info = self._env.reset(seed=seed if seed is not None else self.seed)
        return obs.astype(np.float32), info

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self._env.step(action)
        return obs.astype(np.float32), float(reward), terminated, truncated, info

    def close(self) -> None:
        self._env.close()

    # ------------------------------------------------------------------ #
    #  Info                                                                #
    # ------------------------------------------------------------------ #

    def describe(self) -> None:
        """Print environment summary."""
        meta = self._meta
        print(f"\n{'='*55}")
        print(f"  Environment : {self.env_name}")
        print(f"  Description : {meta['description']}")
        print(f"  obs_dim     : {self.obs_dim}")
        print(f"  act_dim     : {self.act_dim}")
        print(f"  Observation dimensions:")
        for i, label in enumerate(meta["obs_labels"]):
            print(f"    [{i}] {label}")
        print(f"{'='*55}\n")

    def __repr__(self) -> str:
        return f"GymEnvWrapper({self.env_name}, obs_dim={self.obs_dim}, act_dim={self.act_dim})"


# ============================================================================ #
#  Factory function                                                             #
# ============================================================================ #

def make_env(env_name: str, seed: int = 42, render: bool = False) -> GymEnvWrapper:
    return GymEnvWrapper(env_name=env_name, seed=seed, render=render)


# ============================================================================ #
#  Sanity check                                                                 #
# ============================================================================ #

if __name__ == "__main__":
    for env_name in SUPPORTED_ENVS:
        print(f"\n--- {env_name} ---")
        env = make_env(env_name, seed=42)
        env.describe()

        obs, info = env.reset()
        print(f"  Initial obs : {obs.round(4).tolist()}")

        ep_reward = 0.0
        for step in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            if terminated or truncated:
                print(f"  Done at step {step+1} | reward so far: {ep_reward:.3f}")
                break
        else:
            print(f"  10 steps | reward so far: {ep_reward:.3f}")

        env.close()
