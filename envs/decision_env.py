import numpy as np
import gymnasium as gym
from gymnasium import spaces


class DecisionEnv(gym.Env):
    """
    Observation space (7-dim continuous):
        [0] position          – agent's location in [0, 1]
        [1] urgency           – time-pressure signal in [0, 1]; rises each step
        [2] risk_level        – danger at current position in [0, 1]
        [3] uncertainty       – environment stochasticity signal in [0, 1]
        [4] coping_potential  – agent's perceived controllability in [0, 1]
        [5] progress          – fraction of distance covered toward goal
        [6] steps_remaining   – normalised steps left in episode

    Action space (discrete, 3 actions):
        0 – move cautiously   (small step, low risk exposure)
        1 – move normally     (medium step, moderate risk)
        2 – move aggressively (large step, higher risk)

    Reward structure:
        +progress_gain * (1 - risk_level)   on each step
        +1.0                                on reaching the goal
        -risk_level                         penalty proportional to local risk
        -0.01                               small step penalty (efficiency)
        -1.0                                on critical failure (risk > threshold)

    Episode termination:
        - Goal reached  (position >= 1.0)
        - Critical risk event (risk_level > risk_threshold AND bad luck)
        - Max steps exceeded
    """

    metadata = {"render_modes": ["human"]}

    # ------------------------------------------------------------------ #
    #  Configurable constants                                              #
    # ------------------------------------------------------------------ #
    STEP_SIZES       = [0.03, 0.06, 0.10]   # per action
    RISK_THRESHOLDS  = [0.3,  0.5,  0.7]    # risk exposure per action
    FAILURE_PROB_K   = 2.0                   # steepness of failure probability
    MAX_STEPS        = 200

    def __init__(
        self,
        max_steps: int = MAX_STEPS,
        risk_threshold: float = 0.75,
        seed: int | None = None,
    ):
        super().__init__()
        self.max_steps      = max_steps
        self.risk_threshold = risk_threshold

        # Observation and action spaces
        low  = np.zeros(7, dtype=np.float32)
        high = np.ones(7,  dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.action_space      = spaces.Discrete(3)

        # Internal RNG (gymnasium-compatible)
        self.np_random, _ = gym.utils.seeding.np_random(seed)

        # State variables (initialised in reset)
        self._position:         float = 0.0
        self._urgency:          float = 0.0
        self._risk_level:       float = 0.0
        self._uncertainty:      float = 0.0
        self._coping_potential: float = 1.0
        self._step_count:       int   = 0

        # Risk profile: a fixed sequence of "danger zones" per episode
        self._risk_profile: np.ndarray = np.array([])

    # ------------------------------------------------------------------ #
    #  Gymnasium API                                                       #
    # ------------------------------------------------------------------ #

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        self._position         = 0.0
        self._step_count       = 0
        self._urgency          = 0.0
        self._coping_potential = 1.0

        # Generate a random risk landscape for this episode
        # Values represent risk intensity at 10 waypoints along [0, 1]
        self._risk_profile = self.np_random.uniform(0.0, 1.0, size=10).astype(np.float32)

        # Initial uncertainty is high (novel episode)
        self._uncertainty  = float(self.np_random.uniform(0.5, 1.0))
        self._risk_level   = self._get_risk_at(self._position)

        obs  = self._build_obs()
        info = self._build_info()
        return obs, info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self.action_space.contains(action), f"Invalid action: {action}"

        self._step_count += 1
        prev_position     = self._position

        # ---- 1. Move -------------------------------------------------- #
        step_size = self.STEP_SIZES[action]
        # Add small stochastic noise scaled by uncertainty
        noise = float(self.np_random.normal(0.0, 0.02 * self._uncertainty))
        self._position = float(np.clip(self._position + step_size + noise, 0.0, 1.0))

        # ---- 2. Update environment signals ---------------------------- #
        self._risk_level   = self._get_risk_at(self._position)
        self._uncertainty  = self._update_uncertainty(action)
        self._urgency      = min(1.0, self._step_count / self.max_steps)
        self._coping_potential = self._update_coping()

        # ---- 3. Check failure event ----------------------------------- #
        failure = self._check_failure(action)

        # ---- 4. Compute reward ---------------------------------------- #
        progress_gain = self._position - prev_position
        reward = (
            progress_gain * (1.0 - self._risk_level)   # progress reward
            - self._risk_level * 0.1                   # risk penalty
            - 0.005                                    # step cost
        )

        if failure:
            reward -= 1.0

        goal_reached = self._position >= 1.0
        if goal_reached:
            reward += 1.0

        # ---- 5. Termination ------------------------------------------- #
        terminated = goal_reached or failure
        truncated  = (self._step_count >= self.max_steps) and not terminated

        obs  = self._build_obs()
        info = self._build_info(
            goal_reached=goal_reached,
            failure=failure,
            progress_gain=progress_gain,
        )

        return obs, float(reward), terminated, truncated, info

    def render(self, mode: str = "human") -> None:
        bar_len   = 40
        filled    = int(self._position * bar_len)
        bar       = "#" * filled + "-" * (bar_len - filled)
        print(
            f"[{bar}] pos={self._position:.3f} "
            f"risk={self._risk_level:.2f} urgency={self._urgency:.2f} "
            f"cope={self._coping_potential:.2f} step={self._step_count}"
        )

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _get_risk_at(self, position: float) -> float:
        """Interpolate risk from the episode's risk profile."""
        idx   = position * (len(self._risk_profile) - 1)
        lo    = int(idx)
        hi    = min(lo + 1, len(self._risk_profile) - 1)
        frac  = idx - lo
        return float((1 - frac) * self._risk_profile[lo] + frac * self._risk_profile[hi])

    def _update_uncertainty(self, action: int) -> float:
        """
        Uncertainty decreases as the agent takes repeated actions in similar
        regions (familiarity effect), but aggressive actions increase it.
        """
        action_noise = [0.0, 0.02, 0.05][action]
        new_unc = self._uncertainty * 0.98 + action_noise
        return float(np.clip(new_unc, 0.0, 1.0))

    def _update_coping(self) -> float:
        """
        Coping potential rises when risk is low, falls when risk is high.
        """
        delta = -0.05 * self._risk_level + 0.02 * (1.0 - self._risk_level)
        new_cope = self._coping_potential + delta
        return float(np.clip(new_cope, 0.0, 1.0))

    def _check_failure(self, action: int) -> bool:
        """
        Probabilistic failure: aggressive actions in high-risk zones can fail.
        P(failure) = sigmoid( k * (risk - threshold) ) * action_modifier
        """
        action_modifier = [0.5, 1.0, 2.0][action]
        p_fail = self._sigmoid(
            self.FAILURE_PROB_K * (self._risk_level - self.risk_threshold)
        ) * action_modifier * 0.1
        return bool(self.np_random.random() < p_fail)

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + np.exp(-x))

    def _build_obs(self) -> np.ndarray:
        progress = self._position      # already in [0,1]
        steps_remaining = 1.0 - (self._step_count / self.max_steps)
        return np.array(
            [
                self._position,
                self._urgency,
                self._risk_level,
                self._uncertainty,
                self._coping_potential,
                progress,
                steps_remaining,
            ],
            dtype=np.float32,
        )

    def _build_info(
        self,
        goal_reached: bool = False,
        failure: bool = False,
        progress_gain: float = 0.0,
    ) -> dict:
        return {
            "position":         self._position,
            "urgency":          self._urgency,
            "risk_level":       self._risk_level,
            "uncertainty":      self._uncertainty,
            "coping_potential": self._coping_potential,
            "goal_reached":     goal_reached,
            "failure":          failure,
            "progress_gain":    progress_gain,
            "step":             self._step_count,
        }


# --------------------------------------------------------------------------- #
#  Quick sanity check                                                          #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    env = DecisionEnv(seed=42)
    obs, info = env.reset()
    print("Observation space:", env.observation_space)
    print("Action space:     ", env.action_space)
    print("Initial obs:      ", obs)

    total_reward = 0.0
    for _ in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        env.render()
        if terminated or truncated:
            print(f"Episode ended — total reward: {total_reward:.3f}")
            break
