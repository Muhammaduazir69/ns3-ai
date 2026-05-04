"""Uplink open-loop power control over an LEO link — continuous Box action.

State: path loss (dB), slow-fading state (dB), recent BLER. Action: TX power
in [-40, +23] dBm (UE class). Reward: throughput proxy minus power cost.

Reference physics: NTN UE budget ≈ 23 dBm; LEO PL ≈ 162 dB at zenith
500 km. We bound the agent to a saner [-40, +23] range so untrained
policies cannot saturate.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class PowerCtrlEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        episode_steps: int = 200,
        target_snr_db: float = 5.0,
        seed: int | None = None,
    ):
        super().__init__()
        self.episode_steps = int(episode_steps)
        self.target_snr_db = float(target_snr_db)

        # Obs: pl_db, slow_fade_db, last_bler, last_snr_db
        self.observation_space = spaces.Box(
            low=np.array([140.0, -10.0, 0.0, -20.0], dtype=np.float32),
            high=np.array([180.0, 10.0, 1.0, 30.0], dtype=np.float32),
            dtype=np.float32,
        )
        # Action: TX power (dBm) — UE class budget [-40, +23]
        self.action_space = spaces.Box(low=-40.0, high=23.0, shape=(1,),
                                       dtype=np.float32)

        self._rng = np.random.default_rng(seed)
        self._pl = 162.0
        self._slow = 0.0
        self._last_bler = 0.0
        self._last_snr = -10.0
        self._step = 0

    def _bler_from_snr(self, snr_db: float) -> float:
        # smooth BLER curve: 0.5 at 0 dB, →0 above 8 dB, →1 below -5 dB.
        return float(1.0 / (1.0 + np.exp((snr_db - 1.0) * 0.7)))

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._pl = float(self._rng.uniform(155.0, 172.0))
        self._slow = float(self._rng.normal(0.0, 3.0))
        self._last_bler = 0.0
        self._last_snr = -10.0
        self._step = 0
        obs = np.array([self._pl, self._slow, self._last_bler, self._last_snr],
                       dtype=np.float32)
        return obs, {}

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape != (1,):
            raise ValueError(f"action must be shape (1,), got {action.shape}")
        tx_dbm = float(np.clip(action[0], -40.0, 23.0))
        rx = tx_dbm - self._pl + self._slow + float(self._rng.normal(0.0, 1.5))
        noise = -110.0
        snr = rx - noise
        bler = self._bler_from_snr(snr)

        # throughput proxy: (1 - bler) * shannon(snr_lin)
        snr_lin = 10.0 ** (snr / 10.0)
        thr = (1.0 - bler) * np.log2(1.0 + max(snr_lin, 1e-6))
        # reward: throughput - power cost (mWatts)
        power_mw = 10.0 ** (tx_dbm / 10.0)
        reward = float(thr - 0.001 * power_mw)

        # slow walk PL and slow-fade
        self._pl += float(self._rng.normal(0.0, 0.2))
        self._pl = float(np.clip(self._pl, 145.0, 178.0))
        self._slow = 0.95 * self._slow + float(self._rng.normal(0.0, 1.0))
        self._slow = float(np.clip(self._slow, -8.0, 8.0))
        self._last_bler = bler
        self._last_snr = float(np.clip(snr, -19.99, 29.99))
        self._step += 1
        truncated = self._step >= self.episode_steps
        obs = np.array([self._pl, self._slow, self._last_bler, self._last_snr],
                       dtype=np.float32)
        return obs, reward, False, truncated, {
            "snr_db": snr, "bler": bler, "thr_bps_per_hz": thr,
        }

    def render(self):
        return None

    def close(self):
        return None
