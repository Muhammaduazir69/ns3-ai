"""SYNTHETIC eMBB / URLLC / mMTC slice-resource allocation env (NO ns-3).

WARNING: synthetic placeholder. PRB->throughput and per-slice satisfaction are
closed-form proxies; the only randomness is the per-slice *demand* (a traffic
load model, not a measured KPI). It does NOT step the C++ ``ntn-slice`` xApp
through ns3-ai shared memory. Do not present its outputs as real measurements.

Continuous Box action over the simplex of PRB shares. Observation is the
current per-slice load (Mbps demand) and aggregate KPI history. Reward is the
weighted satisfaction across slices — URLLC weights its latency tail much
heavier than eMBB / mMTC.

To make this real, an ns3-ai C++ env binary would have to publish the
``ntn-slice`` RIC step's measured per-slice goodput/latency through the
shared-memory channel; that is net-new C++ and out of scope here.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# canonical slice ordering — match O-RAN slice/sst registry
SLICES = ("eMBB", "URLLC", "mMTC")
WEIGHTS = np.array([0.4, 0.5, 0.1], dtype=np.float32)


class SliceEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        total_prb: int = 273,        # 100 MHz @ 30 kHz SCS in NR FR1
        episode_steps: int = 200,
        seed: int | None = None,
    ):
        super().__init__()
        self.total_prb = int(total_prb)
        self.episode_steps = int(episode_steps)

        # Action: simplex over 3 slices in [0,1]; we'll re-normalise.
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(3,),
                                       dtype=np.float32)
        # Obs: demand (Mbps) per slice, satisfaction history per slice (1-step), aggregate util.
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([2000, 2000, 2000, 1, 1, 1, 1], dtype=np.float32),
            dtype=np.float32,
        )

        self._rng = np.random.default_rng(seed)
        self._demand: np.ndarray | None = None
        self._last_sat = np.zeros(3, dtype=np.float32)
        self._step = 0

    def _sample_demand(self) -> np.ndarray:
        # Mbps demand per slice — diurnal-ish jitter
        base = np.array([400.0, 80.0, 20.0], dtype=np.float32)
        scale = np.array([0.6, 0.4, 0.3], dtype=np.float32)
        d = base * (1.0 + scale * self._rng.standard_normal(3).astype(np.float32))
        return np.maximum(d, 5.0)

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._demand = self._sample_demand()
        self._last_sat = np.zeros(3, dtype=np.float32)
        self._step = 0
        obs = np.concatenate([self._demand, self._last_sat,
                              np.array([0.0], dtype=np.float32)])
        return obs, {}

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape != (3,):
            raise ValueError(f"action must be shape (3,), got {action.shape}")
        action = np.clip(action, 0.0, 1.0)
        s = action.sum()
        shares = action / s if s > 1e-6 else np.full(3, 1.0 / 3, dtype=np.float32)
        prb = (shares * self.total_prb).astype(np.float32)

        # PRB → throughput: ~0.6 Mbps/PRB at MCS 16 average
        capacity = prb * 0.6
        served = np.minimum(self._demand, capacity)
        sat = served / np.maximum(self._demand, 1e-3)
        # URLLC penalty for under-provisioning: square the deficit
        urllc_penalty = (1.0 - sat[1]) ** 2
        reward = float((WEIGHTS * sat).sum() - 0.3 * urllc_penalty)

        self._last_sat = sat.astype(np.float32)
        util = float(capacity.sum() / max(self.total_prb * 0.6, 1e-3))
        self._demand = self._sample_demand()
        self._step += 1
        truncated = self._step >= self.episode_steps
        obs = np.concatenate([self._demand, self._last_sat,
                              np.array([util], dtype=np.float32)]).astype(np.float32)
        info = {"served_mbps": served.tolist(), "satisfaction": sat.tolist()}
        return obs, reward, False, truncated, info

    def render(self):
        return None

    def close(self):
        return None
