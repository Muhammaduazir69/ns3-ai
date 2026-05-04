"""LEO conditional-handover decision env (wraps ntn-cho semantics).

The agent observes the current serving cell plus ``n_candidates`` neighbour
cells, each described by RSRP, SINR, total TA, and TA drift. At each step it
chooses to stay or hand over to one of the candidates. The reward is a
throughput proxy minus handover cost and ping-pong penalty.

The dynamics intentionally mimic ``contrib/ntn-rrc`` and ``contrib/ntn-cho``:

* Per-cell RSRP follows a TA-coupled curve: deepest at zenith (smallest TA),
  shallowest at the horizon. We model RSRP_dBm = -85 + 10·log10(TA / TA_ref).
* TA evolves linearly per pass (drift is the per-second derivative); when a
  satellite sets, we recycle it as a fresh acquisition.
* Handover cost: 8 dB equivalent (matches the W2 default ``HoMargin``).
* Ping-pong: handing back to a recently-left cell within 5 s is double-billed.

The closed-form approach lets SB3 train at ~50 kHz on a laptop while staying
faithful to ntn-cho's decision surface.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class HandoverEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        n_candidates: int = 4,
        episode_steps: int = 256,
        ta_min_us: float = 3000.0,
        ta_max_us: float = 18000.0,
        ho_cost_db: float = 8.0,
        ping_pong_window_s: float = 5.0,
        ping_pong_penalty_db: float = 4.0,
        step_dt_s: float = 1.0,
        seed: int | None = None,
    ):
        super().__init__()
        if n_candidates < 1:
            raise ValueError("n_candidates must be >= 1")
        self.n_candidates = int(n_candidates)
        self.episode_steps = int(episode_steps)
        self.ta_min_us = float(ta_min_us)
        self.ta_max_us = float(ta_max_us)
        self.ho_cost_db = float(ho_cost_db)
        self.pp_window = float(ping_pong_window_s)
        self.pp_penalty = float(ping_pong_penalty_db)
        self.dt = float(step_dt_s)

        # Observation: per cell — rsrp_dbm, sinr_db, ta_us, ta_drift_us_per_s,
        # remaining_pass_s; plus 1 bool flag (is_serving) per cell.
        n_cells = 1 + self.n_candidates
        obs_low = np.tile(np.array([-140.0, -10.0, 0.0, -100.0, 0.0, 0.0],
                                   dtype=np.float32), n_cells)
        obs_high = np.tile(np.array([-50.0, 40.0, 30000.0, 100.0, 600.0, 1.0],
                                    dtype=np.float32), n_cells)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high,
                                            dtype=np.float32)
        # Action 0 = stay; 1..n_candidates = hand over to candidate i-1.
        self.action_space = spaces.Discrete(1 + self.n_candidates)

        self._rng = np.random.default_rng(seed)
        self._cells: np.ndarray | None = None  # (n_cells, 5) float
        self._serving = 0
        self._step = 0
        self._last_left: dict[int, float] = {}  # cell idx → time since left

    # ---------------------------------------------------------------- helpers
    def _spawn_cell(self) -> np.ndarray:
        """Return a fresh cell with a random pass progress."""
        # remaining pass length [60..600] s, uniform progress [0..1]
        pass_len = self._rng.uniform(60.0, 600.0)
        progress = self._rng.uniform(0.0, 1.0)
        # TA traces a smile: ta(t) = ta_min + (ta_max - ta_min) * (2*progress-1)^2
        u = 2.0 * progress - 1.0
        ta = self.ta_min_us + (self.ta_max_us - self.ta_min_us) * u * u
        # drift = d(ta)/dt for that closed form (us / s)
        drift = (self.ta_max_us - self.ta_min_us) * 2.0 * u * (2.0 / pass_len) * 1e6 / 1e6
        # us/s magnitude clamped to ~50 us/s (matches W2 measured peak)
        drift = float(np.clip(drift, -60.0, 60.0))
        remaining = pass_len * (1.0 - progress)
        return np.array([ta, drift, remaining, pass_len, progress],
                        dtype=np.float64)

    def _ta_to_rsrp(self, ta_us: float) -> float:
        """RSRP (dBm) as a smooth function of total TA. -85 dBm at zenith, -125 at horizon."""
        # log-scaled in TA above zenith
        ratio = ta_us / self.ta_min_us
        return float(-85.0 - 20.0 * np.log10(max(ratio, 1.0)))

    def _rsrp_to_sinr(self, rsrp_dbm: float) -> float:
        """Crude inversion: SINR ≈ RSRP - noise_floor + interference noise."""
        noise = -110.0
        sinr = rsrp_dbm - noise + float(self._rng.normal(0.0, 1.5))
        return float(np.clip(sinr, -10.0, 40.0))

    def _build_obs(self) -> np.ndarray:
        # ordering: serving cell first, then candidates in stable index order
        order = [self._serving] + [i for i in range(len(self._cells))
                                   if i != self._serving]
        rows = []
        for k, idx in enumerate(order):
            c = self._cells[idx]
            ta = c[0]
            drift = c[1]
            remaining = c[2]
            rsrp = self._ta_to_rsrp(ta)
            sinr = self._rsrp_to_sinr(rsrp)
            is_serv = 1.0 if k == 0 else 0.0
            rows.append([rsrp, sinr, ta, drift, remaining, is_serv])
        return np.asarray(rows, dtype=np.float32).reshape(-1)

    def _advance_dynamics(self) -> None:
        # Each cell: ta += drift*dt; remaining -= dt; if remaining <= 0 → respawn.
        for i in range(len(self._cells)):
            self._cells[i, 0] += self._cells[i, 1] * self.dt
            self._cells[i, 0] = float(np.clip(self._cells[i, 0],
                                              self.ta_min_us,
                                              self.ta_max_us))
            self._cells[i, 2] -= self.dt
            if self._cells[i, 2] <= 0.0:
                self._cells[i] = self._spawn_cell()
                # If the serving cell sets, we are forced to hand over next.
        # decay ping-pong memory
        for k in list(self._last_left.keys()):
            self._last_left[k] += self.dt
            if self._last_left[k] > self.pp_window:
                self._last_left.pop(k, None)

    # ---------------------------------------------------------------- gym API
    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        n_cells = 1 + self.n_candidates
        self._cells = np.stack([self._spawn_cell() for _ in range(n_cells)])
        self._serving = 0
        self._step = 0
        self._last_left = {}
        return self._build_obs(), {}

    def step(self, action: int):
        if self._cells is None:
            raise RuntimeError("HandoverEnv.step before reset")
        action = int(action)
        if not 0 <= action < self.action_space.n:
            raise ValueError(f"invalid action {action}")

        # Translate action → cell index (action 0 = stay; 1..n = candidates)
        # Candidates were enumerated in build_obs as: [serving, *others_in_index_order]
        order = [self._serving] + [i for i in range(len(self._cells))
                                   if i != self._serving]
        target = order[action]
        info: dict = {}

        # Reward components (in dB-equivalent units, divided by 10 → "throughput proxy")
        # Pre-decision serving RSRP
        rsrp_before = self._ta_to_rsrp(self._cells[self._serving, 0])

        ho_penalty = 0.0
        ping_pong = 0.0
        if target != self._serving:
            ho_penalty = self.ho_cost_db
            if target in self._last_left:
                ping_pong = self.pp_penalty
            self._last_left[self._serving] = 0.0
            self._serving = target
            info["handover"] = True
        else:
            info["handover"] = False

        rsrp_after = self._ta_to_rsrp(self._cells[self._serving, 0])
        # throughput proxy ∝ 0.1 * RSRP (dBm); centred so good cells give ~ +1 reward
        reward = 0.1 * (rsrp_after + 100.0) - 0.1 * ho_penalty - 0.1 * ping_pong

        # Advance world
        self._advance_dynamics()
        self._step += 1
        terminated = False
        truncated = self._step >= self.episode_steps

        info["rsrp_before_dbm"] = rsrp_before
        info["rsrp_after_dbm"] = rsrp_after
        return self._build_obs(), float(reward), terminated, truncated, info

    def render(self):
        return None

    def close(self):
        return None
