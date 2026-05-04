"""Multi-beam selection env on a Walker shell — discrete action over ``n_beams``.

The cell projects ``n_beams`` fixed beams arranged on a triangular lattice on
the ground; the UE moves through the footprint. The agent picks the best beam
each step. Reward is the resulting SNR (dB) divided by 30 to land in roughly
[-1, +1]. The optimal policy is pointwise nearest-beam — easy enough that
``check_env`` finishes quickly but non-trivial as a sanity test for SB3.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class BeamMgmtEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        n_beams: int = 7,
        footprint_km: float = 50.0,
        ue_speed_kmh: float = 60.0,
        episode_steps: int = 200,
        beam_3db_km: float = 12.0,
        seed: int | None = None,
    ):
        super().__init__()
        self.n_beams = int(n_beams)
        self.footprint_km = float(footprint_km)
        self.ue_speed_mps = float(ue_speed_kmh) * 1000.0 / 3600.0
        self.episode_steps = int(episode_steps)
        self.beam_3db_km = float(beam_3db_km)

        # Beams arranged on a hex pattern within the footprint.
        self._beams = self._lay_out_beams(self.n_beams, self.footprint_km)

        # Obs: UE (x, y) in km, plus per-beam range_km vector.
        obs_dim = 2 + self.n_beams
        low = np.concatenate([
            np.array([-footprint_km, -footprint_km], dtype=np.float32),
            np.zeros(self.n_beams, dtype=np.float32),
        ])
        high = np.concatenate([
            np.array([footprint_km, footprint_km], dtype=np.float32),
            np.full(self.n_beams, 2.0 * footprint_km, dtype=np.float32),
        ])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Discrete(self.n_beams)

        self._rng = np.random.default_rng(seed)
        self._ue: np.ndarray | None = None
        self._heading = 0.0
        self._step = 0

    @staticmethod
    def _lay_out_beams(n: int, footprint_km: float) -> np.ndarray:
        """Hex-ish lattice of n beam centres in the [-r, r]² square."""
        rng = np.random.default_rng(7)  # deterministic layout
        # Quasi-random Sobol-ish — good enough for this synthetic env.
        pts = []
        spacing = footprint_km / max(np.sqrt(n), 1.0)
        cols = int(np.ceil(np.sqrt(n)))
        idx = 0
        for r in range(cols):
            for c in range(cols):
                if idx >= n:
                    break
                x = -footprint_km / 2 + c * spacing + (spacing / 2 if r % 2 else 0)
                y = -footprint_km / 2 + r * spacing
                # tiny dither
                x += rng.normal(0, spacing * 0.03)
                y += rng.normal(0, spacing * 0.03)
                pts.append([x, y])
                idx += 1
        return np.asarray(pts[:n], dtype=np.float32)

    def _ranges(self) -> np.ndarray:
        return np.linalg.norm(self._beams - self._ue, axis=1).astype(np.float32)

    def _snr_db(self, beam: int) -> float:
        # Boresight gain 35 dB, falls off Gaussian-style; path loss constant.
        d = float(np.linalg.norm(self._beams[beam] - self._ue))
        gain = 35.0 - 12.0 * (d / self.beam_3db_km) ** 2
        path_loss = 162.0  # dB at LEO link budget point
        tx_pwr = 50.0  # dBm EIRP per beam
        rsrp = tx_pwr + gain - path_loss
        noise = -110.0
        return float(rsrp - noise + self._rng.normal(0.0, 1.0))

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        # Random UE start near footprint edge, heading inward.
        ang = self._rng.uniform(0, 2 * np.pi)
        r0 = self.footprint_km * 0.45
        self._ue = np.array([r0 * np.cos(ang), r0 * np.sin(ang)],
                            dtype=np.float32)
        # heading toward origin plus jitter
        self._heading = float(np.arctan2(-self._ue[1], -self._ue[0])
                              + self._rng.normal(0.0, 0.4))
        self._step = 0
        obs = np.concatenate([self._ue, self._ranges()]).astype(np.float32)
        return obs, {}

    def step(self, action: int):
        action = int(action)
        if not 0 <= action < self.n_beams:
            raise ValueError(f"invalid action {action}")
        snr = self._snr_db(action)
        reward = float(np.clip(snr / 30.0, -1.0, 1.5))

        # advance UE 1 second
        dx = self.ue_speed_mps * np.cos(self._heading) * 1.0 / 1000.0
        dy = self.ue_speed_mps * np.sin(self._heading) * 1.0 / 1000.0
        self._ue = self._ue + np.array([dx, dy], dtype=np.float32)
        # bounce inside the footprint
        if abs(self._ue[0]) > self.footprint_km / 2 or abs(self._ue[1]) > self.footprint_km / 2:
            self._heading += np.pi + float(self._rng.normal(0, 0.2))
        self._step += 1
        obs = np.concatenate([self._ue, self._ranges()]).astype(np.float32)
        truncated = self._step >= self.episode_steps
        return obs, reward, False, truncated, {"snr_db": snr}

    def render(self):
        return None

    def close(self):
        return None
