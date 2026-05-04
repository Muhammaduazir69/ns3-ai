"""W4 unit tests: every Gymnasium env passes the SDK env_checker."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from ns3_ai_ntn.envs import BeamMgmtEnv, HandoverEnv, PowerCtrlEnv, SliceEnv

ALL_ENVS = [HandoverEnv, BeamMgmtEnv, SliceEnv, PowerCtrlEnv]


@pytest.mark.parametrize("cls", ALL_ENVS)
def test_check_env(cls):
    env = cls()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        check_env(env, skip_render_check=True)


@pytest.mark.parametrize("cls", ALL_ENVS)
def test_reset_seed_determinism(cls):
    env_a, env_b = cls(), cls()
    obs_a, _ = env_a.reset(seed=42)
    obs_b, _ = env_b.reset(seed=42)
    np.testing.assert_array_equal(np.asarray(obs_a), np.asarray(obs_b))


def test_handover_handover_is_logged():
    env = HandoverEnv(n_candidates=3, episode_steps=64)
    env.reset(seed=0)
    saw_ho = False
    for _ in range(50):
        # always pick candidate 1 — guaranteed to trigger handovers eventually
        _, _, _, _, info = env.step(1)
        if info.get("handover"):
            saw_ho = True
            break
    assert saw_ho


def test_slice_action_normalised():
    env = SliceEnv()
    env.reset(seed=0)
    # All-zero action must not crash; PRBs reduce to uniform 1/3 split.
    obs, r, term, trunc, info = env.step(np.zeros(3, dtype=np.float32))
    assert obs.shape == (7,)
    assert -2.0 < r < 2.0


def test_powerctrl_clipping():
    env = PowerCtrlEnv()
    env.reset(seed=0)
    # power above limit must be silently clipped, not raise
    obs, r, term, trunc, info = env.step(np.array([100.0], dtype=np.float32))
    assert obs.shape == (4,)
    assert "snr_db" in info
