"""W4 SB3 integration test: PPO trains for a brief budget without error."""

from __future__ import annotations

import pytest

from ns3_ai_ntn.envs import HandoverEnv

sb3 = pytest.importorskip("stable_baselines3")


def test_ppo_quick_train():
    from stable_baselines3 import PPO

    env = HandoverEnv(episode_steps=64)
    model = PPO("MlpPolicy", env, verbose=0, seed=0,
                n_steps=64, batch_size=32, n_epochs=2)
    model.learn(total_timesteps=1024, progress_bar=False)
    obs, _ = env.reset(seed=0)
    action, _ = model.predict(obs, deterministic=True)
    assert env.action_space.contains(int(action))
