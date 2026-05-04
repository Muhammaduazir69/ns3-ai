"""Multi-Agent SAC baseline for cooperative multi-beam selection.

The default action space of ``BeamMgmtEnv`` is discrete; SAC needs continuous
control. We wrap it with a ``GumbelBeamWrapper`` that exposes a continuous
softmax-over-beams action and discretises before stepping the inner env.

Each agent owns one UE; per the MAPPO baseline, parameter sharing is used so
only one SAC policy is trained.
"""

from __future__ import annotations

import argparse
import sys

import gymnasium as gym
import numpy as np

from ns3_ai_ntn.envs import BeamMgmtEnv


class GumbelBeamWrapper(gym.Wrapper):
    """Expose continuous logits over beams; argmax is taken before step."""

    def __init__(self, env: BeamMgmtEnv):
        super().__init__(env)
        n = env.action_space.n
        self.action_space = gym.spaces.Box(low=-5.0, high=5.0, shape=(n,),
                                           dtype=np.float32)

    def step(self, action):
        if not isinstance(action, np.ndarray):
            action = np.asarray(action, dtype=np.float32)
        idx = int(np.argmax(action))
        return self.env.step(idx)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-agents", type=int, default=2)
    parser.add_argument("--total-steps", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    try:
        from stable_baselines3 import SAC
        from stable_baselines3.common.env_util import make_vec_env
    except ImportError as exc:
        print(f"[skip] stable-baselines3 not installed: {exc}", file=sys.stderr)
        return 2

    def make_env():
        return GumbelBeamWrapper(BeamMgmtEnv())

    vec = make_vec_env(make_env, n_envs=args.n_agents, seed=args.seed)
    print(f"[masac] training shared SAC policy for {args.n_agents} agents")
    model = SAC("MlpPolicy", vec, verbose=0, seed=args.seed,
                buffer_size=10_000, batch_size=64, learning_starts=100)
    model.learn(total_timesteps=args.total_steps, progress_bar=False)
    print("[masac] training done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
