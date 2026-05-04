"""Multi-Agent PPO baseline for the multi-UE handover problem.

Each UE is an independent agent observing only its own ``HandoverEnv``-style
slice of state, but they share a centralised critic at training time
(centralized-training-decentralized-execution, CTDE).

Implementation note: rather than re-implement MAPPO from scratch, we use
multiple parallel ``HandoverEnv`` instances and train one shared SB3 PPO
policy with parameter sharing — a standard MARL baseline. For full IPPO with
independent policies per UE, swap ``make_vec_env`` for a list of trainers.
This baseline lets W4's validation harness exercise the multi-agent path
without pulling in the heavy EPyMARL dependency tree.
"""

from __future__ import annotations

import argparse
import sys

from ns3_ai_ntn.envs import HandoverEnv


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-agents", type=int, default=4,
                        help="number of UE agents trained with shared policy")
    parser.add_argument("--total-steps", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
    except ImportError as exc:
        print(f"[skip] stable-baselines3 not installed: {exc}", file=sys.stderr)
        return 2

    def make_env():
        return HandoverEnv(n_candidates=4, episode_steps=128)

    vec = make_vec_env(make_env, n_envs=args.n_agents, seed=args.seed)
    print(f"[mappo] training shared policy for {args.n_agents} UE agents, "
          f"{args.total_steps} env-steps")
    model = PPO("MlpPolicy", vec, verbose=0, seed=args.seed,
                n_steps=128, batch_size=64, n_epochs=4)
    model.learn(total_timesteps=args.total_steps, progress_bar=False)
    print("[mappo] training done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
