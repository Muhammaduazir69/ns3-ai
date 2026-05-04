"""Train PPO on ``HandoverEnv`` and compare to a random baseline.

Usage::

    python -m ns3_ai_ntn.sb3.train_ppo_handover \
        --total-steps 10000 --eval-episodes 30 --seed 0

Reports mean ± std reward for both the trained policy and the random baseline,
plus the gap in reward-per-step. The W4 validation gate is::

    PPO_mean - random_mean  >  random_std

i.e. PPO must beat random by at least one σ on the eval split.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from ns3_ai_ntn.envs import HandoverEnv


def _eval_policy(env_factory, predict_fn, n_episodes: int, seed: int):
    rewards = []
    lengths = []
    for ep in range(n_episodes):
        env = env_factory()
        obs, _ = env.reset(seed=seed + ep * 13)
        done = False
        ep_r = 0.0
        ep_l = 0
        while not done:
            action = predict_fn(obs)
            obs, r, term, trunc, _ = env.step(action)
            ep_r += r
            ep_l += 1
            done = term or trunc
        rewards.append(ep_r)
        lengths.append(ep_l)
    return np.array(rewards), np.array(lengths)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-steps", type=int, default=10000)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-candidates", type=int, default=4)
    parser.add_argument("--episode-steps", type=int, default=128)
    parser.add_argument("--out", type=str, default=None,
                        help="optional JSON results path")
    args = parser.parse_args(argv)

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
    except ImportError as exc:
        print(f"[skip] stable-baselines3 not installed: {exc}", file=sys.stderr)
        return 2

    def make_env():
        return HandoverEnv(n_candidates=args.n_candidates,
                           episode_steps=args.episode_steps)

    train_env = make_vec_env(make_env, n_envs=4, seed=args.seed)
    print(f"[ppo] training PPO for {args.total_steps} steps "
          f"(n_candidates={args.n_candidates}, seed={args.seed})")
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=0,
        seed=args.seed,
        n_steps=128,
        batch_size=64,
        n_epochs=4,
        learning_rate=3e-4,
    )
    t0 = time.time()
    model.learn(total_timesteps=args.total_steps, progress_bar=False)
    train_secs = time.time() - t0
    print(f"[ppo] trained in {train_secs:.1f}s")

    rng = np.random.default_rng(args.seed + 999)

    def random_pred(obs):
        return int(rng.integers(0, 1 + args.n_candidates))

    def ppo_pred(obs):
        action, _ = model.predict(obs, deterministic=True)
        return int(action)

    print(f"[eval] running {args.eval_episodes} episodes per policy")
    ppo_rew, _ = _eval_policy(make_env, ppo_pred,
                              args.eval_episodes, args.seed + 1000)
    rnd_rew, _ = _eval_policy(make_env, random_pred,
                              args.eval_episodes, args.seed + 1000)

    ppo_mean, ppo_std = float(ppo_rew.mean()), float(ppo_rew.std())
    rnd_mean, rnd_std = float(rnd_rew.mean()), float(rnd_rew.std())
    gap = ppo_mean - rnd_mean
    threshold = rnd_std

    result = {
        "total_steps": args.total_steps,
        "eval_episodes": args.eval_episodes,
        "n_candidates": args.n_candidates,
        "train_seconds": train_secs,
        "ppo_mean": ppo_mean,
        "ppo_std": ppo_std,
        "random_mean": rnd_mean,
        "random_std": rnd_std,
        "gap": gap,
        "threshold_one_sigma": threshold,
        "ppo_beats_random_by_one_sigma": bool(gap > threshold),
    }
    print("[result] " + json.dumps(result, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0 if result["ppo_beats_random_by_one_sigma"] else 1


if __name__ == "__main__":
    sys.exit(main())
