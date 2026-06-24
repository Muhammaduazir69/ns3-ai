"""Adaptor exposing canonical ``ns3-gym`` API on top of our SYNTHETIC envs.

The canonical ``ns3-gym`` (https://github.com/tkn-tub/ns3-gym) ships a Python
wrapper class ``ns3env.Ns3Env(port, stepTime, ...)`` that talks to a
ZeroMQ bridge inside a running ns-3 process. External researchers' scripts
import that class directly — we provide a drop-in shim so legacy scripts
written against ns3-gym work unmodified against our Gymnasium 1.0 envs.

WARNING: the envs behind this shim (HandoverEnv/BeamMgmtEnv/SliceEnv/
PowerCtrlEnv) are **synthetic placeholders** — they do NOT boot ns-3 and do
NOT read measured KPIs (see ``ns3_ai_ntn.envs``). There is no NTN ns3-ai C++
gym target to connect to. So, unlike upstream ns3-gym, this shim CANNOT honour
a real-ns-3 connection request. If you pass ns-3 connection parameters
(``startSim``, ``port``, ``simSeed``, ``simArgs`` other than ``env``), the
constructor now FAILS LOUDLY instead of silently discarding them and handing
back a synthetic env dressed up as ns-3. To run against a real ns-3 plane, use
``ns3ai_gym_env.Ns3Env`` with an ns3-ai env binary that publishes the NTN
observation over shared memory (that is net-new C++).

Usage (explicit synthetic mode)::

    from ns3_ai_ntn.ns3gym_compat import Ns3Env
    env = Ns3Env(envName="HandoverEnv", startSim=False)  # synthetic, no ns-3
    obs = env.reset()
    obs, reward, done, info = env.step(action)   # legacy 4-tuple

The shim returns the **legacy 4-tuple** that ns3-gym callers expect, not the
Gymnasium 1.0 5-tuple. Set ``gymnasium_api=True`` to opt in to the new tuple.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym

from ns3_ai_ntn.envs import (
    BeamMgmtEnv,
    HandoverEnv,
    PowerCtrlEnv,
    SliceEnv,
)

_REGISTRY: dict[str, type[gym.Env]] = {
    "HandoverEnv": HandoverEnv,
    "BeamMgmtEnv": BeamMgmtEnv,
    "SliceEnv": SliceEnv,
    "PowerCtrlEnv": PowerCtrlEnv,
}


class Ns3Env:
    """Drop-in replacement for ``ns3gym.ns3env.Ns3Env`` (legacy 4-tuple API).

    Constructor mirrors the ns3-gym signature, ignoring fields that are only
    meaningful when an external ns-3 process is being driven (``port``,
    ``stepTime``, ``simSeed``, ...). The relevant field for us is ``envName``
    or ``simArgs={'env': '...'}``.
    """

    def __init__(
        self,
        port: int = 0,
        stepTime: float = 1.0,
        startSim: bool = False,
        simSeed: int = 0,
        simArgs: dict[str, Any] | None = None,
        debug: bool = False,
        envName: str = "HandoverEnv",
        gymnasium_api: bool = False,
    ):
        # Honesty guard (do NOT silently discard ns-3 connection fields):
        # this shim only has synthetic placeholder envs and no NTN ns3-ai C++
        # gym target. If the caller asks for a real ns-3 connection, fail loud.
        real_conn_args = []
        if startSim:
            real_conn_args.append("startSim=True")
        if port:
            real_conn_args.append(f"port={port}")
        if simSeed:
            real_conn_args.append(f"simSeed={simSeed}")
        sim_args_real = {k: v for k, v in (simArgs or {}).items() if k != "env"}
        if sim_args_real:
            real_conn_args.append(f"simArgs={sim_args_real}")
        if real_conn_args:
            raise RuntimeError(
                "ns3_ai_ntn.ns3gym_compat.Ns3Env cannot honour a real ns-3 "
                "connection: there is no NTN ns3-ai C++ gym target and the "
                "envs here are SYNTHETIC placeholders (no measured KPI). "
                f"Refusing to silently fake ns-3 for: {', '.join(real_conn_args)}. "
                "Pass startSim=False (synthetic sandbox), or drive a real ns3-ai "
                "env binary with ns3ai_gym_env.Ns3Env instead."
            )
        del stepTime, debug  # genuinely unused, ns3-gym signature parity
        if simArgs and "env" in simArgs:
            envName = simArgs["env"]
        if envName not in _REGISTRY:
            raise KeyError(
                f"unknown envName {envName!r}; known: {sorted(_REGISTRY)}"
            )
        self._env: gym.Env = _REGISTRY[envName]()
        self._gymnasium_api = bool(gymnasium_api)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self._closed = False

    def reset(self):
        obs, _info = self._env.reset()
        return obs if not self._gymnasium_api else (obs, _info)

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        if self._gymnasium_api:
            return obs, reward, terminated, truncated, info
        # legacy ns3-gym: (obs, reward, done, info)
        done = bool(terminated or truncated)
        return obs, reward, done, info

    def close(self):
        if not self._closed:
            self._env.close()
            self._closed = True

    def render(self):
        return None

    @staticmethod
    def list_envs() -> list[str]:
        return sorted(_REGISTRY)
