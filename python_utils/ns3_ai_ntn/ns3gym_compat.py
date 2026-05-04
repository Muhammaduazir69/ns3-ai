"""Adaptor exposing canonical ``ns3-gym`` API on top of our Gymnasium envs.

The canonical ``ns3-gym`` (https://github.com/tkn-tub/ns3-gym) ships a Python
wrapper class ``ns3env.Ns3Env(port, stepTime, ...)`` that talks to a
ZeroMQ bridge inside a running ns-3 process. External researchers' scripts
import that class directly — we provide a drop-in shim so legacy scripts
written against ns3-gym work unmodified against our Gymnasium 1.0 envs.

Usage::

    from ns3_ai_ntn.ns3gym_compat import Ns3Env
    env = Ns3Env(envName="HandoverEnv")          # local in-proc, no ns-3 boot
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
        port: int = 5555,
        stepTime: float = 1.0,
        startSim: bool = True,
        simSeed: int = 0,
        simArgs: dict[str, Any] | None = None,
        debug: bool = False,
        envName: str = "HandoverEnv",
        gymnasium_api: bool = False,
    ):
        del port, stepTime, startSim, simSeed, debug  # unused, ns3-gym parity
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
