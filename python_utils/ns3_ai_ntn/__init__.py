"""ns3-ai-ntn — RL/GNN/MARL extensions for the ns3-ntn-toolkit (Workstream W4).

Top-level subpackages:

- ``envs``   — Gymnasium 0.29+/1.0 environments wrapping ntn-cho / ntn-rrc /
               ntn-observability (handover, beam-mgmt, slice, power-ctrl).
- ``sb3``    — Stable-Baselines3 training scripts (PPO baseline).
- ``gnn``    — PyTorch Geometric models for constellation-graph learning.
- ``marl``   — multi-agent RL baselines (MAPPO, MASAC).

The legacy ``ns3ai_utils`` shared-memory bridge (Huazhong original) lives at the
sibling top-level path; it is preserved untouched for backwards compatibility.
"""

__all__ = ["envs", "sb3", "gnn", "marl"]
__version__ = "0.1.0"
