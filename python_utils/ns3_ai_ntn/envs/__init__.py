"""Gymnasium environments wrapping the ns3-ntn-toolkit physical models.

Each env is a self-contained Python simulator that mimics the dominant
behavior of its C++ counterpart so RL training can iterate at >10 kHz on a
laptop without spinning up ns-3 every step. The C++ side stays the source of
truth for protocol fidelity; these envs reproduce its observable dynamics for
the purpose of policy search.

Available environments::

    HandoverEnv      — ntn-cho conditional-handover decision under TA / RSRP
    BeamMgmtEnv      — multi-beam selection on a Walker shell
    SliceEnv         — eMBB / URLLC / mMTC slice resource allocation (W6 stub)
    PowerCtrlEnv     — uplink open-loop power control over an LEO link
"""

from ns3_ai_ntn.envs.handover_env import HandoverEnv
from ns3_ai_ntn.envs.beam_mgmt_env import BeamMgmtEnv
from ns3_ai_ntn.envs.slice_env import SliceEnv
from ns3_ai_ntn.envs.power_ctrl_env import PowerCtrlEnv

__all__ = ["HandoverEnv", "BeamMgmtEnv", "SliceEnv", "PowerCtrlEnv"]
