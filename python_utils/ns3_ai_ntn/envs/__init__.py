"""SYNTHETIC placeholder Gymnasium environments (NO ns-3 in the loop).

WARNING — these four environments are *synthetic placeholders*. They do NOT
boot ns-3 and do NOT read any measured KPI. Their RSRP / SINR / BLER /
throughput come from closed-form proxy formulas, not from a real radio plane.
There is currently NO ns3-ai C++ env binary for handover / power-control /
beam-management / slicing to step against (the only real shared-memory targets
shipped by this module are the upstream demos: ``ns3ai_ltecqi_msg``,
``ns3ai_rltcp_*``, ``ns3ai_apb_*``, ``ns3ai_ratecontrol_*`` — none expose NTN
observations). Use these envs ONLY as a fast policy-search sandbox; never
report their outputs as measured ns-3 results.

To step against a real ns-3 plane, write an ns3-ai env binary that publishes
the NTN observation through the shared-memory channel and drive it with
``ns3ai_gym_env.Ns3Env`` (see ``examples/lte-cqi/use-msg/run_online_lstm.py``);
that is net-new C++ and is out of scope for these Python placeholders.

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
