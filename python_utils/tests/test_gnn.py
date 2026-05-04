"""W4 GNN tests: forward pass + accuracy gate on a 50-node Starlink subset."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from ns3_ai_ntn.gnn.constellation_graph import build_pyg_data, starlink_subset_demo
from ns3_ai_ntn.gnn.gat_topology import GATTopology, train_quick


def test_forward_pass():
    demo = starlink_subset_demo(n_sats=50, seed=0)
    data = build_pyg_data(demo["positions"], demo["edges"])
    model = GATTopology(in_dim=4, hidden=16, n_classes=50, heads=2)
    out = model(data)
    assert out.shape == (50, 50)
    assert torch.isfinite(out).all()


def test_accuracy_gate_70pct():
    """Train the GAT on the synthetic next-hop labels — must clear 70%."""
    demo = starlink_subset_demo(n_sats=50, seed=0)
    data = build_pyg_data(demo["positions"], demo["edges"])
    labels = torch.from_numpy(demo["labels"])
    _, acc = train_quick(data, labels, epochs=300, lr=5e-3)
    assert acc >= 0.70, f"GNN accuracy {acc:.2%} below 70% gate"


def test_ns3gym_compat_smoke():
    from ns3_ai_ntn.ns3gym_compat import Ns3Env

    env = Ns3Env(envName="HandoverEnv")
    obs = env.reset()
    assert env.observation_space.contains(obs)
    obs, reward, done, info = env.step(0)
    assert isinstance(done, bool)
    assert isinstance(reward, float)
    env.close()
