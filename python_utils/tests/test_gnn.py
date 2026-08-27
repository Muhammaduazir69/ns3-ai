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


def test_gnn_is_scored_on_held_out_nodes_against_real_baselines():
    """AI-08: the 70% gate was memorisation, and the honest numbers are these.

    The old gate trained on all 50 nodes, then scored the model on the very
    logits its last gradient step had produced, over those same nodes, with
    ``n_classes = n``. Each node needed only to recall its own index. That
    number could not fail for any reason connected to learning constellation
    topology, and nothing in ns-3 consumes the GAT, so it had no downstream
    meaning either.

    Three things changed. The label is now which of a satellite's ISL
    neighbours lies on the shortest path to a fixed ground gateway, a 4-class
    task whose class space is the same for every node. The nodes are split, and
    the accuracy reported is measured on nodes the optimizer never saw. And the
    node features carry the per-slot geometry a routing decision actually turns
    on, without which the label was not recoverable from the features at all.

    What the corrected measurement shows is that the model does NOT generalise.
    Measured on this graph: 100% on the training nodes, about 27% on held-out
    ones, against a 32% majority-class rate and a 68% trivial greedy-direction
    heuristic. The GAT is beaten by both. This test records that rather than
    hiding it, and nothing in ns-3 consumes the GAT, so the number has no
    downstream meaning either way: it is an offline sandbox.

    If someone improves the model and the final assertion starts failing, that
    is good news and the gate should be raised, not deleted.
    """
    import numpy as np

    demo = starlink_subset_demo(n_sats=50, seed=0)
    pos = demo["positions"]
    nb = demo["neighbors"]
    lab = demo["labels"]
    gw = pos[demo["gateway"]]

    data = build_pyg_data(pos, demo["edges"], neighbors=nb, gateway_pos=gw)
    labels = torch.from_numpy(lab)

    # The task must be non-degenerate: a single dominant class would let any
    # constant predictor look good.
    counts = np.bincount(lab, minlength=demo["n_classes"])
    chance = counts.max() / len(lab)
    assert (counts > 0).all(), f"every next-hop slot must occur: {counts.tolist()}"
    assert chance < 0.5, f"majority class is {chance:.2%}; the task is too skewed to score"

    # A non-learned baseline: forward to whichever neighbour lies most nearly in
    # the gateway's direction. Any learned model worth reporting must be
    # compared against this, not against chance alone.
    greedy = np.zeros(len(lab), dtype=np.int64)
    for i in range(len(lab)):
        to_gw = gw - pos[i]
        best, bi = -9.0, 0
        for s in range(nb.shape[1]):
            v = pos[nb[i, s]] - pos[i]
            c = float(np.dot(to_gw, v) / (np.linalg.norm(to_gw) * np.linalg.norm(v) + 1e-12))
            if c > best:
                best, bi = c, s
        greedy[i] = bi
    greedy_acc = float((greedy == lab).mean())
    assert greedy_acc > chance, (
        f"the greedy baseline ({greedy_acc:.2%}) must beat chance ({chance:.2%}), or the "
        f"features carry no routing signal and nothing here is measurable"
    )

    model, val_acc = train_quick(
        data, labels, epochs=300, lr=5e-3, n_classes=demo["n_classes"]
    )
    train_acc = model.last_train_acc

    # The model must at least fit what it was shown; if it cannot, the harness
    # is broken rather than the model being honest.
    assert train_acc > 0.8, f"training accuracy {train_acc:.2%} — the harness is not learning"

    # It does NOT beat chance on held-out nodes. Measured: about 27% against a
    # 32% majority-class rate. That is recorded here as the finding rather than
    # asserted away, because asserting `val_acc >= chance` would have been
    # claiming more than the measurement supports, which is the exact habit this
    # whole audit is about.
    assert 0.0 <= val_acc <= 1.0, "sanity"

    # The honest part: the model memorises. Recorded as an assertion so the gap
    # cannot be quietly forgotten, and so an improvement is noticed.
    assert train_acc - val_acc > 0.3, (
        f"train {train_acc:.2%} vs val {val_acc:.2%}: if this gap has closed the model now "
        f"generalises and this test should be replaced by a real accuracy gate"
    )
    assert greedy_acc > val_acc, (
        f"a trivial greedy-direction heuristic ({greedy_acc:.2%}) still beats the GAT on "
        f"held-out nodes ({val_acc:.2%}, against a {chance:.2%} majority-class rate). If this "
        f"ever fails the model has genuinely improved: raise the gate rather than deleting the "
        f"assertion"
    )


def test_gnn_labels_are_node_independent():
    """AI-08: the class space must not be the node set.

    The label used to be the index of the nearest neighbour, so there were as
    many classes as nodes and a held-out node's answer was a class the model may
    never have seen. No split can rescue that; the task itself does not
    generalise.
    """
    small = starlink_subset_demo(n_sats=20, seed=1)
    big = starlink_subset_demo(n_sats=60, seed=1)
    assert small["n_classes"] == big["n_classes"], (
        "the number of classes must not depend on the number of satellites"
    )
    assert small["labels"].max() < small["n_classes"]
    assert big["labels"].max() < big["n_classes"]
    # And the label must index the neighbour list, so it names a real ISL.
    for i in range(len(small["labels"])):
        slot = int(small["labels"][i])
        assert 0 <= slot < small["neighbors"].shape[1]


def test_ns3gym_compat_smoke():
    from ns3_ai_ntn.ns3gym_compat import Ns3Env

    env = Ns3Env(envName="HandoverEnv")
    obs = env.reset()
    assert env.observation_space.contains(obs)
    obs, reward, done, info = env.step(0)
    assert isinstance(done, bool)
    assert isinstance(reward, float)
    env.close()


# ---------------------------------------------------------------------------
# AI-11: the C++<->Python shared-memory contract has a version and schema
# handshake, and the two sides must compute the schema digest identically.
# ---------------------------------------------------------------------------

def test_ns3ai_schema_hash_matches_the_cpp_definition():
    """AI-11: both peers must hash the same bytes the same way.

    SimInitMsg used to carry only the two spaces and SimInitAck only
    done/stopSimReq, so neither side ever checked it was talking to a compatible
    peer: an agent built against one observation layout and a scenario built
    against another would connect, exchange bytes and produce silently
    meaningless numbers. The module had a VERSION file and neither it nor
    anything else crossed the wire.

    The digest is FNV-1a over the serialised obs space followed by the
    serialised act space, computed identically in
    model/gym-interface/cpp/ns3-ai-gym-interface.cc. If the two implementations
    ever diverge, every connection fails with a schema mismatch that is not a
    real mismatch, so this pins the algorithm rather than trusting it.
    """
    import sys
    from pathlib import Path

    pydir = Path(__file__).resolve().parents[1].parent / "model" / "gym-interface" / "py"
    sys.path.insert(0, str(pydir))
    import messages_pb2 as pb
    from ns3ai_gym_env.envs.ns3_environment import (
        NS3AI_HANDSHAKE_VERSION,
        _schema_hash,
    )

    m = pb.SimInitMsg()
    m.obsSpace.type = pb.Box
    m.obsSpace.name = "obs"
    box = pb.BoxSpace()
    box.low = -1.0
    box.high = 1.0
    m.obsSpace.space.Pack(box)
    m.actSpace.type = pb.Discrete
    m.actSpace.name = "act"
    d = pb.DiscreteSpace()
    d.n = 2
    m.actSpace.space.Pack(d)

    # Verified against the C++ SchemaHashOf() on these exact serialised bytes.
    assert _schema_hash(m) == "a1248ee8401fe878", (
        "the Python digest no longer matches the value the C++ side computes for "
        "this message; the two implementations have diverged"
    )

    # It must actually depend on the layout, or it cannot detect a mismatch.
    m2 = pb.SimInitMsg()
    m2.CopyFrom(m)
    d2 = pb.DiscreteSpace()
    d2.n = 7  # a different action space
    m2.actSpace.space.Pack(d2)
    assert _schema_hash(m2) != _schema_hash(m), (
        "changing the action space must change the digest; a digest that ignores "
        "the layout would accept every mismatch"
    )

    # The handshake message must carry the new fields at all.
    assert "protocolVersion" in [f.name for f in pb.SimInitMsg.DESCRIPTOR.fields]
    assert "schemaHash" in [f.name for f in pb.SimInitMsg.DESCRIPTOR.fields]
    ack_fields = [f.name for f in pb.SimInitAck.DESCRIPTOR.fields]
    for needed in ("protocolVersion", "moduleVersion", "schemaHash", "compatible",
                   "incompatibleReason"):
        assert needed in ack_fields, f"SimInitAck must carry {needed}"

    # An old peer sends none of them, and that must remain parseable rather than
    # becoming a hard error: this is a compatibility check, not a version wall.
    legacy = pb.SimInitAck()
    legacy.done = True
    round_tripped = pb.SimInitAck()
    round_tripped.ParseFromString(legacy.SerializeToString())
    assert round_tripped.protocolVersion == ""
    assert round_tripped.compatible is False
    assert NS3AI_HANDSHAKE_VERSION == "1"
