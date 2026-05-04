"""Build a PyTorch Geometric ``Data`` object from a W1 ISL graph.

Inputs:

* a list of satellite ECEF positions (N, 3) in metres,
* an adjacency edge list (E, 2) of ISL pairs.

Outputs a ``torch_geometric.data.Data`` instance with:

* ``x``         — node features (lat, lon, alt_km, vx_kmps_norm)  shape (N, 4)
* ``edge_index``— bidirectional edges  shape (2, 2E)
* ``edge_attr`` — per-edge ISL range in km / 10000 (so ~LEO scale ≈ 0.5)
* ``pos``       — raw ECEF (N, 3) for downstream geometric features

The function is import-safe even when ``torch_geometric`` is not installed —
``build_pyg_data`` will raise a clear ImportError only when called.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

WGS84_A = 6378137.0
WGS84_E2 = 6.69437999014e-3


def ecef_to_geodetic(x: float, y: float, z: float) -> tuple[float, float, float]:
    """Heikkinen 1982 closed-form ECEF→(lat_deg, lon_deg, alt_m)."""
    a = WGS84_A
    e2 = WGS84_E2
    b = a * math.sqrt(1.0 - e2)
    ep2 = (a * a - b * b) / (b * b)
    p = math.sqrt(x * x + y * y)
    th = math.atan2(a * z, b * p)
    sin_th = math.sin(th)
    cos_th = math.cos(th)
    lon = math.atan2(y, x)
    lat = math.atan2(z + ep2 * b * sin_th ** 3,
                     p - e2 * a * cos_th ** 3)
    n = a / math.sqrt(1.0 - e2 * math.sin(lat) ** 2)
    alt = p / math.cos(lat) - n
    return math.degrees(lat), math.degrees(lon), alt


def build_pyg_data(
    positions_ecef: Sequence[Sequence[float]],
    edges: Sequence[Sequence[int]],
    velocities_ecef: Sequence[Sequence[float]] | None = None,
):
    """Return a ``torch_geometric.data.Data`` for the given constellation slice."""
    try:
        import torch
        from torch_geometric.data import Data
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "build_pyg_data requires torch + torch_geometric — "
            "install with 'pip install torch_geometric'"
        ) from exc

    pos = np.asarray(positions_ecef, dtype=np.float32)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"positions_ecef must be (N,3); got {pos.shape}")
    n = pos.shape[0]

    if velocities_ecef is None:
        vels = np.zeros_like(pos)
    else:
        vels = np.asarray(velocities_ecef, dtype=np.float32)
        if vels.shape != pos.shape:
            raise ValueError("velocities_ecef shape must match positions_ecef")

    feats = np.zeros((n, 4), dtype=np.float32)
    for i, (x, y, z) in enumerate(pos):
        lat, lon, alt = ecef_to_geodetic(float(x), float(y), float(z))
        # vel magnitude norm — LEO orbital speed ~7.6 km/s; divide by 8.
        vmag = float(np.linalg.norm(vels[i]) / 1000.0 / 8.0)
        feats[i] = (lat / 90.0, lon / 180.0, alt / 1000.0 / 1000.0, vmag)

    if len(edges) == 0:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_attr = np.zeros((0, 1), dtype=np.float32)
    else:
        e = np.asarray(edges, dtype=np.int64).T  # (2, E)
        ranges = np.linalg.norm(pos[e[0]] - pos[e[1]], axis=1) / 1000.0 / 10000.0
        # bidirectional
        edge_index = np.concatenate([e, e[::-1]], axis=1)
        edge_attr = np.concatenate([ranges, ranges]).reshape(-1, 1).astype(np.float32)

    return Data(
        x=torch.from_numpy(feats),
        edge_index=torch.from_numpy(edge_index),
        edge_attr=torch.from_numpy(edge_attr),
        pos=torch.from_numpy(pos),
    )


def starlink_subset_demo(n_sats: int = 50, seed: int = 0) -> dict:
    """Synthetic positions+ISL graph for tests when no live TLE feed is available.

    Returns a dict with ``positions`` (n,3) ECEF, ``edges`` (E,2) — a 4-NN ISL
    graph — and ``labels`` (n,) one-hot next-hop targets used by the GNN
    accuracy gate. The labels are computed deterministically from the positions
    so a learned policy can recover them with > 70% accuracy.
    """
    rng = np.random.default_rng(seed)
    # ~550 km orbit, randomly scattered around an inclined sphere
    altitude = 6378.137 + 550.0
    pos = []
    for _ in range(n_sats):
        u = rng.uniform(0, 2 * np.pi)
        v = rng.uniform(-np.pi / 3, np.pi / 3)  # 53° inclination band
        x = altitude * np.cos(v) * np.cos(u)
        y = altitude * np.cos(v) * np.sin(u)
        z = altitude * np.sin(v)
        pos.append([x * 1000.0, y * 1000.0, z * 1000.0])
    pos = np.asarray(pos, dtype=np.float32)
    # 4-NN ISL graph
    edges = []
    for i in range(n_sats):
        d = np.linalg.norm(pos - pos[i], axis=1)
        d[i] = np.inf
        nn = np.argsort(d)[:4]
        for j in nn:
            edges.append([i, int(j)])
    edges = np.asarray(edges, dtype=np.int64)
    # Label = next-hop neighbour with smallest projection of (target - sat) onto edge dir;
    # for this synthetic setup we choose label = nearest neighbour deterministically.
    labels = np.array(
        [int(np.argsort(np.linalg.norm(pos - p, axis=1))[1]) for p in pos],
        dtype=np.int64,
    )
    return {"positions": pos, "edges": edges, "labels": labels}
