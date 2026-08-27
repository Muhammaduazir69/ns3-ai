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
    neighbors: Sequence[Sequence[int]] | None = None,
    gateway_pos: Sequence[float] | None = None,
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

    # AI-08: optional per-slot routing geometry.
    #
    # The next-hop label indexes a node's ordered neighbour list, and the four
    # base features (lat, lon, alt, |v|) carry nothing about where those
    # neighbours are, so the slot index was not recoverable from them at all:
    # the task was unlearnable as posed, and the only way to score was to
    # memorise. When `neighbors` and `gateway_pos` are supplied, each node gains
    # one feature per slot, the cosine between the direction to that neighbour
    # and the direction to the gateway. That is the quantity a routing decision
    # actually turns on.
    k_slots = 0
    slot_feats = None
    if neighbors is not None and gateway_pos is not None:
        nb = np.asarray(neighbors, dtype=np.int64)
        gw = np.asarray(gateway_pos, dtype=np.float64)
        k_slots = int(nb.shape[1])
        slot_feats = np.zeros((n, k_slots), dtype=np.float32)
        for i in range(n):
            to_gw = gw - pos[i]
            ngw = np.linalg.norm(to_gw)
            for sIdx in range(k_slots):
                to_nb = pos[int(nb[i, sIdx])] - pos[i]
                nnb = np.linalg.norm(to_nb)
                if ngw < 1e-9 or nnb < 1e-9:
                    slot_feats[i, sIdx] = 0.0
                else:
                    slot_feats[i, sIdx] = float(np.dot(to_gw, to_nb) / (ngw * nnb))

    feats = np.zeros((n, 4 + k_slots), dtype=np.float32)
    for i, (x, y, z) in enumerate(pos):
        lat, lon, alt = ecef_to_geodetic(float(x), float(y), float(z))
        # vel magnitude norm — LEO orbital speed ~7.6 km/s; divide by 8.
        vmag = float(np.linalg.norm(vels[i]) / 1000.0 / 8.0)
        feats[i, :4] = (lat / 90.0, lon / 180.0, alt / 1000.0 / 1000.0, vmag)
        if slot_feats is not None:
            feats[i, 4:] = slot_feats[i]

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


def starlink_subset_demo(n_sats: int = 50, seed: int = 0, k: int = 4) -> dict:
    """Synthetic positions + ISL graph for tests when no live TLE feed exists.

    Returns ``positions`` (n,3) ECEF, ``edges`` (E,2) for a k-NN ISL graph,
    ``labels`` (n,) and ``neighbors`` (n,k).

    AI-08: the label used to be the index of the nearest neighbour, i.e. a NODE
    ID, with ``n_classes = n``. That is not a learnable task in any held-out
    sense: the label space IS the node set, so a validation node's answer is a
    class the model may never have seen, and the only way to score well is to
    memorise the training nodes. The gate that reported 70% was scoring the
    model on the very logits its last gradient step had just produced, over all
    50 nodes, with 50 classes.

    The label is now the audit's own suggestion: which of this satellite's k ISL
    neighbours lies on the shortest path to a fixed ground gateway. The class
    space is {0..k-1} and is the SAME for every node, so a held-out satellite is
    a genuine test of whether the model learned a routing rule rather than a
    lookup table.
    """
    rng = np.random.default_rng(seed)
    # ~550 km orbit, randomly scattered around an inclined band
    altitude = 6378.137 + 550.0
    pos = []
    for _ in range(n_sats):
        u = rng.uniform(0, 2 * np.pi)
        v = rng.uniform(-np.pi / 3, np.pi / 3)  # 53 degree inclination band
        x = altitude * np.cos(v) * np.cos(u)
        y = altitude * np.cos(v) * np.sin(u)
        z = altitude * np.sin(v)
        pos.append([x * 1000.0, y * 1000.0, z * 1000.0])
    pos = np.asarray(pos, dtype=np.float32)

    # k-NN ISL graph. neighbors[i] is i's ordered neighbour list, and the label
    # indexes into it, which is what makes the class space node-independent.
    neighbors = np.zeros((n_sats, k), dtype=np.int64)
    edges = []
    for i in range(n_sats):
        d = np.linalg.norm(pos - pos[i], axis=1)
        d[i] = np.inf
        nn = np.argsort(d)[:k]
        neighbors[i] = nn
        for j in nn:
            edges.append([i, int(j)])
    edges = np.asarray(edges, dtype=np.int64)

    # Shortest path to a fixed ground gateway, over the ISL graph, by Dijkstra.
    # The gateway is the satellite closest to a fixed ECEF point, so the target
    # is deterministic and independent of the RNG draw order.
    gw_point = np.array([6378137.0, 0.0, 0.0], dtype=np.float64)
    gateway = int(np.argmin(np.linalg.norm(pos - gw_point, axis=1)))

    INF = float("inf")
    dist = [INF] * n_sats
    dist[gateway] = 0.0
    visited = [False] * n_sats
    # Undirected adjacency with Euclidean weights.
    adj: list[list[tuple[int, float]]] = [[] for _ in range(n_sats)]
    for i in range(n_sats):
        for j in neighbors[i]:
            w = float(np.linalg.norm(pos[i] - pos[j]))
            adj[i].append((int(j), w))
            adj[int(j)].append((i, w))
    for _ in range(n_sats):
        u = -1
        best = INF
        for v_ in range(n_sats):
            if not visited[v_] and dist[v_] < best:
                best = dist[v_]
                u = v_
        if u < 0:
            break
        visited[u] = True
        for v_, w in adj[u]:
            if dist[u] + w < dist[v_]:
                dist[v_] = dist[u] + w

    # Next hop = the neighbour minimising dist[j] + w(i,j). Unreachable nodes
    # and the gateway itself fall back to the nearest neighbour, which is
    # label 0 by construction of the ordered neighbour list.
    labels = np.zeros(n_sats, dtype=np.int64)
    for i in range(n_sats):
        best_idx = 0
        best_cost = INF
        for slot, j in enumerate(neighbors[i]):
            j = int(j)
            if dist[j] == INF:
                continue
            c = dist[j] + float(np.linalg.norm(pos[i] - pos[j]))
            if c < best_cost:
                best_cost = c
                best_idx = slot
        labels[i] = best_idx

    return {
        "positions": pos,
        "edges": edges,
        "labels": labels,
        "neighbors": neighbors,
        "gateway": gateway,
        "n_classes": k,
    }
