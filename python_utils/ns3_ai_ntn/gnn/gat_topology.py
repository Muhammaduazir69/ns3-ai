"""Graph Attention model for next-hop prediction over an ISL graph.

The model is intentionally tiny (two GATConv layers) so the W4 unit test can
forward-pass on a 50-node graph in well under a second on CPU. It also
includes a self-attention fallback path that runs without ``torch_geometric``
installed — useful for the bare CI machine where only ``torch`` is present.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _DenseAttentionBlock(nn.Module):
    """Plain dense self-attention used when torch_geometric is unavailable."""

    def __init__(self, in_dim: int, out_dim: int, heads: int = 2):
        super().__init__()
        self.q = nn.Linear(in_dim, out_dim * heads)
        self.k = nn.Linear(in_dim, out_dim * heads)
        self.v = nn.Linear(in_dim, out_dim * heads)
        self.heads = heads
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        n = x.size(0)
        q = self.q(x).view(n, self.heads, self.out_dim)
        k = self.k(x).view(n, self.heads, self.out_dim)
        v = self.v(x).view(n, self.heads, self.out_dim)
        # (n, n, heads)
        att = torch.einsum("nhd,mhd->nmh", q, k) / (self.out_dim ** 0.5)
        if mask is not None:
            att = att.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        att = F.softmax(att, dim=1)
        out = torch.einsum("nmh,mhd->nhd", att, v)
        return out.reshape(n, self.heads * self.out_dim)


class GATTopology(nn.Module):
    """Two-layer GAT (or dense-attention fallback) with a per-node classifier head."""

    def __init__(self, in_dim: int, hidden: int = 16, n_classes: int = 50,
                 heads: int = 4):
        super().__init__()
        try:
            from torch_geometric.nn import GATConv
            self._has_pyg = True
            self.g1 = GATConv(in_dim, hidden, heads=heads)
            self.g2 = GATConv(hidden * heads, hidden, heads=1)
        except ImportError:
            self._has_pyg = False
            self.g1 = _DenseAttentionBlock(in_dim, hidden, heads=heads)
            self.g2 = _DenseAttentionBlock(hidden * heads, hidden, heads=1)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, data) -> torch.Tensor:
        x = data.x
        if self._has_pyg:
            ei = data.edge_index
            h = F.elu(self.g1(x, ei))
            h = F.elu(self.g2(h, ei))
        else:
            mask = self._mask_from_edge_index(data.edge_index, x.size(0))
            h = F.elu(self.g1(x, mask=mask))
            h = F.elu(self.g2(h, mask=mask))
        return self.head(h)

    @staticmethod
    def _mask_from_edge_index(edge_index: torch.Tensor, n: int) -> torch.Tensor:
        m = torch.eye(n, dtype=torch.bool, device=edge_index.device)
        for i in range(edge_index.size(1)):
            a = int(edge_index[0, i].item())
            b = int(edge_index[1, i].item())
            m[a, b] = True
            m[b, a] = True
        return m


def train_quick(
    data,
    labels: torch.Tensor,
    epochs: int = 200,
    lr: float = 5e-3,
    n_classes: int | None = None,
    val_frac: float = 0.3,
    seed: int = 0,
) -> tuple[GATTopology, float]:
    """Train the model and return ``(model, VALIDATION accuracy)``.

    AI-08: this used to step the optimizer on every node and then score the
    model on the very logits that step had produced, over those same nodes,
    with ``n_classes = n``. That is a memorisation check: each of the 50 nodes
    needed only to recall its own index, and the 70% gate could not fail for any
    reason connected to learning constellation topology.

    Now the nodes are split into train and validation masks, the loss is taken
    on the training nodes ONLY, and the accuracy returned is the one measured on
    nodes the optimizer never saw. ``n_classes`` defaults to the number of
    distinct labels rather than the number of nodes, which is what makes the
    task node-independent.

    The training accuracy is available as ``model.last_train_acc`` for anyone
    who wants to see the gap.
    """
    n = labels.numel()
    if n_classes is None:
        n_classes = int(labels.max().item()) + 1
    # AI-08: seed the GLOBAL torch RNG before constructing the model.
    #
    # The split below uses its own generator and was already deterministic, but
    # the weight initialisation draws from the global RNG, so the reported
    # accuracy depended on which other tests had run first. An accuracy gate
    # whose value moves with test ORDER is not a gate: this test passed in
    # isolation and failed in the full suite for that reason alone.
    torch.manual_seed(seed)
    model = GATTopology(in_dim=int(data.x.size(1)),
                        hidden=24, n_classes=int(n_classes), heads=4)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    # Deterministic split, so the gate does not move between runs.
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_val = max(1, int(round(val_frac * n)))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    train_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask = torch.zeros(n, dtype=torch.bool)
    val_mask[val_idx] = True

    last_val = 0.0
    last_train = 0.0
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(data)
        # Loss on the TRAINING nodes only. The graph convolution still sees the
        # whole graph, which is standard transductive practice; what must not
        # leak is the validation nodes' LABELS.
        loss = loss_fn(logits[train_mask], labels[train_mask])
        loss.backward()
        opt.step()
        if ep % 20 == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                pred = model(data).argmax(dim=1)
                last_val = float((pred[val_mask] == labels[val_mask]).float().mean().item())
                last_train = float((pred[train_mask] == labels[train_mask]).float().mean().item())
    model.last_train_acc = last_train
    model.last_val_acc = last_val
    return model, last_val
