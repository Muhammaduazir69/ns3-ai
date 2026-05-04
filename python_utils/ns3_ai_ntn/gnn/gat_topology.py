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
) -> tuple[GATTopology, float]:
    """Train the model end-to-end and return (model, final_accuracy)."""
    n = labels.numel()
    model = GATTopology(in_dim=int(data.x.size(1)),
                        hidden=24, n_classes=n, heads=4)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    last_acc = 0.0
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(data)
        loss = loss_fn(logits, labels)
        loss.backward()
        opt.step()
        if ep % 20 == 0 or ep == epochs - 1:
            with torch.no_grad():
                pred = logits.argmax(dim=1)
                acc = float((pred == labels).float().mean().item())
                last_acc = acc
    return model, last_acc
