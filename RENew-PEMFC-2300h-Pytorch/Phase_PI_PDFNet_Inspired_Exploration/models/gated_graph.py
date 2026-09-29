"""PDFNet Eq. (10)–(11) gated dynamic graph learner."""
from __future__ import annotations

import torch
from torch import nn


class GatedDynamicGraph(nn.Module):
    def __init__(self, variables: int, dimension: int) -> None:
        super().__init__()
        self.variables, self.dimension = variables, dimension
        self.W_0 = nn.Parameter(torch.empty(2 * dimension, 1))
        self.W_1 = nn.Parameter(torch.empty(2 * dimension, 1))
        nn.init.xavier_uniform_(self.W_0)
        nn.init.xavier_uniform_(self.W_1)

    def forward(self, base_graph: torch.Tensor, value: torch.Tensor, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        expected_nodes, expected_features = self.variables, self.dimension
        if base_graph.ndim not in {2, 3} or tuple(base_graph.shape[-2:]) != (expected_nodes, expected_nodes):
            raise ValueError("base graph shape violation")
        if value.ndim not in {2, 3} or tuple(value.shape[-2:]) != (expected_nodes, expected_features) or value.shape != hidden.shape:
            raise ValueError("graph value/hidden shape violation")
        z0, z1 = base_graph @ value, base_graph.transpose(-2, -1) @ value
        tau0 = torch.relu(torch.tanh(torch.cat([z0, hidden], dim=-1) @ self.W_0))
        tau1 = torch.relu(torch.tanh(torch.cat([z1, hidden], dim=-1) @ self.W_1))
        e0, e1 = z0 + tau0 * hidden, z1 + tau1 * hidden
        adjacency = torch.softmax(torch.relu(e0 @ e1.transpose(-2, -1)), dim=-1)
        return adjacency, tau0, tau1
