"""BiFPN-lite modules for Ultralytics model YAMLs.

Ultralytics only gives special channel inference to the built-in ``Concat``
class. ``WeightedConcat`` is therefore installed with a small parse-time patch:
the YAML can name ``WeightedConcat`` explicitly, while ``parse_model()`` still
uses the same channel rule as ``Concat``.
"""
from __future__ import annotations

import torch
from torch import nn


class WeightedConcat(nn.Module):
    """Concat feature maps after learnable non-negative per-input weighting."""

    def __init__(self, dimension: int = 1, n_inputs: int = 2, eps: float = 1e-4):
        super().__init__()
        if n_inputs < 2:
            raise ValueError(f"WeightedConcat expects at least 2 inputs, got {n_inputs}")
        self.d = dimension
        self.eps = eps
        self.weights = nn.Parameter(torch.ones(n_inputs, dtype=torch.float32))

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        if len(x) != self.weights.numel():
            raise ValueError(f"WeightedConcat expected {self.weights.numel()} inputs, got {len(x)}")
        weights = torch.relu(self.weights)
        weights = weights / (weights.sum() + self.eps)
        return torch.cat([xi * wi for xi, wi in zip(x, weights)], self.d)


def install_bifpn_patch() -> None:
    """Make Ultralytics parse ``WeightedConcat`` with Concat channel semantics."""
    from ultralytics.nn import tasks as _tasks

    _tasks.__dict__["WeightedConcat"] = WeightedConcat
    # parse_model checks ``m is Concat`` before summing channels from list inputs.
    # Pointing the global Concat symbol at WeightedConcat makes explicit
    # WeightedConcat YAML layers use that same branch in this training process.
    _tasks.__dict__["Concat"] = WeightedConcat
