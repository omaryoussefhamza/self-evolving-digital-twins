"""Validation-drop trigger and server-side practice on synthetic examples."""

from math import isclose

import torch
from torch import nn
from torch.utils.data import TensorDataset


def validation_dropped(reference: float | None, current: float, threshold: float) -> bool:
    if reference is None or current >= reference:
        return False
    drop = reference - current
    return drop >= threshold or isclose(drop, threshold, rel_tol=0, abs_tol=1e-15)


def adapt(model: nn.Module, practice: TensorDataset, batch_size: int,
          learning_rate: float) -> tuple[int, int]:
    """Take one SGD pass through the already shuffled practice set."""
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    features, labels = practice.tensors
    examples = steps = 0
    for batch, targets in zip(features.split(batch_size), labels.split(batch_size)):
        optimizer.zero_grad()
        nn.functional.cross_entropy(model(batch), targets).backward()
        optimizer.step()
        examples += len(targets)
        steps += 1
    return examples, steps
