"""Full-participation FedAvg with fresh client-side SGD each round."""

from copy import deepcopy

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .continual import ReplayRound


def classifier(input_size: int, classes: int, hidden_size: int = 8) -> nn.Module:
    return nn.Sequential(nn.Flatten(), nn.Linear(input_size, hidden_size), nn.ReLU(),
                         nn.Linear(hidden_size, classes))


def weighted_average(updates: list[tuple[int, dict[str, torch.Tensor]]]) -> dict[str, torch.Tensor]:
    """Average floating-point model states by local dataset size."""
    if not updates or any(count <= 0 for count, _ in updates):
        raise ValueError("FedAvg needs a positive example count for each client.")
    total = sum(count for count, _ in updates)
    return {
        name: sum(state[name] * (count / total) for count, state in updates)
        for name in updates[0][1]
    }


def train_round(model: nn.Module, clients: list[TensorDataset], *, local_epochs: int,
                batch_size: int, learning_rate: float, generator: torch.Generator,
                replay: ReplayRound | None = None) -> float:
    updates = []
    loss_sum = 0.0
    presentations = 0
    for device, dataset in enumerate(clients):
        local = deepcopy(model)
        local.train()
        optimizer = torch.optim.SGD(local.parameters(), lr=learning_rate)
        batches = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
        for _ in range(local_epochs):
            for features, labels in batches:
                optimizer.zero_grad()
                loss = nn.functional.cross_entropy(local(features), labels)
                objective = loss if replay is None else replay.local_loss(device, local, loss, len(labels))
                objective.backward()
                optimizer.step()
                loss_sum += loss.item() * len(labels)
                presentations += len(labels)
        updates.append((len(dataset), local.state_dict()))
        if replay is not None:
            replay.memories[device].remember(replay.shard, dataset)
    model.load_state_dict(weighted_average(updates))
    return loss_sum / presentations
