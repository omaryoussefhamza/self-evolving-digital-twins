"""Independent condition-specific splits for a synthetic classification task."""

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import TensorDataset

from .scenario import Step


@dataclass
class ConditionData:
    clients: list[TensorDataset]
    validation: TensorDataset
    test: TensorDataset
    return_clients: list[TensorDataset] | None = None

    def training_shards(self, step: Step) -> list[TensorDataset]:
        if step.scenario == "return" and self.return_clients is not None:
            return self.return_clients
        return self.clients


@dataclass
class ExperimentData:
    conditions: dict[str, ConditionData]
    input_size: int
    classes: int


def _examples(rng, count: int, prevalence: float, shift: float) -> TensorDataset:
    # Stratify labels so small clients retain their specified class mix.
    labels = np.zeros(count, dtype=np.int64)
    labels[:round(prevalence * count)] = 1
    rng.shuffle(labels)
    angle = np.deg2rad(120 * shift)
    direction = np.array([np.cos(angle), np.sin(angle), 0, 0])
    features = rng.normal(size=(count, 4)) + (2 * labels[:, None] - 1) * direction
    return TensorDataset(torch.from_numpy(features.astype(np.float32)), torch.from_numpy(labels))


def _client_shards(seed, index, split, devices, samples_per_device, shift):
    # Device identities (size and class prior) stay constant through all stages.
    positions = np.linspace(-1, 1, devices) if devices > 1 else [0.0]
    clients = []
    for device, position in enumerate(positions):
        count = round(samples_per_device * (1 + 0.25 * position))
        rng = np.random.default_rng([seed, index, split, device])
        clients.append(_examples(rng, count, 0.5 + 0.15 * position, shift))
    return clients


def synthetic_practice(step: Step, seed: int, round_number: int, count: int) -> TensorDataset:
    rng = np.random.default_rng([seed, round_number, 4])
    return _examples(rng, count, 0.5, step.shift)


def synthetic_data(steps: list[Step], seed: int, devices: int, samples_per_device: int,
                   validation_size: int, test_size: int) -> ExperimentData:
    conditions = {}
    returning = {step.condition for step in steps if step.scenario == "return"}
    for step in steps:
        if step.condition in conditions:
            continue
        index = len(conditions)
        validation_rng = np.random.default_rng([seed, index, 1])
        test_rng = np.random.default_rng([seed, index, 2])
        conditions[step.condition] = ConditionData(
            _client_shards(seed, index, 0, devices, samples_per_device, step.shift),
            _examples(validation_rng, validation_size, 0.5, step.shift),
            _examples(test_rng, test_size, 0.5, step.shift),
            _client_shards(seed, index, 3, devices, samples_per_device, step.shift)
            if step.condition in returning else None,
        )
    return ExperimentData(conditions, input_size=4, classes=2)
