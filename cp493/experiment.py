"""Run a seeded experiment and write one JSON object per completed round."""

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import tomllib

import torch
from torch.utils.data import TensorDataset

from .adaptation import adapt, validation_dropped
from .continual import DeviceMemory, ReplayRound
from .data import synthetic_data, synthetic_practice
from .fedavg import classifier, train_round
from .scenario import schedule


METHODS = ("fedavg", "self_evolution", "continual")


@dataclass(frozen=True)
class Config:
    seeds: list[int]
    rounds: int
    devices: int
    samples_per_device: int
    validation_size: int
    test_size: int
    local_epochs: int
    batch_size: int
    learning_rate: float
    validation_drop: float = 0.10
    practice_examples: int = 64
    memory_capacity: int = 16
    replay_per_batch: int = 4
    distillation_weight: float = 0.5

    def __post_init__(self):
        if not self.seeds or any(type(s) is not int or not 0 <= s < 2**63 for s in self.seeds):
            raise ValueError("Provide at least one nonnegative 63-bit integer seed.")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Seeds must be distinct.")
        for key in ("rounds", "devices", "samples_per_device", "validation_size",
                    "test_size", "local_epochs", "batch_size", "practice_examples"):
            value = getattr(self, key)
            if type(value) is not int or value < 1:
                raise ValueError(f"{key} must be a positive integer.")
        if self.learning_rate <= 0 or not math.isfinite(self.learning_rate):
            raise ValueError("learning_rate must be finite and positive.")
        if not 0 < self.validation_drop <= 1:
            raise ValueError("validation_drop must be in (0, 1].")
        for key in ("memory_capacity", "replay_per_batch"):
            value = getattr(self, key)
            if type(value) is not int or value < 0:
                raise ValueError(f"{key} must be a nonnegative integer.")
        if self.distillation_weight < 0 or not math.isfinite(self.distillation_weight):
            raise ValueError("distillation_weight must be finite and nonnegative.")

    @classmethod
    def load(cls, path: Path):
        with path.open("rb") as source:
            return cls(**tomllib.load(source))


@torch.inference_mode()
def accuracy(model: torch.nn.Module, data: TensorDataset) -> float:
    model.eval()
    features, labels = data.tensors
    return (model(features).argmax(dim=1) == labels).sum().item() / len(labels)


def records(config: Config, *, method="fedavg", dataset_factory=synthetic_data,
            round_method=train_round, model_factory=classifier, practice_factory=synthetic_practice):
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}")
    steps = schedule(config.rounds)
    for seed in config.seeds:
        data = dataset_factory(steps, seed, config.devices, config.samples_per_device,
                               config.validation_size, config.test_size)
        torch.manual_seed(seed)
        model = model_factory(data.input_size, data.classes)
        generator = torch.Generator().manual_seed(seed)
        seen = []
        reference = None
        memories = []
        if method == "continual":
            memories = [DeviceMemory(config.memory_capacity, seed, i) for i in range(config.devices)]
        for number, step in enumerate(steps, start=1):
            condition = data.conditions[step.condition]
            clients = condition.training_shards(step)
            replay = None
            training_options = dict(local_epochs=config.local_epochs, batch_size=config.batch_size,
                                    learning_rate=config.learning_rate, generator=generator)
            if memories:
                replay = ReplayRound(model, memories, step, config.replay_per_batch, config.distillation_weight)
                training_options["replay"] = replay
            loss = round_method(model, clients, **training_options)
            before = accuracy(model, condition.validation)
            triggered = method == "self_evolution" and validation_dropped(
                reference, before, config.validation_drop
            )
            after = before
            extra_examples = extra_steps = 0
            if triggered:
                practice = practice_factory(step, seed, number, config.practice_examples)
                extra_examples, extra_steps = adapt(model, practice, config.batch_size, config.learning_rate)
                after = accuracy(model, condition.validation)
            if step.condition not in seen:
                seen.append(step.condition)
            scores = {name: accuracy(model, data.conditions[name].test) for name in seen}
            yield {
                "method": method,
                "round": number,
                "scenario": step.scenario,
                "condition": step.condition,
                "seed": seed,
                "training_loss": loss,
                "ordinary_examples": config.local_epochs * sum(map(len, clients)),
                "ordinary_steps": config.local_epochs * sum(math.ceil(len(c) / config.batch_size) for c in clients),
                "validation_reference": reference,
                "validation_before_adaptation": before,
                "validation_after_adaptation": after,
                "adaptation_triggered": triggered,
                "adaptation_examples": extra_examples,
                "adaptation_steps": extra_steps,
                "current_test_accuracy": scores[step.condition],
                "test_accuracy_by_condition": scores,
                **(replay.metrics() if replay else {}),
            }
            reference = after


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--method", choices=METHODS, default="fedavg")
    args = parser.parse_args()
    config = Config.load(args.config)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as workdir:
        pending = Path(workdir) / "rounds.jsonl"
        with pending.open("w", encoding="utf-8") as output:
            for row in records(config, method=args.method):
                output.write(json.dumps(row, allow_nan=False) + "\n")
        # Only publish a complete run, including when replacing a previous result.
        pending.replace(args.output)
    print(f"Wrote {config.rounds * len(config.seeds)} rounds to {args.output}")


if __name__ == "__main__":
    main()
