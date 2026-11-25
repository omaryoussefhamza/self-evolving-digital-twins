"""Official MNIST splits, device shards and training-only server practice."""

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import pad
from torch.utils.data import TensorDataset

from .data import ConditionData, ExperimentData
from .experiment import Config
from .scenario import Step


ARCHIVE_SHA256 = "731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1"
LOW, HIGH = "digits_0_4", "digits_5_9"


def digit_schedule(rounds):
    if rounds < 3 or rounds % 3:
        raise ValueError("MNIST requires three equal, nonempty stages.")
    return [step for step in (Step("initial", LOW), Step("new_digits", HIGH), Step("return", LOW))
            for _ in range(rounds // 3)]


@dataclass(frozen=True)
class MNISTConfig(Config):
    proxy_per_digit: int = 32
    hidden_size: int = 64

    def __post_init__(self):
        super().__post_init__()
        digit_schedule(self.rounds)
        if self.validation_size % 10 or (self.devices * self.samples_per_device) % 5:
            raise ValueError("Validation and stage training totals must divide evenly across their digits.")
        for value in (self.proxy_per_digit, self.hidden_size):
            if type(value) is not int or value < 1:
                raise ValueError("Proxy pool size per digit and hidden width must be positive integers.")


def load_archive(path: Path):
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != ARCHIVE_SHA256:
            raise ValueError("MNIST checksum differs from the published Keras archive.")
        stream.seek(0)
        with np.load(stream, allow_pickle=False) as archive:
            return {name: archive[name] for name in ("x_train", "y_train", "x_test", "y_test")}


class SourceImages(TensorDataset):
    """A role-specific subset, with row IDs in its official source split."""

    def __init__(self, archive, split, indices):
        self.source_split = split
        self.indices = np.asarray(indices, dtype=np.int64)
        pixels = torch.from_numpy(archive[f"x_{split}"][self.indices]).unsqueeze(1).float().div_(255)
        labels = torch.from_numpy(archive[f"y_{split}"][self.indices]).long()
        super().__init__(pixels, labels)


class PracticePool(SourceImages):
    def __init__(self, archive, train_indices):
        super().__init__(archive, "train", train_indices)

    def generate(self, label_counts, seed, round_number, count):
        """Select labeled proxy parents, then translate with zero-filled borders."""
        rng = np.random.default_rng([seed, round_number, 1])
        pixels, labels = self.tensors
        counts = label_counts.numpy()
        chosen_labels = rng.choice(10, size=count, p=counts / counts.sum())
        parents = np.empty(count, dtype=np.int64)
        for digit in np.unique(chosen_labels):
            mask = chosen_labels == digit
            parents[mask] = rng.choice(np.flatnonzero(labels.numpy() == digit), size=mask.sum())
        offsets = np.array([(-1, 0), (1, 0), (0, -1), (0, 1)])
        directions = rng.integers(len(offsets), size=count)
        padded = pad(pixels[parents], (1, 1, 1, 1))
        images = torch.empty_like(pixels[parents])
        for direction, (dy, dx) in enumerate(offsets):
            selected = directions == direction
            images[selected] = padded[selected, :, 1-dy:29-dy, 1-dx:29-dx]
        provenance = {
            "source": "server_proxy_from_official_train",
            "train_indices": self.indices[parents].tolist(),
            "translations_yx": offsets[directions].tolist(),
            "client_label_counts": counts.tolist(),
        }
        return TensorDataset(images, labels[parents]), provenance


def build_splits(archive, config, seed):
    if len(archive["y_test"]) != config.test_size:
        raise ValueError("MNIST evaluates the entire official test split.")
    rng = np.random.default_rng([seed, 0])
    val_count = config.validation_size // 10
    proxy_end = val_count + config.proxy_per_digit
    per_digit = config.devices * config.samples_per_device // 5
    validation, proxy, initial, returning = [], [], [], []
    for digit in range(10):
        indices = rng.permutation(np.flatnonzero(archive["y_train"] == digit))
        end = proxy_end + per_digit
        if len(indices) < end + (per_digit if digit < 5 else 0):
            raise ValueError(f"Insufficient training images for digit {digit}.")
        validation.append(indices[:val_count])
        proxy.append(indices[val_count:proxy_end])
        initial.append(indices[proxy_end:end])
        if digit < 5:
            returning.append(indices[end:end + per_digit])

    def partition(groups):
        shuffled = rng.permutation(np.concatenate(groups))
        return [SourceImages(archive, "train", shard) for shard in np.split(shuffled, config.devices)]

    conditions = {}
    for name, start in ((LOW, 0), (HIGH, 5)):
        digits = slice(start, start + 5)
        test_ids = np.flatnonzero((archive["y_test"] >= start) & (archive["y_test"] < start + 5))
        conditions[name] = ConditionData(
            partition(initial[digits]),
            SourceImages(archive, "train", np.concatenate(validation[digits])),
            SourceImages(archive, "test", test_ids),
            partition(returning) if name == LOW else None,
        )
    return ExperimentData(conditions, input_size=784, classes=10), PracticePool(archive, np.concatenate(proxy))


def split_manifest(data, pool):
    """Audit intersections before training. IDs are scoped to the official split."""
    subsets = {"server_proxy": pool}
    for name, c in data.conditions.items():
        subsets[f"{name}/validation"] = c.validation
        subsets[f"{name}/test"] = c.test
        for stage, clients in ((name, c.clients), ("return", c.return_clients or [])):
            for i, shard in enumerate(clients):
                subsets[f"{stage}/client_{i}"] = shard
    seen = {"train": set(), "test": set()}
    manifest = {}
    for name, subset in subsets.items():
        ids = set(subset.indices.tolist())
        previous = seen[subset.source_split]
        if len(ids) != len(subset) or not ids.isdisjoint(previous):
            raise ValueError(f"Overlapping MNIST source indices in {name}.")
        previous.update(ids)
        entry = {
            "source_split": subset.source_split, "size": len(subset),
            "digit_counts": torch.bincount(subset.tensors[1], minlength=10).tolist(),
            "ordered_index_sha256": hashlib.sha256(subset.indices.astype("<i8").tobytes()).hexdigest(),
        }
        if subset.source_split == "train":
            entry["indices"] = subset.indices.tolist()
        manifest[name] = entry
    return manifest
