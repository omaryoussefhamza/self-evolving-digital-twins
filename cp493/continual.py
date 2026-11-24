"""Client-local exemplar replay and prediction distillation."""

from copy import deepcopy
import random

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset

from .scenario import Step


class DeviceMemory:
    def __init__(self, capacity: int, seed: int, device: int):
        self.capacity = capacity
        self.rng = random.Random(f"{seed}/replay/{device}")
        self.entries: list[tuple[Step, torch.Tensor, torch.Tensor]] = []
        self.seen_examples = 0
        self.last_shard = None

    def remember(self, shard: Step, training: TensorDataset):
        # Consecutive rounds of one stage use the same shard. Admit it once.
        if shard == self.last_shard:
            return
        self.last_shard = shard
        for features, label in training:
            self.seen_examples += 1
            if len(self.entries) < self.capacity:
                slot = len(self.entries)
            else:
                slot = self.rng.randrange(self.seen_examples)
                if slot >= self.capacity:
                    continue
            # Clone rows so small exemplars never retain entire shard storage.
            entry = (shard, features.detach().clone(), label.detach().clone())
            if slot == len(self.entries):
                self.entries.append(entry)
            else:
                self.entries[slot] = entry

    def sample(self, current: Step, limit: int) -> list[int]:
        eligible = [i for i, (source, _, _) in enumerate(self.entries) if source != current]
        return self.rng.sample(eligible, min(limit, len(eligible)))


class ReplayRound:
    def __init__(self, server: nn.Module, memories: list[DeviceMemory], shard: Step,
                 replay_per_batch: int, distillation_weight: float):
        self.teacher = deepcopy(server).eval().requires_grad_(False)
        self.memories = memories
        self.shard = shard
        self.replay_per_batch = replay_per_batch
        self.weight = distillation_weight
        self.unique_slots = [set() for _ in memories]
        self.presentations = 0
        self.teacher_presentations = 0

    def local_loss(self, device: int, student: nn.Module, current_ce: torch.Tensor,
                   current_count: int) -> torch.Tensor:
        memory = self.memories[device]
        slots = memory.sample(self.shard, self.replay_per_batch)
        if not slots:
            return current_ce
        self.unique_slots[device].update(slots)
        self.presentations += len(slots)
        features = torch.stack([memory.entries[i][1] for i in slots])
        labels = torch.stack([memory.entries[i][2] for i in slots])
        logits = student(features)
        replay_ce = F.cross_entropy(logits, labels, reduction="sum")
        supervised = (current_count * current_ce + replay_ce) / (current_count + len(slots))
        if self.weight == 0:
            return supervised
        with torch.no_grad():
            targets = self.teacher(features).softmax(dim=1)
        self.teacher_presentations += len(slots)
        distillation = F.kl_div(logits.log_softmax(dim=1), targets, reduction="batchmean")
        return supervised + self.weight * distillation

    def metrics(self) -> dict:
        return {
            "memory_capacity_per_device": self.memories[0].capacity,
            "memory_size_by_device": [len(m.entries) for m in self.memories],
            "distinct_replayed_examples": sum(len(s) for s in self.unique_slots),
            "replay_presentations": self.presentations,
            "teacher_presentations": self.teacher_presentations,
            "extra_optimizer_steps": 0,
            "distillation_weight": self.weight,
        }
