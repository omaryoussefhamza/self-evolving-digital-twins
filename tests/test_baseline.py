import json
import math
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import torch
from torch import nn
from torch.utils.data import TensorDataset

from cp493.data import synthetic_data
from cp493.fedavg import train_round, weighted_average
from cp493.scenario import schedule


ROOT = Path(__file__).resolve().parents[1]


def data_for(seed):
    return synthetic_data(schedule(8), seed, 3, 64, 96, 256).conditions


def splits(condition):
    return [*condition.clients, condition.validation, condition.test]


class BaselineTests(unittest.TestCase):
    def test_seed_and_device_partition(self):
        first, repeat, other = (data_for(seed) for seed in [493, 493, 494])
        for key in first:
            for a, b, c in zip(splits(first[key]), splits(repeat[key]), splits(other[key])):
                for x, y in zip(a.tensors, b.tensors):
                    self.assertTrue(torch.equal(x, y))
                self.assertFalse(torch.equal(a.tensors[0], c.tensors[0]))
            clients = first[key].clients
            self.assertEqual([len(client) for client in clients], [48, 64, 80])
            self.assertEqual([int(client.tensors[1].sum()) for client in clients], [17, 32, 52])

    def test_no_examples_shared_between_clients_conditions_or_splits(self):
        seen = set()
        for condition in data_for(493).values():
            for dataset in splits(condition):
                examples = {tuple(row) for row in dataset.tensors[0].tolist()}
                self.assertEqual(len(examples), len(dataset))
                self.assertTrue(seen.isdisjoint(examples))
                seen.update(examples)

    def test_schedule_returns_to_original_condition(self):
        steps = schedule(8)
        self.assertEqual([s.scenario for s in steps],
                         ["stable", "stable", "sudden", "sudden", "gradual", "gradual", "return", "return"])
        self.assertGreater(steps[3].shift, steps[4].shift)
        self.assertGreater(steps[4].shift, steps[5].shift)
        self.assertGreater(steps[5].shift, steps[6].shift)
        self.assertEqual(steps[0].condition, steps[-1].condition)
        self.assertEqual(steps[0].shift, steps[-1].shift)

    def test_fedavg_weighting(self):
        result = weighted_average([(1, {"w": torch.tensor([2., 6.])}),
                                   (3, {"w": torch.tensor([6., 2.])})])
        torch.testing.assert_close(result["w"], torch.tensor([5., 3.]))

    def test_one_round_uses_same_start_and_weights_by_examples(self):
        model = nn.Linear(1, 2, bias=False)
        nn.init.zeros_(model.weight)
        clients = [TensorDataset(torch.ones(n, 1), torch.full((n,), label))
                   for n, label in [(1, 0), (3, 1)]]
        # With equal initial logits, each local SGD update has magnitude lr / 2.
        loss = train_round(model, clients, local_epochs=1, batch_size=4,
                           learning_rate=0.2, generator=torch.Generator().manual_seed(0))
        torch.testing.assert_close(model.weight, torch.tensor([[-0.05], [0.05]]))
        self.assertAlmostEqual(loss, math.log(2), places=6)

    def test_smoke_cli(self):
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "rounds.jsonl"
            command = [sys.executable, "-m", "cp493.experiment", "--config",
                       "configs/smoke.toml", "--output", str(output)]
            subprocess.run(command, cwd=ROOT, check=True, capture_output=True)
            first = output.read_bytes()
            subprocess.run(command, cwd=ROOT, check=True, capture_output=True)
            self.assertEqual(output.read_bytes(), first)
        rows = [json.loads(line) for line in first.splitlines()]
        self.assertEqual(len(rows), 8)
        seen = set()
        for i, (row, step) in enumerate(zip(rows, schedule(8)), start=1):
            self.assertEqual((row["round"], row["seed"]), (i, 493))
            self.assertEqual((row["scenario"], row["condition"]), (step.scenario, step.condition))
            seen.add(step.condition)
            scores = row["test_accuracy_by_condition"]
            self.assertEqual(set(scores), seen)
            self.assertEqual(row["current_test_accuracy"], scores[step.condition])
            self.assertTrue(all(0 <= score <= 1 for score in scores.values()))
            self.assertTrue(math.isfinite(row["training_loss"]) and row["training_loss"] >= 0)
