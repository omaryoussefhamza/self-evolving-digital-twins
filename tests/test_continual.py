from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import Mock

import torch
from torch import nn
from torch.utils.data import TensorDataset

from cp493.continual import DeviceMemory, ReplayRound
from cp493.data import synthetic_data
from cp493.experiment import Config, records
from cp493.fedavg import train_round
from cp493.scenario import schedule


SMOKE = Config.load(Path(__file__).resolve().parents[1] / "configs/smoke.toml")


class ContinualTests(unittest.TestCase):
    def test_capacity_no_repeat_admission_and_independent_storage(self):
        memory = DeviceMemory(3, 493, 0)
        features = torch.arange(30.).reshape(-1, 1)
        shard = TensorDataset(features, torch.zeros(30, dtype=torch.long))
        stable, shifted = schedule(8)[0], schedule(8)[2]
        memory.remember(stable, shard)
        memory.remember(stable, shard)
        features.fill_(-1)
        self.assertEqual(memory.seen_examples, 30)
        self.assertEqual(len(memory.entries), 3)
        self.assertTrue(all(x.item() >= 0 for _, x, _ in memory.entries))
        self.assertEqual(memory.sample(stable, 4), [])
        self.assertEqual(sorted(memory.sample(shifted, 4)), [0, 1, 2])
        memory.remember(shifted, shard)
        self.assertEqual(memory.seen_examples, 60)
        self.assertEqual(len(memory.entries), 3)

    def test_only_device_training_rows_enter_memory(self):
        steps = schedule(SMOKE.rounds)
        conditions = synthetic_data(steps, 493, 3, 64, 96, 256).conditions
        allowed = [set() for _ in range(3)]
        round_iter = iter(steps)

        def inspect(model, clients, *, replay, **training):
            step = next(round_iter)

            def check_entries():
                for memory, local_rows in zip(replay.memories, allowed, strict=True):
                    for source, features, label in memory.entries:
                        self.assertIn((source, tuple(features.tolist()), label.item()), local_rows)

            check_entries()  # On entry, only data from completed rounds is eligible.
            starting_weights = {k: v.clone() for k, v in model.state_dict().items()}
            loss = train_round(model, clients, replay=replay, **training)
            for key, weights in replay.teacher.state_dict().items():
                torch.testing.assert_close(weights, starting_weights[key], rtol=0, atol=0)
            for device, shard in enumerate(conditions[step.condition].training_shards(step)):
                allowed[device].update((step, tuple(x.tolist()), y.item()) for x, y in shard)
            check_entries()
            return loss

        practice = Mock(side_effect=AssertionError("practice generation during continual learning"))
        rows = list(records(SMOKE, method="continual", round_method=inspect, practice_factory=practice))
        practice.assert_not_called()
        self.assertTrue(all(r["memory_size_by_device"] == [16, 16, 16] for r in rows))

    def test_teacher_is_frozen_previous_server_and_guides_student(self):
        memory = DeviceMemory(1, 493, 0)
        stable, shifted = schedule(8)[0], schedule(8)[2]
        memory.remember(stable, TensorDataset(torch.ones(1, 1), torch.tensor([0])))
        server = nn.Linear(1, 2, bias=False)
        with torch.no_grad():
            server.weight.copy_(torch.tensor([[2.], [-2.]]))
        replay = ReplayRound(server, [memory], shifted, 1, 0.5)
        teacher = replay.teacher
        snapshot = server.weight.detach().clone()
        # Move the student away from its starting weights as in a later minibatch.
        with torch.no_grad():
            server.weight.zero_()
        current_ce = nn.functional.cross_entropy(server(torch.ones(1, 1)), torch.tensor([1]))
        loss = replay.local_loss(0, server, current_ce, 1)
        loss.backward()
        # Opposite hard labels cancel at equal logits. Only KL drives this gradient.
        target = snapshot.squeeze(1).softmax(0)
        torch.testing.assert_close(server.weight.grad[:, 0], 0.5 * (0.5 - target))
        self.assertFalse(teacher.training)
        self.assertFalse(teacher.weight.requires_grad)
        self.assertIsNone(teacher.weight.grad)
        torch.testing.assert_close(teacher.weight, snapshot)

    def test_reproducibility_and_work(self):
        rows = list(records(SMOKE, method="continual"))
        self.assertEqual(rows, list(records(SMOKE, method="continual")))
        for row in rows:
            self.assertEqual((row["ordinary_examples"], row["ordinary_steps"]), (192, 12))
            self.assertEqual(row["extra_optimizer_steps"], 0)
            self.assertEqual(row["memory_capacity_per_device"], 16)
            expected = 0 if row["round"] <= 2 else 48
            self.assertEqual(row["replay_presentations"], expected)
            self.assertEqual(row["teacher_presentations"], expected)
            self.assertLessEqual(row["distinct_replayed_examples"], expected)
            self.assertEqual(row["distinct_replayed_examples"] > 0, expected > 0)
        # With replay disabled, the shared initialization, data, shuffle and SGD
        # must reproduce the baseline throughout the run.
        no_replay = records(replace(SMOKE, memory_capacity=0), method="continual")
        for a, b in zip(no_replay, records(SMOKE), strict=True):
            self.assertEqual(a["training_loss"], b["training_loss"])
            self.assertEqual(a["test_accuracy_by_condition"], b["test_accuracy_by_condition"])

    def test_existing_method_smoke_regression(self):
        # Correct predictions on the 256-example test sets in the saved pre-change runs.
        expected = {
            "fedavg": ([122, 140, 96, 101, 141, 168, 172, 182],
                       [122, 140, 143, 150, 156, 160, 172, 182], []),
            "self_evolution": ([122, 140, 112, 128, 160, 167, 175, 183],
                               [122, 140, 127, 124, 126, 146, 175, 183], [3, 7]),
        }
        for method, (current, original, triggers) in expected.items():
            with self.subTest(method=method):
                rows = list(records(SMOKE, method=method))
                self.assertEqual([r["current_test_accuracy"] * 256 for r in rows], current)
                self.assertEqual([r["test_accuracy_by_condition"]["original"] * 256 for r in rows], original)
                self.assertEqual([r["round"] for r in rows if r["adaptation_triggered"]], triggers)
