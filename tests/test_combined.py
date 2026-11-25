import json
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from cp493 import experiment, fedavg
from cp493.adaptation import adapt, validation_dropped
from cp493.continual import DeviceMemory
from cp493.data import synthetic_data, synthetic_practice
from cp493.scenario import schedule


ROOT = Path(__file__).resolve().parents[1]
SMOKE = experiment.Config.load(ROOT / "configs/smoke.toml")


def snapshot(model):
    return {name: tensor.clone() for name, tensor in model.state_dict().items()}


def smoke_data():
    return synthetic_data(schedule(SMOKE.rounds), SMOKE.seeds[0], SMOKE.devices, SMOKE.samples_per_device,
                          SMOKE.validation_size, SMOKE.test_size)


class CombinedTests(unittest.TestCase):
    def test_round_order_through_real_trigger_and_no_trigger(self):
        data = smoke_data()
        events = []
        score = experiment.accuracy
        average = fedavg.weighted_average
        splits = {c.validation: ("validation", key) for key, c in data.conditions.items()}
        splits.update({c.test: ("test", key) for key, c in data.conditions.items()})

        def train(model, clients, **options):
            self.assertIn("replay", options)
            events.append("train")
            return fedavg.train_round(model, clients, **options)

        def aggregate(updates):
            events.append("aggregate")
            return average(updates)

        def evaluate(model, dataset):
            events.append(splits[dataset])
            return score(model, dataset)

        def trigger(*args):
            events.append("trigger")
            return validation_dropped(*args)

        def practice(*args):
            events.append("generate")
            return synthetic_practice(*args)

        def update(*args):
            events.append("adapt")
            return adapt(*args)

        run = experiment.records(SMOKE, method="combined", round_method=train,
                                 dataset_factory=lambda *args: data, practice_factory=practice)
        encountered = []
        triggers = []
        with (patch.object(fedavg, "weighted_average", aggregate),
              patch.object(experiment, "accuracy", evaluate),
              patch.object(experiment, "validation_dropped", trigger),
              patch.object(experiment, "adapt", update)):
            for row in run:
                current = row["condition"]
                if current not in encountered:
                    encountered.append(current)
                expected = ["train", "aggregate", ("validation", current), "trigger"]
                if row["adaptation_triggered"]:
                    expected.extend(["generate", "adapt", ("validation", current)])
                    triggers.append(row["round"])
                expected.extend(("test", name) for name in encountered)
                self.assertEqual(events, expected)
                events.clear()
        self.assertEqual(triggers, [3])

    def test_only_local_training_shards_enter_memory_even_on_practice_rounds(self):
        data = smoke_data()
        admissions = []
        remember = DeviceMemory.remember

        def audit(memory, step, dataset):
            admissions.append((memory, step, dataset))
            return remember(memory, step, dataset)

        run = experiment.records(SMOKE, method="combined", dataset_factory=lambda *args: data)
        with patch.object(DeviceMemory, "remember", audit):
            memories = []
            allowed = [set() for _ in range(SMOKE.devices)]
            for row, step in zip(run, schedule(SMOKE.rounds), strict=True):
                clients = data.conditions[step.condition].training_shards(step)
                if not memories:
                    memories = [m for m, _, _ in admissions]
                    self.assertEqual(len(set(memories)), SMOKE.devices)
                self.assertEqual(admissions, [(m, step, shard) for m, shard in zip(memories, clients)])
                for device, shard in enumerate(clients):
                    allowed[device].update((step, tuple(x.tolist()), y.item()) for x, y in shard)
                    for origin, x, y in memories[device].entries:
                        self.assertIn((origin, tuple(x.tolist()), y.item()), allowed[device])
                self.assertEqual(row["memory_size_by_device"], [SMOKE.memory_capacity] * SMOKE.devices)
                admissions.clear()

    def test_teacher_stays_frozen_through_practice_and_refreshes_next_round(self):
        observations = []

        def train(model, clients, *, replay, **options):
            observations.append((snapshot(model), replay.teacher, model))
            return fedavg.train_round(model, clients, replay=replay, **options)

        last_final = None
        for row in experiment.records(SMOKE, method="combined", round_method=train):
            start, teacher, server = observations.pop()
            if last_final is not None:
                torch.testing.assert_close(start, last_final, rtol=0, atol=0)
            torch.testing.assert_close(teacher.state_dict(), start, rtol=0, atol=0)
            self.assertFalse(teacher.training)
            self.assertTrue(all(not p.requires_grad and p.grad is None for p in teacher.parameters()))
            last_final = snapshot(server)
            if row["adaptation_triggered"]:
                self.assertTrue(any(not torch.equal(v, last_final[k]) for k, v in start.items()))

    def test_reproducibility_and_work(self):
        rows = list(experiment.records(SMOKE, method="combined"))
        self.assertEqual(rows, list(experiment.records(SMOKE, method="combined")))
        for row in rows:
            replay = 48 if row["round"] >= 3 else 0
            triggered = row["adaptation_triggered"]
            self.assertEqual((row["ordinary_examples"], row["ordinary_steps"]), (192, 12))
            self.assertEqual(row["adaptation_examples"], 64 if triggered else 0)
            self.assertEqual(row["adaptation_steps"], 4 if triggered else 0)
            self.assertEqual(row["extra_optimizer_steps"], row["adaptation_steps"])
            self.assertEqual(row["extra_training_presentations"], replay + row["adaptation_examples"])
            self.assertEqual(row["extra_validation_presentations"], 96 if triggered else 0)
            self.assertEqual((row["replay_presentations"], row["teacher_presentations"]), (replay, replay))
            self.assertEqual((row["replay_forward_passes"], row["teacher_forward_passes"]),
                             (replay // 4, replay // 4))

    def test_first_three_methods_match_checked_in_smoke_outputs(self):
        for method in ("fedavg", "self_evolution", "continual"):
            with self.subTest(method=method):
                saved = (ROOT / "results" / f"{method}.jsonl").read_text().splitlines()
                run = experiment.records(SMOKE, method=method)
                for row, line in zip(run, saved, strict=True):
                    reference = json.loads(line)
                    # Allow small platform-dependent SGD loss rounding.
                    self.assertAlmostEqual(row.pop("training_loss"), reference.pop("training_loss"), places=6)
                    self.assertEqual(row, reference)
