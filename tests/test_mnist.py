from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from cp493.fedavg import classifier
from cp493.mnist import digit_records
from cp493.mnist_data import HIGH, LOW, MNISTConfig, build_splits, digit_schedule, load_archive, split_manifest


ROOT = Path(__file__).resolve().parents[1]
SMOKE = MNISTConfig.load(ROOT / "configs/mnist_smoke.toml")
SMALL = replace(SMOKE, samples_per_device=10, validation_size=20, proxy_per_digit=4, test_size=30)


def fixture():
    # Source/index tests run offline. CLI experiments use the checksum-verified archive.
    rng = np.random.default_rng(27)
    arrays = {}
    for split, per_digit in (("train", 24), ("test", 3)):
        labels = np.repeat(np.arange(10, dtype=np.uint8), per_digit)
        arrays[f"x_{split}"] = rng.integers(0, 256, size=(len(labels), 28, 28), dtype=np.uint8)
        arrays[f"y_{split}"] = labels
    return arrays


class MNISTTests(unittest.TestCase):
    def test_sequence_and_pilot_preserve_settings(self):
        pilot = MNISTConfig.load(ROOT / "configs/mnist_pilot.toml")
        self.assertEqual(replace(pilot, seeds=SMOKE.seeds, rounds=SMOKE.rounds), SMOKE)
        for cfg in (SMOKE, pilot):
            n = cfg.rounds // 3
            self.assertEqual([(s.scenario, s.condition) for s in digit_schedule(cfg.rounds)],
                             [("initial", LOW)] * n + [("new_digits", HIGH)] * n + [("return", LOW)] * n)
        with self.assertRaises(ValueError):
            digit_schedule(8)

    def test_source_indices_separation_and_reproducibility(self):
        archive = fixture()
        data, pool = build_splits(archive, SMALL, 493)
        manifest = split_manifest(data, pool)
        repeat = split_manifest(*build_splits(archive, SMALL, 493))
        other = split_manifest(*build_splits(archive, SMALL, 494))
        self.assertEqual(manifest, repeat)
        self.assertNotEqual(manifest[f"{LOW}/client_0"], other[f"{LOW}/client_0"])
        test_ids, train_ids = set(), set()
        for name, c in data.conditions.items():
            self.assertEqual(manifest[f"{name}/test"], other[f"{name}/test"])
            self.assertEqual(c.test.source_split, "test")
            self.assertTrue(test_ids.isdisjoint(c.test.indices))
            test_ids.update(c.test.indices)
            torch.testing.assert_close(c.test.tensors[0][:, 0], torch.from_numpy(
                archive["x_test"][c.test.indices]).float() / 255)
            for shard in [*c.clients, *(c.return_clients or []), c.validation]:
                self.assertEqual(shard.source_split, "train")
                self.assertTrue(train_ids.isdisjoint(shard.indices))
                train_ids.update(shard.indices)
                torch.testing.assert_close(shard.tensors[0][:, 0], torch.from_numpy(
                    archive["x_train"][shard.indices]).float() / 255)
                np.testing.assert_array_equal(shard.tensors[1].numpy(), archive["y_train"][shard.indices])
                self.assertTrue(all((label < 5) == (name == LOW) for label in shard.tensors[1]))
        self.assertEqual(test_ids, set(range(SMALL.test_size)))
        self.assertEqual(pool.source_split, "train")
        self.assertTrue(train_ids.isdisjoint(pool.indices))
        self.assertEqual(len(train_ids), SMALL.validation_size + 3 * SMALL.devices * SMALL.samples_per_device)
        self.assertEqual(len(pool), 10 * SMALL.proxy_per_digit)
        self.assertIs(data.conditions[LOW].training_shards(digit_schedule(6)[-1]),
                      data.conditions[LOW].return_clients)

    def test_practice_reconstructs_from_training_only_parents_and_reported_counts(self):
        archive = fixture()
        _, pool = build_splits(archive, SMALL, 493)
        # Deliberately mix digit groups. The generator has no task or schedule argument.
        counts = torch.tensor([0, 0, 7, 0, 0, 0, 0, 23, 0, 0])
        batch, trace = pool.generate(counts, 493, 1, 64)
        repeat, trace_again = pool.generate(counts, 493, 1, 64)
        self.assertEqual(trace, trace_again)
        torch.testing.assert_close(batch.tensors[0], repeat.tensors[0])
        self.assertEqual(set(batch.tensors[1].tolist()), {2, 7})
        for (pixels, label), parent, (dy, dx) in zip(batch, trace["train_indices"],
                                                     trace["translations_yx"], strict=True):
            self.assertIn(parent, pool.indices)
            self.assertEqual(label.item(), archive["y_train"][parent])
            self.assertEqual(abs(dy) + abs(dx), 1)
            target = np.roll(archive["x_train"][parent], (dy, dx), axis=(0, 1))
            if dy:
                target[0 if dy > 0 else -1] = 0
            if dx:
                target[:, 0 if dx > 0 else -1] = 0
            torch.testing.assert_close(pixels[0], torch.from_numpy(target).float() / 255)

    def test_combined_practice_is_test_independent_and_memory_is_client_only(self):
        data, pool = build_splits(fixture(), SMALL, 493)
        torch.manual_seed(493)
        model = classifier(784, 10, SMALL.hidden_size)
        # Exercise both digit groups and return regardless of fixture validation accuracy.
        with patch("cp493.experiment.validation_dropped", side_effect=lambda ref, *_: ref is not None):
            first = list(digit_records(SMALL, "combined", 493, data, pool, model))
            for condition in data.conditions.values():
                labels = condition.test.tensors[1]
                labels.copy_((labels + 1) % 10)
            second = list(digit_records(SMALL, "combined", 493, data, pool, model))
        allowed = set(pool.indices.tolist())
        for step, a, b in zip(digit_schedule(6), first, second, strict=True):
            for key in ("current_test_accuracy", "original_test_accuracy", "test_accuracy_by_condition"):
                a.pop(key)
                b.pop(key)
            self.assertEqual(a, b)
            if a["adaptation_triggered"]:
                trace = a["practice_provenance"]
                self.assertTrue(set(trace["train_indices"]) <= allowed)
                clients = data.conditions[step.condition].training_shards(step)
                counts = sum(torch.bincount(c.tensors[1], minlength=10) for c in clients)
                self.assertEqual(trace["client_label_counts"], counts.tolist())
                self.assertEqual(a["adaptation_examples"], SMALL.practice_examples)
        self.assertEqual(first[-1]["memory_candidates_cumulative"], 3 * SMALL.devices * SMALL.samples_per_device)

    def test_wrong_archive_fails_before_reading_images(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "mnist.npz"
            path.write_bytes(b"incorrect archive")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_archive(path)
