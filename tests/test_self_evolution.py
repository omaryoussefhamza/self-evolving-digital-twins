from pathlib import Path
import unittest

import torch

from cp493.adaptation import adapt, validation_dropped
from cp493.data import synthetic_data, synthetic_practice
from cp493.experiment import Config, records
from cp493.fedavg import classifier, train_round
from cp493.scenario import schedule


SMOKE = Config.load(Path(__file__).resolve().parents[1] / "configs/smoke.toml")


def contents(dataset):
    return tuple(t.numpy().tobytes() for t in dataset.tensors)


class SelfEvolutionTests(unittest.TestCase):
    def test_trigger_requires_reference_and_inclusive_absolute_drop(self):
        for reference, current, expected in [
            (None, 0.30, False), (0.70, 0.75, False), (0.70, 0.70, False),
            (0.70, 0.61, False), (0.70, 0.60, True), (0.70, 0.50, True),
        ]:
            with self.subTest(reference=reference, current=current):
                self.assertEqual(validation_dropped(reference, current, 0.10), expected)

    def test_fresh_return_shards_and_disjoint_practice(self):
        steps = schedule(8)
        conditions = synthetic_data(steps, 493, 3, 64, 96, 256).conditions
        original = conditions[steps[0].condition]
        returning = conditions[steps[-1].condition]
        self.assertIs(original.test, returning.test)
        early, late = original.training_shards(steps[0]), returning.training_shards(steps[-1])
        for a, b in zip(early, late):
            self.assertEqual(len(a), len(b))
            self.assertEqual(a.tensors[1].sum().item(), b.tensors[1].sum().item())
        datasets = list(late)
        for condition in conditions.values():
            datasets.extend([*condition.clients, condition.validation, condition.test])
        for number, step in enumerate(steps, start=1):
            practice = synthetic_practice(step, 493, number, 64)
            self.assertEqual(contents(practice), contents(synthetic_practice(step, 493, number, 64)))
            self.assertEqual(practice.tensors[1].sum().item(), 32)
            datasets.append(practice)
        seen = set()
        for dataset in datasets:
            features = {tuple(row) for row in dataset.tensors[0].tolist()}
            self.assertTrue(seen.isdisjoint(features))
            seen.update(features)

    def test_practice_tracks_condition_and_counts_partial_batches(self):
        stable, shifted = schedule(8)[0], schedule(8)[2]
        a = synthetic_practice(stable, 1, 1, 17)
        b = synthetic_practice(shifted, 1, 1, 17)
        # Holding stream and labels fixed isolates the 120-degree mean rotation.
        sign = 2 * b.tensors[1].float() - 1
        torch.testing.assert_close(b.tensors[0][:, 0] - a.tensors[0][:, 0], -1.5 * sign)
        torch.testing.assert_close(b.tensors[0][:, 2:], a.tensors[0][:, 2:])
        model = classifier(4, 2)
        before = model[-1].weight.detach().clone()
        self.assertEqual(adapt(model, b, 16, 0.08), (17, 2))
        self.assertFalse(torch.equal(before, model[-1].weight))

    def test_paired_inputs_and_work(self):
        def run(method):
            initial = []
            inputs = []

            def observe(model, clients, *, generator, **settings):
                if not inputs:
                    initial.extend(p.detach().numpy().tobytes() for p in model.parameters())
                inputs.append(([contents(c) for c in clients], generator.get_state().numpy().tobytes(),
                               settings))
                return train_round(model, clients, generator=generator, **settings)

            rows = list(records(SMOKE, method=method, round_method=observe))
            return initial, inputs, rows

        start_a, inputs_a, baseline = run("fedavg")
        start_b, inputs_b, evolving = run("self_evolution")
        self.assertEqual(start_a, start_b)
        self.assertEqual(inputs_a, inputs_b)
        self.assertNotEqual(inputs_a[0][0], inputs_a[6][0])
        self.assertEqual(inputs_a[6][0], inputs_a[7][0])
        reference = None
        triggers = []
        for a, b in zip(baseline, evolving, strict=True):
            for key in ("round", "seed", "scenario", "condition", "ordinary_steps", "ordinary_examples"):
                self.assertEqual(a[key], b[key])
            self.assertEqual((a["ordinary_examples"], a["ordinary_steps"]), (192, 12))
            self.assertEqual((a["adaptation_triggered"], a["adaptation_examples"], a["adaptation_steps"]),
                             (False, 0, 0))
            self.assertEqual(b["validation_reference"], reference)
            expected = validation_dropped(reference, b["validation_before_adaptation"], SMOKE.validation_drop)
            self.assertEqual(b["adaptation_triggered"], expected)
            if expected:
                triggers.append(b["round"])
            self.assertEqual((b["adaptation_examples"], b["adaptation_steps"]),
                             (64, 4) if expected else (0, 0))
            reference = b["validation_after_adaptation"]
        self.assertTrue(triggers)

    def test_flipping_test_labels_only_changes_test_scores(self):
        def flip_test_labels(*args):
            data = synthetic_data(*args)
            for condition in data.conditions.values():
                labels = condition.test.tensors[1]
                labels.copy_(1 - labels)
            return data

        original = records(SMOKE, method="self_evolution")
        flipped = records(SMOKE, method="self_evolution", dataset_factory=flip_test_labels)
        for a, b in zip(original, flipped, strict=True):
            self.assertEqual(a.pop("current_test_accuracy"), 1 - b.pop("current_test_accuracy"))
            scores_a, scores_b = a.pop("test_accuracy_by_condition"), b.pop("test_accuracy_by_condition")
            self.assertEqual(scores_a, {key: 1 - value for key, value in scores_b.items()})
            self.assertEqual(a, b)
