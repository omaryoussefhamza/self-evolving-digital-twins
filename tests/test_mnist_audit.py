from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from cp493.fedavg import classifier
from cp493.mnist import proxy_usage, run_benchmark, train_digits
from cp493.mnist_audit import (MatchedOriginals, confusion, control_records, official_low_test,
                               proxy_origin, saved_events, verify_source_labels)
from cp493.mnist_data import LOW, build_splits, digit_schedule
from tests.test_mnist import SMALL, fixture


class MNISTAuditTests(unittest.TestCase):
    def test_new_run_records_pool_access_and_counts_actual_exposure(self):
        with TemporaryDirectory() as temporary, patch(
            "cp493.experiment.validation_dropped", side_effect=lambda prior, *_: prior is not None
        ):
            output = Path(temporary)
            run_benchmark(SMALL, fixture(), output)
            for method in ("fedavg", "self_evolution", "continual", "combined"):
                run = json.loads((output / f"{method}-493.json").read_text())
                active = method in ("self_evolution", "combined")
                use = run["proxy_use"]
                self.assertEqual(use["reserved_proxy_originals"], 10 * SMALL.proxy_per_digit)
                self.assertEqual(use["proxy_originals_accessible_to_update"],
                                 use["reserved_proxy_originals"] if active else 0)
                self.assertEqual(use["generated_practice_examples"], 5 * SMALL.practice_examples if active else 0)
                self.assertEqual(use["plain_proxy_presentations"], 0)
                parents = {p for r in run["records"] if "practice_provenance" in r
                           for p in r["practice_provenance"]["train_indices"]}
                self.assertEqual(use["unique_parents_used"], len(parents))
                self.assertEqual(use["proxy_sgd_steps"], sum(r["adaptation_steps"] for r in run["records"]))

    def test_confusion_uses_all_ten_logits_and_keeps_empty_rows(self):
        logits = np.zeros((4, 10))
        truth = np.array([0, 0, 1, 4])
        logits[np.arange(4), [8, 5, 8, 4]] = 2
        matrix, predicted = confusion(truth, logits)
        self.assertEqual(matrix.shape, (10, 10))
        self.assertEqual(predicted.tolist(), [8, 5, 8, 4])
        self.assertEqual(matrix[0, 8], 1)
        self.assertEqual(matrix[1, 8], 1)
        self.assertEqual(matrix.trace(), 1)
        self.assertEqual(matrix[5:].sum(), 0)
        self.assertEqual(matrix.sum(axis=0).tolist(), [0, 0, 0, 0, 1, 1, 0, 0, 2, 0])
        with self.assertRaisesRegex(ValueError, "ten finite logits"):
            confusion(truth, logits[:, :5])

    def test_reference_selection_and_original_label_mapping(self):
        archive = fixture()
        data, pool = build_splits(archive, SMALL, 493)
        subset = data.conditions[LOW].test
        ids, _, truth = official_low_test(archive, subset)
        self.assertEqual(len(ids), np.count_nonzero(archive["y_test"] < 5))
        np.testing.assert_array_equal(truth, archive["y_test"][ids])
        verify_source_labels(archive, data, pool)
        subset.tensors[1][0] = 9
        with self.assertRaises(AssertionError):
            official_low_test(archive, subset)
        with self.assertRaises(AssertionError):
            verify_source_labels(archive, data, pool)

    def test_row_overlap_and_identical_pixels_are_distinct_checks(self):
        archive = {"x_train": np.array([[1], [1], [2]]), "y_train": np.array([0, 0, 1]),
                   "x_test": np.array([[2], [1]])}
        manifest = {"server_proxy": {"indices": [0], "ordered_index_sha256": "fixture"},
                    "low/client_0": {"indices": [1]}, "low/validation": {"indices": [0]}}
        checks = proxy_origin(archive, manifest)["overlap_checks"]
        self.assertEqual(checks["client_training_including_return"]["shared_source_ids"], [])
        self.assertEqual(checks["validation"]["shared_source_ids"], [0])
        self.assertEqual(checks["official_test"]["shared_source_ids"], [])
        for role in ("client_training_including_return", "official_test"):
            self.assertEqual(checks[role]["identical_pixel_pairs"], [{"proxy_train_index": 0, "other_index": 1}])

    def test_plain_control_keeps_raw_parents_in_order_including_repeats(self):
        archive = fixture()
        _, pool = build_splits(archive, SMALL, 493)
        p, q = (int(i) for i in pool.indices[:2])
        counts = torch.ones(10, dtype=torch.long)
        event = {"train_indices": [q, p, q], "client_label_counts": counts.tolist()}
        provider = MatchedOriginals(pool, {2: event})
        batch, trace = provider.draw(counts, 493, 2, 3)
        self.assertEqual(trace["transform"], "identity")
        self.assertEqual(trace["train_indices"], [q, p, q])
        np.testing.assert_array_equal(batch.tensors[0].numpy()[:, 0],
                                      archive["x_train"][[q, p, q]].astype(np.float32) / 255)
        self.assertEqual(batch.tensors[1].tolist(), archive["y_train"][[q, p, q]].tolist())
        with self.assertRaisesRegex(ValueError, "distribution"):
            provider.draw(2 * counts, 493, 2, 3)
        event["train_indices"][0] = -1
        with self.assertRaisesRegex(ValueError, "outside"):
            provider.draw(counts, 493, 2, 3)

    def test_control_inherits_events_and_matches_work_without_own_trigger(self):
        config = replace(SMALL, practice_examples=5, batch_size=4)
        data, pool = build_splits(fixture(), config, 493)
        torch.manual_seed(493)
        initial = classifier(784, 10, config.hidden_size)
        reference = classifier(784, 10, config.hidden_size)
        reference.load_state_dict(initial.state_dict())
        # Schedule only for this test fixture. A saved validation event supplies the pilot schedule.
        source = list(train_digits(config, "combined", 493, data, reference, pool.generate,
                                   matched_practice_rounds={3}))
        trace = source[2]["practice_provenance"]
        with patch("cp493.experiment.validation_dropped", side_effect=AssertionError("independent trigger")):
            rows = control_records(config, 493, "continual", data, pool, initial, {3: trace}, source)
        self.assertEqual([r["round"] for r in rows if r["matched_update_applied"]], [3])
        self.assertEqual(rows[2]["practice_provenance"]["train_indices"], trace["train_indices"])
        self.assertEqual(sum(r["plain_proxy_presentations"] for r in rows), 5)
        self.assertEqual(sum(r["plain_proxy_optimizer_steps"] for r in rows), 2)
        self.assertTrue(all(r["generated_practice_examples"] == 0 for r in rows))
        exposure = proxy_usage(rows, len(pool), plain=True)
        self.assertEqual(exposure["proxy_originals_accessible_to_update"], len(pool))
        self.assertEqual(exposure["generated_practice_examples"], 0)
        self.assertEqual(exposure["plain_proxy_presentations"], 5)
        self.assertEqual([r["replay_presentations"] for r in rows], [r["replay_presentations"] for r in source])

    def test_event_selection_ignores_test_scores_and_detects_tampering(self):
        data, pool = build_splits(fixture(), SMALL, 493)
        rows = []
        reference = None
        for i, step in enumerate(digit_schedule(6), 1):
            before = .50 if i >= 3 else .80
            row = {"round": i, "method": "self_evolution", "seed": 493, "condition": step.condition,
                   "validation_reference": reference, "validation_before_adaptation": before,
                   "validation_after_adaptation": before, "adaptation_triggered": i == 3,
                   "adaptation_examples": 0, "adaptation_steps": 0}
            if i == 3:
                clients = data.conditions[step.condition].training_shards(step)
                counts = sum(torch.bincount(c.tensors[1], minlength=10) for c in clients)
                _, trace = pool.generate(counts, 493, i, SMALL.practice_examples)
                row.update(practice_provenance=trace, adaptation_examples=64, adaptation_steps=2)
            rows.append(row)
            reference = before
        # No test score is present in this input.
        self.assertEqual(set(saved_events(rows, SMALL, "self_evolution", 493, data, pool)), {3})
        rows[2]["practice_provenance"]["train_indices"][0] = -1
        with self.assertRaisesRegex(ValueError, "parent IDs"):
            saved_events(rows, SMALL, "self_evolution", 493, data, pool)
