import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from cp493.evaluate_pilot import (NOT_REACHED, evaluate, forgetting_pp, measure, paired_summary,
                                  recovery_rounds, write_table)
from cp493.experiment import Config


ROOT = Path(__file__).resolve().parents[1]


def example_records():
    rows = []
    for r in range(1, 41):
        # A high score before/after the stage must never count toward recovery.
        scores = {"original": .80}
        if r >= 11:
            scores["shifted"] = .65 if r <= 20 else .99
        current = scores["shifted"] if 11 <= r <= 20 else .99
        rows.append({"round": r, "test_accuracy_by_condition": scores, "current_test_accuracy": current,
                     "adaptation_examples": 64 if r == 11 else 0, "adaptation_steps": 4 if r == 11 else 0,
                     "replay_presentations": 8, "teacher_presentations": 8, "teacher_forward_passes": 2,
                     "extra_validation_presentations": 96 if r == 11 else 0, "memory_capacity_per_device": 4})
    return rows


class EvaluationTests(unittest.TestCase):
    def test_retention_clamp_reference_and_peak_window(self):
        self.assertEqual(forgetting_pp(.80, .85), 0)
        rows = example_records()
        for r, original in [(9, 1.), (11, .85), (20, .60), (30, .45), (31, .10)]:
            rows[r - 1]["test_accuracy_by_condition"]["original"] = original
        metrics, curves = measure(rows)
        self.assertEqual(curves["forgetting_pp", 11], 0)
        self.assertAlmostEqual(metrics["forgetting_at_20_pp"], 20)
        self.assertAlmostEqual(metrics["max_forgetting_11_30_pp"], 35)
        self.assertEqual(curves["original_accuracy_pct", 31], 10)
        self.assertEqual([r for m, r in curves if m == "forgetting_pp"], list(range(11, 31)))

    def test_unreached_and_inclusive_first_crossing_on_fixed_shifted_set(self):
        rows = example_records()
        self.assertIsNone(recovery_rounds(rows))
        self.assertEqual(measure(rows)[0]["recovered_in_stage"], 0)
        for round_number in (20, 15, 11):
            with self.subTest(round_number=round_number):
                rows[round_number - 1]["test_accuracy_by_condition"]["shifted"] = .70
                self.assertEqual(recovery_rounds(rows), round_number - 10)
        self.assertIsNone(recovery_rounds(rows, threshold=1.0))

    def test_work_totals_keep_units_and_capacity(self):
        metrics, _ = measure(example_records())
        self.assertEqual(metrics["adaptation_examples"], 64)
        self.assertEqual(metrics["adaptation_steps"], 4)
        self.assertEqual(metrics["replay_presentations"], 320)
        self.assertEqual(metrics["teacher_presentations"], 320)
        self.assertEqual(metrics["teacher_forward_passes"], 80)
        self.assertEqual(metrics["extra_validation_presentations"], 96)
        self.assertEqual(metrics["memory_capacity_per_device"], 4)

    def test_paired_statistics_with_different_missing_seeds(self):
        # Both means use three observations, but only seeds 494 and 495 form pairs.
        values = {493: None, 494: 8, 495: 7, 496: 3, 497: None}
        baseline = {497: None, 496: None, 495: 5, 494: 7, 493: 9}
        stats = paired_summary(values, baseline)
        self.assertEqual((stats["n_observed"], stats["n_pairs"]), (3, 2))
        self.assertEqual(stats["mean"], 6)
        self.assertAlmostEqual(stats["sample_sd"], 7 ** .5)
        self.assertEqual(stats["paired_mean_delta_vs_fl"], 1.5)
        self.assertAlmostEqual(stats["paired_sample_sd"], .5 ** .5)
        self.assertIsNone(stats["delta_seed_496"])
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.csv"
            write_table(path, [{"metric": "recovery_rounds", **stats}])
            with path.open() as inp:
                entry = next(csv.DictReader(inp))
            self.assertEqual(entry["seed_493"], NOT_REACHED)
            self.assertEqual(entry["delta_seed_496"], "")
        empty = paired_summary({493: None}, {493: None})
        self.assertIsNone(empty["mean"])
        self.assertIsNone(empty["sample_sd"])
        self.assertIsNone(paired_summary({493: 1}, {493: 2})["paired_sample_sd"])

    def test_saved_pilot_has_five_paired_trajectories_per_method(self):
        config = Config.load(ROOT / "configs/pilot.toml")
        metrics, curves = evaluate(config, ROOT / "results/pilot")
        self.assertEqual(len(curves), 4 * (40 + 40 + 20))
        self.assertTrue(all(r["n_observed"] == r["n_pairs"] == 5 for r in curves))
        reference = next(r for r in curves if (r["method"], r["metric"], r["round"]) ==
                         ("combined", "original_accuracy_pct", 10))
        end = next(r for r in curves if (r["method"], r["metric"], r["round"]) ==
                   ("combined", "original_accuracy_pct", 20))
        drop = next(r for r in metrics if (r["method"], r["metric"]) == ("combined", "forgetting_at_20_pp"))
        for seed in config.seeds:
            key = f"seed_{seed}"
            self.assertEqual(drop[key], max(0, reference[key] - end[key]))
