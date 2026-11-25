from dataclasses import replace
import json
from pathlib import Path
from statistics import stdev
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from cp493 import pilot
from cp493.data import synthetic_data
from cp493.experiment import Config, METHODS, records
from cp493.fedavg import classifier, train_round
from cp493.scenario import schedule


ROOT = Path(__file__).resolve().parents[1]
PILOT = Config.load(ROOT / "configs/pilot.toml")


class PilotTests(unittest.TestCase):
    def test_timing_and_fixed_settings(self):
        smoke = Config.load(ROOT / "configs/smoke.toml")
        self.assertEqual(replace(PILOT, rounds=8, seeds=[493]), smoke)
        self.assertEqual((PILOT.rounds, len(PILOT.seeds)), (40, 5))
        steps = schedule(40)
        self.assertEqual([s.scenario for s in steps],
                         [label for label in ("stable", "sudden", "gradual", "return") for _ in range(10)])
        self.assertEqual([s.condition for s in steps[20:30]], [f"gradual_{i}" for i in range(1, 11)])
        for i, step in enumerate(steps[20:30], 1):
            self.assertAlmostEqual(120 * step.shift, 120 - 6 * i)
        self.assertEqual({s.condition for s in steps[30:]}, {"original"})
        self.assertEqual(len({s.condition for s in steps}), 12)

    def test_pairing_and_complete_real_records(self):
        def capture(method):
            trace = []
            def model(*args):
                net = classifier(*args)
                trace.append(tuple(p.detach().numpy().tobytes() for p in net.parameters()))
                return net

            def data(*args):
                bundle = synthetic_data(*args)
                for c in bundle.conditions.values():
                    for split in (c.validation, c.test):
                        trace.append(tuple(t.numpy().tobytes() for t in split.tensors))
                return bundle

            def train(model, clients, **kwargs):
                trace.append(tuple(t.numpy().tobytes() for c in clients for t in c.tensors))
                trace.append(kwargs["generator"].get_state().numpy().tobytes())
                return train_round(model, clients, **kwargs)

            def inspect(*args, **kwargs):
                return records(*args, **kwargs, model_factory=model, dataset_factory=data, round_method=train)
            with patch.object(pilot, "records", inspect):
                run = pilot.run_one(PILOT, method, 493)
            pilot.validate(run, PILOT, method, 493)
            for row in run["records"]:
                for key in ("adaptation_triggered", "adaptation_examples", "adaptation_steps",
                            "teacher_forward_passes", "replay_presentations", "teacher_presentations"):
                    self.assertIn(key, row)
            run["records"].pop()
            with self.assertRaises(ValueError):
                pilot.validate(run, PILOT, method, 493)
            return trace

        reference = capture("fedavg")
        for method in METHODS[1:]:
            self.assertEqual(reference, capture(method))

    def test_paired_summary_and_missing_seed(self):
        config = replace(PILOT, seeds=[493, 494])
        with TemporaryDirectory() as folder:
            directory = Path(folder)
            for seed in config.seeds:
                for method in METHODS:
                    run = pilot.run_one(config, method, seed)
                    path = directory / f"{method}-{seed}.json"
                    path.write_text(json.dumps(run))
            individual, aggregates = pilot.summarize(config, directory)
            self.assertEqual((individual, aggregates), pilot.summarize(config, directory))
            self.assertEqual(len(individual), 2 * 4 * 6)
            group = next(r for r in aggregates if (r["method"], r["window"]) == ("combined", "gradual_mean"))
            diffs = []
            for seed in config.seeds:
                def gradual_avg(method):
                    saved = json.loads((directory / f"{method}-{seed}.json").read_text())
                    return sum(r["current_test_accuracy"] for r in saved["records"][20:30]) / 10
                diffs.append(gradual_avg("combined") - gradual_avg("fedavg"))
            self.assertAlmostEqual(group["current_delta_vs_fl_mean"], sum(diffs) / 2)
            self.assertAlmostEqual(group["current_delta_vs_fl_sd"], stdev(diffs))
            path.unlink()
            with self.assertRaises(FileNotFoundError):
                pilot.summarize(config, directory)
