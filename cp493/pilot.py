"""Run and summarize the paired, 40-round synthetic pilot."""

import argparse
import csv
from dataclasses import asdict, replace
import json
from pathlib import Path
from statistics import mean, stdev
from tempfile import TemporaryDirectory

import torch

from .experiment import Config, METHODS, records
from .scenario import schedule

# Inclusive, one-based round numbers fixed before training.
WINDOWS = {"stable_end": (10, 10), "sudden_start": (11, 11), "sudden_end": (20, 20),
           "gradual_mean": (21, 30), "return_start": (31, 31), "return_end": (40, 40)}


def run_one(config, method, seed):
    rows = list(records(replace(config, seeds=[seed]), method=method, detailed_work=True))
    for row in rows:
        row["original_test_accuracy"] = row["test_accuracy_by_condition"]["original"]
    return {"method": method, "seed": seed, "config": asdict(config), "records": rows}


def validate(run, config, method, seed):
    if (run["method"], run["seed"], run["config"]) != (method, seed, asdict(config)):
        raise ValueError(f"Run configuration or identity mismatch for {method}, seed {seed}")
    seen = set()
    for i, (row, step) in enumerate(zip(run["records"], schedule(config.rounds), strict=True), 1):
        if (row["method"], row["seed"], row["round"], row["scenario"], row["condition"]) != (
                method, seed, i, step.scenario, step.condition):
            raise ValueError(f"Unexpected record for {method}, seed {seed}, round {i}")
        seen.add(step.condition)
        scores = row["test_accuracy_by_condition"]
        if (set(scores) != seen or scores["original"] != row["original_test_accuracy"]
                or scores[step.condition] != row["current_test_accuracy"]):
            raise ValueError(f"Inconsistent condition scores in round {i}")


def run_pilot(config, directory):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    directory.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=directory) as temp:
        pending = Path(temp) / "run.json"
        for seed in config.seeds:
            for method in METHODS:
                run = run_one(config, method, seed)
                validate(run, config, method, seed)
                target = directory / f"{method}-{seed}.json"
                pending.write_text(json.dumps(run, indent=2, allow_nan=False) + "\n")
                pending.replace(target)
                print(f"Wrote {target}")


def summarize(config, directory):
    individual = []
    for seed in config.seeds:
        runs = {}
        for method in METHODS:
            run = json.loads((directory / f"{method}-{seed}.json").read_text())
            validate(run, config, method, seed)
            runs[method] = run["records"]
        for window, (first, last) in WINDOWS.items():
            baseline = runs["fedavg"][first-1:last]
            for method, rows in runs.items():
                selected = rows[first-1:last]
                result = dict(seed=seed, window=window, method=method)
                for metric in ("current", "original"):
                    key = metric + "_test_accuracy"
                    result[metric] = mean(r[key] for r in selected)
                    result[metric + "_delta_vs_fl"] = result[metric] - mean(r[key] for r in baseline)
                individual.append(result)

    aggregates = []
    for window in WINDOWS:
        for method in METHODS:
            matched = [r for r in individual if (r["window"], r["method"]) == (window, method)]
            group = dict(window=window, method=method, n=len(matched))
            for metric in ("current", "original", "current_delta_vs_fl", "original_delta_vs_fl"):
                values = [r[metric] for r in matched]
                group[metric + "_mean"] = mean(values)
                group[metric + "_sd"] = stdev(values)
            aggregates.append(group)
    return individual, aggregates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "summarize"))
    parser.add_argument("--config", type=Path, default=Path("configs/pilot.toml"))
    parser.add_argument("--directory", type=Path, default=Path("results/pilot"))
    args = parser.parse_args()
    config = Config.load(args.config)
    if config.rounds != 40 or len(config.seeds) < 2:
        parser.error("The pilot windows require 40 rounds and sample SD requires at least two seeds.")
    if args.action == "run":
        run_pilot(config, args.directory)
    else:
        individual, aggregates = summarize(config, args.directory)
        for filename, rows in (("paired_seeds.csv", individual), ("summary.csv", aggregates)):
            with (args.directory / filename).open("w", newline="") as out:
                writer = csv.DictWriter(out, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            print(f"Wrote {args.directory / filename}")


if __name__ == "__main__":
    main()
