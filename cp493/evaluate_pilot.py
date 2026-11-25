"""Evaluate the saved 40-round synthetic pilot and plot its paired results."""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev
from tempfile import TemporaryDirectory

from .experiment import Config, METHODS
from .pilot import validate

REFERENCE_ROUND = 10
SHIFTED_STAGE = range(11, 21)
FORGETTING_WINDOW = range(11, 31)
RECOVERY_THRESHOLD = 0.70
NOT_REACHED = "not reached within the shifted stage"


def forgetting_pp(reference_accuracy, accuracy):
    """Nonnegative loss in percentage points, using accuracy fractions as inputs."""
    return 100 * max(0.0, reference_accuracy - accuracy)


def recovery_rounds(records, threshold=RECOVERY_THRESHOLD):
    """First hit after k completed shifted rounds (k=1..10), or None at stage end."""
    shifted = [row["test_accuracy_by_condition"]["shifted"]
               for row in records if row["round"] in SHIFTED_STAGE]
    return next((k for k, score in enumerate(shifted, 1) if score >= threshold), None)


def measure(records):
    """Return per-run scalars and per-round series from validated, ordered records."""
    reference = records[REFERENCE_ROUND - 1]["test_accuracy_by_condition"]["original"]
    series = {}
    drops = []
    for row in records:
        r = row["round"]
        original = row["test_accuracy_by_condition"]["original"]
        series["current_accuracy_pct", r] = 100 * row["current_test_accuracy"]
        series["original_accuracy_pct", r] = 100 * original
        if r in FORGETTING_WINDOW:
            drop = forgetting_pp(reference, original)
            series["forgetting_pp", r] = drop
            drops.append(drop)

    recovery = recovery_rounds(records)
    # The baseline and practice-only method do not allocate device memory.
    capacities = {row.get("memory_capacity_per_device", 0) for row in records}
    if len(capacities) != 1:
        raise ValueError("Memory capacity changed during a fixed-capacity run.")
    scalars = {
        "forgetting_at_20_pp": series["forgetting_pp", 20],
        "max_forgetting_11_30_pp": max(drops),
        "recovery_rounds": recovery,
        "recovered_in_stage": int(recovery is not None),
        "memory_capacity_per_device": capacities.pop(),
    }
    for metric in ("adaptation_examples", "adaptation_steps", "replay_presentations",
                   "teacher_forward_passes", "teacher_presentations", "extra_validation_presentations"):
        scalars[metric] = sum(row[metric] for row in records)
    return scalars, series


def paired_summary(values, baseline):
    """Compare like-numbered seeds. Unobserved crossing times have no numeric value."""
    raw = {f"seed_{s}": v for s, v in values.items()}
    deltas = {f"delta_seed_{s}": v - baseline[s] if v is not None and baseline[s] is not None else None
              for s, v in values.items()}
    observed = [v for v in values.values() if v is not None]
    pairs = [d for d in deltas.values() if d is not None]
    return {
        **raw, "n_observed": len(observed),
        "mean": mean(observed) if observed else None,
        "sample_sd": stdev(observed) if len(observed) >= 2 else None,
        **deltas, "n_pairs": len(pairs),
        "paired_mean_delta_vs_fl": mean(pairs) if pairs else None,
        "paired_sample_sd": stdev(pairs) if len(pairs) >= 2 else None,
    }


def summary_rows(by_method, seeds, *, per_round):
    rows = []
    for key in by_method["fedavg"][seeds[0]]:
        baseline = {s: by_method["fedavg"][s][key] for s in seeds}
        for method in METHODS:
            values = {s: by_method[method][s][key] for s in seeds}
            labels = {"metric": key[0], "round": key[1]} if per_round else {"metric": key}
            rows.append({"method": method, **labels, **paired_summary(values, baseline)})
    return rows


def evaluate(config, directory):
    if config.rounds != 40:
        raise ValueError("The specified retention and recovery windows require the 40-round pilot.")
    scalars, series = {}, {}
    for method in METHODS:
        scalars[method], series[method] = {}, {}
        for seed in config.seeds:
            run = json.loads((directory / f"{method}-{seed}.json").read_text())
            validate(run, config, method, seed)
            scalars[method][seed], series[method][seed] = measure(run["records"])
    return (summary_rows(scalars, config.seeds, per_round=False),
            summary_rows(series, config.seeds, per_round=True))


def write_table(path, rows):
    with path.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: NOT_REACHED if k.startswith("seed_") and v is None else v
                             for k, v in row.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/pilot.toml"))
    parser.add_argument("--input", type=Path, default=Path("results/pilot"))
    parser.add_argument("--output", type=Path, default=Path("results/evaluation"))
    args = parser.parse_args()
    config = Config.load(args.config)
    metrics, curves = evaluate(config, args.input)

    from .pilot_plots import plot_evaluation

    args.output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output) as temp:
        directory = Path(temp)
        write_table(directory / "metrics_by_seed.csv", metrics)
        write_table(directory / "trajectories_by_seed.csv", curves)
        plot_evaluation(metrics, curves, config.seeds, directory)
        for path in sorted(directory.iterdir()):
            dest = args.output / path.name
            path.replace(dest)
            print(dest)


if __name__ == "__main__":
    main()
