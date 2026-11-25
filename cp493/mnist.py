"""Run the paired ten-class MNIST digit-recognition benchmark."""

import argparse
from copy import deepcopy
import csv
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from .experiment import METHODS, records
from .fedavg import classifier
from .mnist_data import (ARCHIVE_SHA256, HIGH, LOW, MNISTConfig, build_splits, digit_schedule,
                         load_archive, split_manifest)


def digit_records(config, method, seed, data, pool, initial_model):
    # Reuse the same immutable shards and independent copies of the same initial
    # network across methods. records() resets the ordinary shuffle stream per seed.
    return train_digits(config, method, seed, data, deepcopy(initial_model), pool.generate)


def train_digits(config, method, seed, data, model, practice_from_counts, *, matched_practice_rounds=None):
    """Update the supplied model in place, yielding after final evaluation in each round."""
    run = records(replace(config, seeds=[seed]), method=method, schedule_factory=digit_schedule,
                  dataset_factory=lambda *_: data, model_factory=lambda *_: model,
                  practice_from_counts=practice_from_counts, detailed_work=True, record_memory_work=True,
                  matched_practice_rounds=matched_practice_rounds)
    for row in run:
        row["original_test_accuracy"] = row["test_accuracy_by_condition"][LOW]
        yield row


def proxy_work(rows, *, plain=False):
    events = [r for r in rows if "practice_provenance" in r]
    parents = {i for r in events for i in r["practice_provenance"]["train_indices"]}
    return {
        "update_rounds": " ".join(str(r["round"]) for r in events),
        "unique_parents_used": len(parents),
        "generated_practice_examples": 0 if plain else sum(r["adaptation_examples"] for r in rows),
        "plain_proxy_presentations": sum(r["plain_proxy_presentations"] for r in rows) if plain else 0,
        "proxy_sgd_steps": sum(r["plain_proxy_optimizer_steps" if plain else "adaptation_steps"] for r in rows),
    }


def proxy_usage(rows, pool_size, *, plain=False):
    accessible = plain or rows[0]["method"] in ("self_evolution", "combined")
    return {"reserved_proxy_originals": pool_size,
            "proxy_originals_accessible_to_update": pool_size if accessible else 0,
            **proxy_work(rows, plain=plain)}


def summarize(rows):
    stage_length = len(rows) // 3
    first, second, last = rows[stage_length - 1], rows[2 * stage_length - 1], rows[-1]
    work = proxy_work(rows)
    result = {
        "method": last["method"], "seed": last["seed"],
        "initial_end_original_pct": 100 * first["original_test_accuracy"],
        "new_digits_end_current_pct": 100 * second["current_test_accuracy"],
        "new_digits_end_original_pct": 100 * second["original_test_accuracy"],
        "new_digits_end_forgetting_pp": 100 * max(0, first["original_test_accuracy"] - second["original_test_accuracy"]),
        "return_end_original_pct": 100 * last["original_test_accuracy"],
        "return_end_digits_5_9_pct": 100 * last["test_accuracy_by_condition"][HIGH],
        "trigger_rounds": work["update_rounds"], "distinct_proxy_parents_used": work["unique_parents_used"],
    }
    for key in ("ordinary_examples", "ordinary_steps", "adaptation_examples", "adaptation_steps",
                "replay_presentations", "replay_forward_passes", "teacher_presentations",
                "teacher_forward_passes", "extra_validation_presentations"):
        result[key] = sum(r[key] for r in rows)
    for key in ("memory_capacity_per_device", "memory_candidates_cumulative", "memory_writes_cumulative"):
        result[key] = last[key]
    result["final_memory_examples_all_devices"] = sum(last["memory_size_by_device"])
    return result


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run_benchmark(config, archive, output):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    output.mkdir(parents=True, exist_ok=True)
    summary = []
    with TemporaryDirectory(dir=output) as tmp:
        staging = Path(tmp)
        for seed in config.seeds:
            data, pool = build_splits(archive, config, seed)
            filename = f"splits-{seed}.json"
            write_json(staging / filename, split_manifest(data, pool))
            torch.manual_seed(seed)
            initial = classifier(data.input_size, data.classes, config.hidden_size)
            digest = hashlib.sha256()
            for tensor in initial.state_dict().values():
                digest.update(tensor.numpy().tobytes())
            for method in METHODS:
                rows = list(digit_records(config, method, seed, data, pool, initial))
                summary.append(summarize(rows))
                write_json(staging / f"{method}-{seed}.json", {
                    "benchmark": "MNIST digit recognition", "archive_sha256": ARCHIVE_SHA256,
                    "split_manifest": filename, "initial_weights_sha256": digest.hexdigest(),
                    "seed": seed, "method": method, "config": asdict(config),
                    "proxy_use": proxy_usage(rows, len(pool)), "records": rows,
                })
        with (staging / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
        for path in staging.iterdir():
            path.replace(output / path.name)
    print(f"Recorded {len(summary)} MNIST digit-recognition runs of {config.rounds} rounds in {output}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("data/mnist/mnist.npz"))
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run_benchmark(MNISTConfig.load(args.config), load_archive(args.archive), args.output)


if __name__ == "__main__":
    main()
