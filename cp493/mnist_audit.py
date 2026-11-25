"""Audit the saved MNIST pilot and run exposure-matched, plain-parent controls."""

import argparse
from copy import deepcopy
import csv
from dataclasses import asdict
import hashlib
from itertools import islice
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from torch.utils.data import TensorDataset

from .adaptation import validation_dropped
from .experiment import METHODS
from .fedavg import classifier
from .mnist import proxy_usage, train_digits, write_json
from .mnist_data import (ARCHIVE_SHA256, LOW, MNISTConfig, build_splits, digit_schedule,
                         load_archive, split_manifest)


PAIRS = (("fedavg", "self_evolution"), ("continual", "combined"))
AUDIT_ROUND = 12


def check(ok, message):
    if not ok:
        raise ValueError(message)


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def proxy_origin(archive, manifest):
    pool_ids = manifest["server_proxy"]["indices"]
    by_pixels = {}
    for i in pool_ids:
        by_pixels.setdefault(archive["x_train"][i].tobytes(), []).append(i)
    comparisons = {}
    for role, split, indices in [
        ("client_training_including_return", "train",
         [i for name, s in manifest.items() if "/client_" in name for i in s["indices"]]),
        ("validation", "train", [i for name, s in manifest.items() if name.endswith("/validation")
                                 for i in s["indices"]]),
        ("official_test", "test", range(len(archive["x_test"]))),
    ]:
        matches = []
        for i in indices:
            for parent in by_pixels.get(archive[f"x_{split}"][i].tobytes(), []):
                matches.append({"proxy_train_index": parent, "other_index": i})
        comparisons[role] = {
            "official_split": split, "examples_compared": len(indices),
            "shared_source_ids": sorted(set(pool_ids).intersection(indices)) if split == "train" else [],
            "identical_pixel_pairs": matches,
        }
    labels = archive["y_train"]
    return {
        "official_source": "train", "reserved_originals": len(pool_ids),
        "ordered_index_sha256": manifest["server_proxy"]["ordered_index_sha256"],
        "train_indices_by_digit": {digit: [i for i in pool_ids if labels[i] == digit] for digit in range(10)},
        "within_pool_identical_pixels": [ids for ids in by_pixels.values() if len(ids) > 1],
        "overlap_checks": comparisons,
    }


def saved_events(rows, config, method, seed, data, pool):
    """Verify validation decisions and reconstruct draws without consulting test scores."""
    events = {}
    reference = None
    for number, (row, step) in enumerate(zip(rows, digit_schedule(config.rounds), strict=True), 1):
        check((row["round"], row["seed"], row["method"], row["condition"]) ==
              (number, seed, method, step.condition), "Saved round identity mismatch.")
        check(row["validation_reference"] == reference, "Validation reference mismatch.")
        expected = method in ("self_evolution", "combined") and validation_dropped(
            reference, row["validation_before_adaptation"], config.validation_drop)
        check(row["adaptation_triggered"] == expected, "Saved trigger disagrees with validation.")
        reference = row["validation_after_adaptation"]
        if expected:
            clients = data.conditions[step.condition].training_shards(step)
            counts = sum(torch.bincount(c.tensors[1], minlength=10) for c in clients)
            batch, trace = pool.generate(counts, seed, number, config.practice_examples)
            check(trace == row["practice_provenance"], "Cannot reproduce saved parent IDs and translations.")
            check(row["adaptation_examples"] == len(batch) and
                  row["adaptation_steps"] == (len(batch) + config.batch_size - 1) // config.batch_size,
                  "Practice count mismatch.")
            events[number] = trace
        else:
            check(row["adaptation_examples"] == row["adaptation_steps"] == 0
                  and "practice_provenance" not in row, "Unscheduled practice in saved run.")
    return events


def official_low_test(archive, subset):
    labels = archive["y_test"]
    ids = np.flatnonzero(np.isin(labels, np.arange(5)))
    images = archive["x_test"][ids]
    truth = labels[ids].astype(np.int64)
    check(subset.source_split == "test" and np.array_equal(subset.indices, ids),
          "Reference set differs from the official 0–4 selection.")
    inputs, targets = subset.tensors
    np.testing.assert_array_equal(inputs.numpy(), images[:, None].astype(np.float32) / 255)
    np.testing.assert_array_equal(targets.numpy(), truth)
    return ids, images, truth


def verify_source_labels(archive, data, pool):
    subsets = [pool]
    for c in data.conditions.values():
        subsets.extend([*c.clients, *(c.return_clients or []), c.validation, c.test])
    for subset in subsets:
        official = archive[f"y_{subset.source_split}"][subset.indices]
        np.testing.assert_array_equal(subset.tensors[1].numpy(), official)


def confusion(truth, logits):
    if logits.shape != (len(truth), 10) or not np.isfinite(logits).all():
        raise ValueError("Expected ten finite logits for each test image.")
    if not np.isin(truth, np.arange(10)).all():
        raise ValueError("Expected original digit labels.")
    predictions = logits.argmax(axis=1)
    counts = np.bincount(10 * truth.astype(np.int64) + predictions, minlength=100)
    return counts.reshape(10, 10), predictions


def audit_predictions(config, seed, method, data, pool, initial, saved, images, truth):
    model = deepcopy(initial)
    replay = train_digits(config, method, seed, data, model, pool.generate)
    for record, expected in zip(islice(replay, AUDIT_ROUND), saved[:AUDIT_ROUND], strict=True):
        check(record == expected, f"{method}, seed {seed}: replay mismatch at round {record['round']}.")
    replay.close()
    model.eval()
    with torch.inference_mode():
        logits = model(data.conditions[LOW].test.tensors[0]).numpy()
    matrix, predicted = confusion(truth, logits)
    # Independent ten-class forward pass from official pixels and the two affine layers.
    weights = {k: v.numpy() for k, v in model.state_dict().items()}
    flat = images.reshape(len(images), -1).astype(np.float32) / 255
    hidden = np.maximum(0, np.einsum("bi,oi->bo", flat, weights["1.weight"], dtype=np.float64)
                        + weights["1.bias"])
    raw = np.einsum("bi,oi->bo", hidden, weights["3.weight"]) + weights["3.bias"]
    np.testing.assert_allclose(raw, logits, rtol=1e-5, atol=2e-5)
    np.testing.assert_array_equal(predicted, raw.argmax(axis=1))
    score = matrix.trace() / matrix.sum()
    check(score == saved[AUDIT_ROUND - 1]["original_test_accuracy"], "Independent accuracy differs.")
    return matrix, predicted


class MatchedOriginals:
    """Identity-transform control using only recorded validation events and proxy parents."""

    def __init__(self, pool, events):
        self.pool = pool
        self.events = events
        self.positions = {int(index): pos for pos, index in enumerate(pool.indices)}

    def draw(self, counts, seed, number, count):
        trace = self.events[number]
        parents = trace["train_indices"]
        check(len(parents) == count and counts.tolist() == trace["client_label_counts"],
              "Matched proxy count or active label distribution differs.")
        check(all(p in self.positions for p in parents), "Parent outside the server proxy pool.")
        positions = [self.positions[p] for p in parents]
        pixels, labels = self.pool.tensors
        provenance = {"source": "server_proxy_from_official_train", "train_indices": parents,
                      "transform": "identity", "client_label_counts": counts.tolist()}
        return TensorDataset(pixels[positions], labels[positions]), provenance


def control_records(config, seed, method, data, pool, initial, events, translated):
    provider = MatchedOriginals(pool, events)
    run = train_digits(config, method, seed, data, deepcopy(initial), provider.draw,
                       matched_practice_rounds=events.keys())
    rows = []
    first_update = min(events, default=config.rounds + 1)
    for row, ref in zip(run, translated, strict=True):
        if row["round"] <= first_update:
            check(row["training_loss"] == ref["training_loss"] and
                  row["validation_before_adaptation"] == ref["validation_before_adaptation"],
                  "The paired arms diverged before their first proxy update.")
        if row["round"] < first_update:
            check(row["test_accuracy_by_condition"] == ref["test_accuracy_by_condition"],
                  "Paired arms differ before exposure.")
        for key in ("ordinary_examples", "ordinary_steps", "adaptation_examples", "adaptation_steps",
                    "replay_presentations", "teacher_presentations", "extra_validation_presentations"):
            check(row[key] == ref[key], f"Unmatched work at round {row['round']}: {key}")
        if row["adaptation_triggered"]:
            check(row["practice_provenance"]["train_indices"] == ref["practice_provenance"]["train_indices"],
                  "Paired arms drew different parents.")
        row["method"] = method + "_matched_originals"
        row["matched_update_applied"] = row.pop("adaptation_triggered")
        row["plain_proxy_presentations"] = row.pop("adaptation_examples")
        row["plain_proxy_optimizer_steps"] = row.pop("adaptation_steps")
        row["generated_practice_examples"] = 0
        rows.append(row)
    return rows


def write_matrices(path, matrices):
    with path.open("w", encoding="utf-8") as out:
        out.write("MNIST digit recognition, pilot round 12. Rows are official true digits, columns are predicted digits.\n")
        for (seed, method), matrix in matrices.items():
            out.write(f"\nSeed {seed} | {method} | correct {matrix.trace()} / {matrix.sum()}\n")
            for label, values in [("true/pred", range(10)), *enumerate(matrix), ("pred total", matrix.sum(axis=0))]:
                out.write(f"{label!s:>10}" + "".join(f"{v:6d}" for v in values) + "\n")


def audit(archive, baseline, configs, directory):
    origins, matrices, exposures, contrasts = {}, {}, [], []
    prediction_rows = []
    for scope in ("smoke", "pilot"):
        config = MNISTConfig.load(configs / f"mnist_{scope}.toml")
        location = baseline / scope
        for seed in config.seeds:
            data, pool = build_splits(archive, config, seed)
            verify_source_labels(archive, data, pool)
            manifest = json.loads((location / f"splits-{seed}.json").read_text())
            check(split_manifest(data, pool) == manifest, "Reconstructed splits differ from saved manifest.")
            origins[f"{scope}-{seed}"] = proxy_origin(archive, manifest)
            torch.manual_seed(seed)
            initial = classifier(data.input_size, data.classes, config.hidden_size)
            init_hash = hashlib.sha256(b"".join(t.numpy().tobytes() for t in initial.state_dict().values())).hexdigest()
            saved, events = {}, {}
            for method in METHODS:
                run = json.loads((location / f"{method}-{seed}.json").read_text())
                check((run["archive_sha256"], run["initial_weights_sha256"], run["config"]) ==
                      (ARCHIVE_SHA256, init_hash, asdict(config)), "Baseline archive, weights or config mismatch.")
                saved[method] = run["records"]
                events[method] = saved_events(run["records"], config, method, seed, data, pool)
                exposures.append({"run": scope, "method": method, "seed": seed, **proxy_usage(run["records"], len(pool))})
            if scope == "smoke":
                continue

            ids, images, truth = official_low_test(archive, data.conditions[LOW].test)
            if not prediction_rows:
                prediction_rows = [{"official_test_index": int(i), "true_digit": int(y)}
                                   for i, y in zip(ids, truth, strict=True)]
            for method in ("fedavg", "self_evolution"):
                matrix, pred = audit_predictions(config, seed, method, data, pool, initial, saved[method], images, truth)
                matrices[seed, method] = matrix
                for entry, digit in zip(prediction_rows, pred, strict=True):
                    entry[f"{method}_{seed}"] = int(digit)
            check(events["self_evolution"] == events["combined"],
                  "The two references have different exposures; four-way matching is unavailable.")
            for local, adaptive in PAIRS:
                rows = control_records(config, seed, local, data, pool, initial, events[adaptive], saved[adaptive])
                arm = local + "_matched_originals"
                use = {"run": "matched_control", "method": arm, "seed": seed,
                       **proxy_usage(rows, len(pool), plain=True)}
                exposures.append(use)
                write_json(directory / f"{arm}-{seed}.json", {
                    "benchmark": "MNIST digit recognition", "config": asdict(config), "archive_sha256": ARCHIVE_SHA256,
                    "seed": seed, "method": arm, "reference_arm": adaptive, "initial_weights_sha256": init_hash,
                    "schedule_source": "reference arm's recorded validation events", "proxy_use": use, "records": rows,
                })
                for number in (6, 12, 18):
                    for condition, plain_score in rows[number-1]["test_accuracy_by_condition"].items():
                        original = saved[local][number-1]["test_accuracy_by_condition"][condition]
                        shifted = saved[adaptive][number-1]["test_accuracy_by_condition"][condition]
                        contrasts.append({
                            "seed": seed, "local_method": local, "translated_method": adaptive,
                            "round": number, "condition": condition, "no_proxy_pct": 100 * original,
                            "matched_plain_pct": 100 * plain_score, "matched_translated_pct": 100 * shifted,
                            "plain_minus_no_proxy_pp": 100 * (plain_score - original),
                            "translated_minus_plain_pp": 100 * (shifted - plain_score),
                        })

    write_json(directory / "proxy_origins.json", {"archive_sha256": ARCHIVE_SHA256, "manifests": origins})
    write_matrices(directory / "round12_matrices.txt", matrices)
    write_csv(directory / "round12_predictions.csv", prediction_rows)
    write_csv(directory / "proxy_usage.csv", exposures)
    write_csv(directory / "paired_scores.csv", contrasts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("data/mnist/mnist.npz"))
    parser.add_argument("--baseline", type=Path, default=Path("results/mnist"))
    parser.add_argument("--configs", type=Path, default=Path("configs"))
    parser.add_argument("--output", type=Path, default=Path("results/mnist/audit"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a fresh output directory to preserve previously saved results.")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    archive = load_archive(args.archive)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as tmp:
        staging = Path(tmp) / "audit"
        staging.mkdir()
        audit(archive, args.baseline, args.configs, staging)
        staging.rename(args.output)
    print(f"Wrote six round-12 confusion matrices and six matched pilot controls to {args.output}.")


if __name__ == "__main__":
    main()
