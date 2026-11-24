# Self-Evolving Digital Twins

### Federated and Continual Learning for Adaptive Digital Twin Networks

## Overview

This repository contains the research and implementation of a **Self-Evolving Digital Twin (DT) framework** integrating **Federated Learning (FL)**, **Continual Learning (CL)**, and adaptive model evolution.

The project focuses on improving the accuracy, adaptability, and reliability of digital twins in dynamic network environments while addressing **catastrophic forgetting**, where models lose previously learned knowledge when adapting to new tasks.

## Research Objectives

- Develop a federated learning framework for distributed digital twin networks.
- Enable adaptive model updates through self-evolution mechanisms.
- Integrate continual learning to preserve previously acquired knowledge.
- Optimize the trade-offs between model accuracy, training latency, energy consumption, and system reliability.

## Implementation

The performance evaluation consists of four experimental configurations:

| Experiment | Description |
|---|---|
| **1. FL Baseline** | Standard federated learning. |
| **2. FL + Self-Evolution** | Federated learning with adaptive model evolution. |
| **3. FL + Continual Learning** | Federated learning with knowledge retention mechanisms. |
| **4. Proposed Framework** | Integration of federated learning, self-evolution, and continual learning. |

## Running the three implemented methods

The runnable baseline uses synthetic binary labels, with 0 for normal and 1 for attack.
The observations are development data. Their accuracy measures performance on the
generator and does not establish real-world attack detection performance.

### Setup and run

From the repository root on macOS or Linux, using Python 3.12, run

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m cp493.experiment --config configs/smoke.toml --method fedavg --output results/fedavg.jsonl
.venv/bin/python -m cp493.experiment --config configs/smoke.toml --method self_evolution --output results/self_evolution.jsonl
.venv/bin/python -m cp493.experiment --config configs/smoke.toml --method continual --output results/continual.jsonl
```

The smoke configuration runs three CPU clients for eight rounds with seed 493.
To change the run, copy the TOML file and pass its path with `--config`.
Multiple seeds produce independent model initializations and data splits in one
JSON Lines file. A successful run replaces the chosen output file.
Omitting `--method` runs the standard FL baseline.

### Conditions and data

Each example has four continuous features with independent unit Gaussian noise.
The two classes have opposite means. A rotating direction in the first two
features controls which observations look like each class. The last two
features contain only noise.

| Smoke rounds | Scenario | Condition and class-mean direction |
| --- | --- | --- |
| 1–2 | Stable | `original`, 0° |
| 3–4 | Sudden change | `shifted`, 120° |
| 5–6 | Gradual change | `gradual_1` at 90°, then `gradual_2` at 60° |
| 7–8 | Return | `original`, 0°, with fresh training shards and the original validation and test sets |

The schedule divides eight or more rounds across the four stages as evenly as
possible. Earlier stages receive any extra rounds. Each gradual round has a
distinct condition, evenly spaced from the sudden shift toward 60°.

Within each condition, every client owns a disjoint training shard. The generator
assigns client sizes from 75% to 125% of the configured average and positive-class
shares from 35% to 65%, with integer rounding. The smoke shards contain 17/48,
32/64, and 52/80 positive examples. Separate random streams produce the client
shards, validation data, and test data. Each smoke condition has a balanced
96-example validation set and a balanced 256-example test set. On return, each
client draws a fresh shard once for the stage with its original size, label
count, and class-conditional distribution. Rounds 7 and 8 share those new shards.

### Training and results

The server initializes a four-input network with eight ReLU hidden units and two
output logits. Each round, every client copies the same server model and performs
shuffled minibatch SGD on its current shard. The server averages the resulting
parameters using each client's fraction of the total training examples. Model
weights carry across stage boundaries. The CLI uses one CPU thread and
deterministic PyTorch operations for repeatable runs in the same environment.

Each JSON line identifies the seed, one-based round, scenario, and condition.
`training_loss` is mean cross-entropy across all local example presentations
during ordinary FedAvg, measured before each minibatch update. After all updates,
`test_accuracy_by_condition` records the fraction classified correctly on each
encountered condition's fixed test set. `current_test_accuracy` is its entry
for the current condition. Ordinary training receives only client shards.
All three methods start with identical weights, shard contents, and ordinary
minibatch order for the same seed. Test evaluation is read-only and follows
the adaptation decision and any resulting update.

### Validation trigger and practice

After each ordinary FedAvg round, score the server model on the current
condition's fixed validation set. For `self_evolution`, compare that accuracy
with the previous round's **final** validation accuracy on that round's
condition, including any previous adaptation. A drop of at least
`validation_drop` (0.10 in the smoke configuration, or 10 percentage points)
triggers one adaptation pass. The first round establishes a reference for each
seed. At a transition the two validation sets represent different conditions,
so their relative difficulty can also affect this trigger.

On a trigger, the simulator generates `practice_examples` new labeled examples
using the current condition's known class means and unit Gaussian noise. It
rounds half the requested count to assign positive labels and shuffles the
labels before drawing the features. The smoke configuration gives 32 examples
of each class. A dedicated NumPy stream keyed by seed and round keeps practice
independent of client, validation, and test examples. Generation requires the
simulator's condition parameter and reads no observed examples.

The server performs one SGD pass over the generated data, with the ordinary
batch size and learning rate. For smoke this gives four updates of 16 examples
at learning rate 0.08. The resulting model goes to clients in the next round.
Each result records the reference, validation scores before and after adaptation,
the trigger flag, and separate example and optimizer-step counts for ordinary
training and practice. Ordinary examples are counted once per local epoch.
Each method uses 192 ordinary example presentations and 12 client SGD steps per
round. Every trigger adds 64 presentations and four server SGD steps.

### Continual learning with local replay and distillation

Each device owns a reservoir of up to `memory_capacity` labeled training examples
(16 in the smoke config). After its first completed local training round on each
new shard, the device considers every row of that shard for admission exactly once.
The reservoir fills the first 16 slots. For the nth example thereafter, it draws
a uniform integer from 1 through n and replaces that slot if it fits within the
capacity. This gives each observed example an equal chance of retention.
Each device uses a separate seeded random stream and stores independent row copies.
Only local training shards enter this code path. The server aggregates model
parameters weighted by current shard size. Raw examples stay on their device.

For each ordinary minibatch, the device samples up to `replay_per_batch` (4)
eligible examples uniformly without replacement. It excludes entries from the
current shard. A remembered example may be drawn again on a later SGD step.
Thus rounds 1–2 use ordinary FL alone. On return, examples from the initial
stable shard are eligible because the return shard is fresh.

At the start of each round, clients use a frozen, evaluation-mode copy of the
incoming server model as teacher. One SGD step minimizes mean labeled
cross-entropy over the combined current and remembered examples, plus
`distillation_weight` (0.5) times the mean KL divergence
`KL(teacher probabilities || student probabilities)` over remembered examples.
Both distributions use softmax at temperature 1. An empty replay batch gives
only the current-example loss. The reported `training_loss` remains the mean
current-example cross-entropy, comparable across methods.

Each continual-learning JSON row gives the capacity per device, post-round
occupancy by device, distinct remembered examples used in that round, student
replay presentations (including repeats), teacher forward-only presentations,
and extra optimizer steps. An example used for both supervision and distillation
counts as one student presentation and one teacher presentation. The shared
ordinary budget counts only current examples and their local SGD steps.

The two ideas adapted from Li et al.'s lifelong HyperKD work are exemplar
replay and distillation of an earlier model's knowledge. This experiment uses
device-local reservoir sampling and matches binary output probabilities. The
paper's cross-spectral–spatial feature distillation is specific to hyperspectral
imagery and outside the scope of this classifier. [Li et al., HyperKD](https://doi.org/10.1109/TGRS.2025.3552449)

### Extending the experiment

The loop in `cp493.experiment.records` accepts a dataset factory, a round-training
function, and a model factory. A dataset factory returns `ExperimentData` with
the input size, class count, and client, validation, and test splits keyed by
condition, with optional fresh training shards for the return stage. A paired
practice factory takes a scenario step, seed, round number, and example count
and generates practice data for the chosen dataset. A round-training function updates the supplied server model from
the current client shards and returns the mean local loss. To compare methods
on paired runs, keep the schedule, data seed, splits, and initial model fixed.

## Performance Evaluation

The framework will be evaluated based on:

- Model accuracy and training loss
- Catastrophic forgetting and knowledge retention
- Training time and computational cost
- Energy consumption and communication overhead
- Overall system reliability and efficiency

## Research Affiliation

**Wilfrid Laurier University**

Supervised by **Prof. Samuel Okegbile**.

**Project Status:** Under Development
