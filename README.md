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

## Running the four methods

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
.venv/bin/python -m cp493.experiment --config configs/smoke.toml --method combined --output results/combined.jsonl
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
All four methods start with identical weights, shard contents, and ordinary
minibatch order for the same seed. Test evaluation is read-only and follows
the adaptation decision and any resulting update.

### Validation trigger and practice

After each ordinary FedAvg round, score the server model on the current
condition's fixed validation set. For `self_evolution` and `combined`, compare that accuracy
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

### Combined round

The `combined` method uses the same settings as the two individual mechanisms.

1. Freeze a copy of the incoming server model. Each client starts from those
   same weights and performs its ordinary SGD steps with local replay and
   distillation. Only that client's observed training shard enters its reservoir.
2. Aggregate client weights with FedAvg, weighted by current shard size.
3. Evaluate the aggregated model on the current validation set. Compare this
   accuracy with the previous round's final validation accuracy. Round 1
   establishes the reference.
4. On a drop of at least 10 percentage points in the smoke configuration,
   generate 64 practice examples and take four server SGD steps. Recheck
   validation once. The resulting server weights start the next round.
5. Score the final model on the fixed test set of each encountered condition
   and write the round record. On return, the current and original scores
   refer to the same test set.

Each combined row includes the trigger, practice counts, distinct replayed
examples, and memory occupancy per device. `extra_training_presentations` sums
student replay and server practice examples. `extra_optimizer_steps` counts
the extra practice steps. Replay shares the ordinary SGD steps. The row also
records replay and teacher forward calls in batches, teacher presentations in
examples, and the extra validation examples scored after practice. These
inference counts are separate from training presentations.

### Paired synthetic pilot

From the repository root, run the pilot and regenerate its summaries with

```sh
.venv/bin/python -m unittest discover -s tests -v
/usr/bin/time -p .venv/bin/python -m cp493.pilot run
.venv/bin/python -m cp493.pilot summarize
```

The default configuration is `configs/pilot.toml`. Only the seed list
(493 through 497) and round count differ from the smoke configuration.
All four methods use identical initial weights, ordinary shuffle streams,
client shards and held-out examples within each seed.

| Rounds | Distribution and evaluation |
| --- | --- |
| 1–10 | Stable at 0°, evaluated on the original fixed test set. |
| 11–20 | Sudden jump to 120°, evaluated on original and shifted fixed sets. |
| 21–30 | One 6° step per round from 114° through 60°. Each angle has its own training shards and fixed validation/test sets. Each round adds its test set to the earlier ones. |
| 31–40 | Return to 0° with fresh client training shards shared across these ten rounds. The original validation/test sets are reused, and all twelve test sets remain in the record. |

Each `results/pilot/<method>-<seed>.json` contains the method, active seed,
full pilot configuration and 40 ordered records. These include trigger
inputs and decisions, practice examples and updates, replay and teacher
presentations and forward calls, and current, original and all encountered
test accuracies. Work counts are per round.

`paired_seeds.csv` preserves each seed's checkpoint scores and differences
from FL on that same seed. `summary.csv` gives their means and sample standard
deviations as fractions. Checkpoints are rounds 10, 11, 20, 31 and 40.
The gradual checkpoint first averages rounds 21–30 within each run. Its
current score therefore spans ten different test conditions, while its original
score always uses the same set within a seed. The five seeds are the units
of variation. Scores follow the scheduled training and any triggered practice.

### Evaluating the saved pilot

From the repository root, use the existing environment and the 20 saved run
files in `results/pilot`. The evaluation command reads the run files, validates
their identities, round order and test conditions, and writes to `results/evaluation`.

```sh
.venv/bin/python -m pip install -r requirements-evaluation.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m cp493.evaluate_pilot
```

**Retention and forgetting.** For each method and seed, retention is test
accuracy on that seed's fixed original test set in every round (1–40).
Let `A[r]` be that accuracy as a fraction. The reference is the same run's
`A[10]`. For rounds 11–30 the drop in percentage points is
`D[r] = 100 × max(0, A[10] − A[r])`. The two run metrics are `D[20]` and
the maximum of the twenty values `D[11]` through `D[30]`. Both the clamp and
the maximum are applied within a seed before summarizing across seeds.

**Recovery.** The pilot threshold is 70%. Read the `shifted` test accuracy in
each record for rounds 11–20. Throughout this stage it equals current-condition
accuracy and uses the same 256-example test set within each seed. If round
`r` is the first score at or above 0.70, recovery took `r − 10` completed
shifted rounds (1–10). A single crossing qualifies. A run that misses the
threshold is marked **not reached within the shifted stage**. Recovery-time
means and sample SDs use only observed crossings. The separate
`recovered_in_stage` indicator (1 or 0) includes all five seeds.

**Extra work.** Sum the recorded practice examples, extra practice SGD steps,
student replay presentations, teacher batch forward calls, teacher example
presentations, and extra post-practice validation presentations separately
over all 40 rounds. A repeated presentation counts each time. Memory capacity
is the fixed number of training examples one device can retain, with zero for
methods without a reservoir. Capacity is reported once per run.

| Output in `results/evaluation/` | Contents |
| --- | --- |
| `metrics_by_seed.csv` | Forgetting, recovery, and work for each method. |
| `trajectories_by_seed.csv` | Current and original accuracy (%) by round, plus forgetting (percentage points) in rounds 11–30. |
| `current_accuracy.png`, `original_accuracy.png`, `forgetting.png` | Four method means with an aligned panel showing between-seed sample SD. Dashed vertical lines mark stage boundaries. |
| `recovery_and_work.png` | Individual seed observations, means and sample SDs, with separate axes for each unit and a column for missed recoveries. |

Both tables include seeds 493–497 individually, the observed count, the mean,
and the sample SD (denominator `n − 1`). The `delta_seed_*` columns subtract
standard FL **in the same seed**. The paired mean and sample SD summarize
those differences. For recovery time, a pair requires a crossing in both
runs. Missing differences are blank, as are means with zero observations
and SDs with fewer than two. Read the recovery times alongside the attainment
counts because the set of observed seeds can differ between methods.

During rounds 21–30 the current-condition accuracy curve evaluates a
different test set at each round. Its movement reflects both model changes
and differences between the test conditions. The original-condition curve
uses a fixed reference throughout.

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
