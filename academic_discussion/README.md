# TFP Academic Discussion

This directory records the scientific reasoning behind TFP as a compact reinforcement-learning and robotics benchmark. It is intentionally separate from the root README: the root document describes the software and benchmark interface, while the documents here preserve hypotheses, negative results, design revisions, quantitative observations, threats to validity, and proposed experiments.

## Evidence labels

Every claim should be interpreted using one of three evidence levels.

- **Controlled result** — produced by the packaged test-map × seed protocol and suitable for direct cross-model comparison under the stated configuration.
- **Development observation** — produced during debugging, short training, smoke tests, or targeted diagnostics. Useful for forming hypotheses, but not a benchmark result.
- **Hypothesis / proposed analysis** — a design expectation that still requires a controlled experiment.

This distinction is essential because TFP has evolved through repeated environment, reward, and model revisions. A short sanity run can demonstrate that a training path is functional, but it does not establish generalization or statistical significance.

## Research questions

TFP asks a narrow robotics question: **how much task competence can a compact policy recover from local perception when failures are driven by partial observability, interaction timing, long-horizon dependencies, dynamic objects, and safety hazards?** The environments are deliberately small enough for ablation studies while retaining failure modes that resemble mobile-robot decision making: local aliasing, deadlocks, doors, pickup/drop sequencing, moving targets, access constraints, and collision risk.

The project therefore emphasizes four reusable research axes: representation of temporal context, reward shaping aligned with executable actions, deployment-time recovery mechanisms, and task-specific low-bandwidth priors that do not expose a global plan.

## Documents

| File | Scope |
|---|---|
| `00_methodology_and_reproducibility.md` | experimental protocol, PPO stack, reproducibility, metrics, evidence policy |
| `01_local_transport.md` | controlled findings on action memory, Tabu/TabuX, loops, and reward design |
| `02_color_sort.md` | multi-object goal conditioning and target-switching hypotheses |
| `03_room_door_transport.md` | door reward failure, actionable geodesics, route priors, long-horizon navigation |
| `04_moving_cargo_evasion.md` | moving-target interception, temporal dynamics, hazard-aware control |
| `05_keyed_hazard_logistics.md` | access dependencies, multi-cargo planning, two-wolf safety failures and mitigation |
| `06_cross_task_findings.md` | cross-task findings, negative results, design principles, research agenda |
| `07_experiment_history.md` | historical trials, failed approaches, mask/reward debugging, design evolution |
| `REFERENCES.md` | literature used to contextualize TFP design decisions |

## Reproducibility rule

A training run uses an immutable `PPOConfig` snapshot. TFP Studio locks training controls after launch and exposes only Pause/Resume and Graceful Stop. The checkpoint stores the complete training configuration and a protocol fingerprint. Pausing stops environment sampling without changing the optimizer or protocol; graceful stopping saves an `.interrupted.pt` checkpoint instead of overwriting the canonical run checkpoint.

Evaluation exports raw episode records in addition to aggregate summaries. For safety-oriented tasks, records include hazard collisions, near misses, and failure reason so that success rate is not the only outcome examined.
