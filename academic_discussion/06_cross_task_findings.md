# Cross-Task Findings and Research Agenda

## 1. Failure representation should match failure timescale

A consistent TFP pattern is that adding generic memory is less useful than representing the variable that actually drives failure. Local Transport benefits from action history because oscillations are short behavioral sequences. Multi-room transport benefits from room/doorway context because the ambiguity is topological. Moving Cargo requires carrier and hazard motion histories. Keyed Logistics combines persistent symbolic access state with fast hazard dynamics, motivating separate memory timescales.

**Working principle:** choose the smallest temporal state that makes the dominant failure mechanism observable before increasing generic recurrent capacity.

## 2. Reward should measure executable progress

Several development failures came from rewarding proxies that were easier to optimize than the mission itself. Asymmetric forward/back shaping can reward oscillation. Fixed positive door bonuses can reward toggling. Current moving-target position can reward chasing rather than interception. Hazard movement can create free safety reward if measured after the environment changes.

The later TFP tasks therefore increasingly define potential over executable action cost and isolate the effect of the agent action before exogenous dynamics. This does not guarantee policy invariance because TFP also uses milestones and penalties, but it gives a strong engineering test: a reversible zero-progress action cycle should not have positive expected shaped reward.

## 3. Runtime recovery and learned competence are different axes

Local Transport demonstrates that TabuX can recover large amounts of success from an otherwise weak feed-forward policy. This is valuable for robotics, but it also warns against reporting only final deployed success. A deployment controller may hide a policy failure. TFP therefore preserves policy architecture, reward, masking, and runtime controller identity as distinct experiment factors.

## 4. Priors can rescue sample efficiency but can also dominate learning

Long-horizon tasks became trainable only after introducing low-bandwidth route/interception priors. During development, however, an overly strong prior could be degraded by PPO residual learning or could suppress useful exploratory corrections. Bounding the residual/prior interaction was therefore necessary.

This yields an open question: when does a structured prior act as useful inductive bias, and when does it become a hidden planner? TFP's design constraint is that the prior may expose a next waypoint or local risk direction, but not the global map, complete path, or oracle action sequence. Ablations with prior disabled are still required to quantify its contribution.

## 5. Safety must be evaluated as a distribution of failure causes

Binary success is inadequate when dynamic hazards exist. A policy may fail because it cannot find a key, because it times out through over-caution, or because it reaches the right room and collides with a wolf. The fifth task therefore motivates explicit failure-reason and near-miss reporting. This same idea should extend to Moving Cargo.

## 6. Negative results worth preserving

The project history contains several useful negative results:

- adding recurrent memory alone did not automatically remove pickup-zone oscillation;
- multiple intrinsic-reward variants did not reliably fix task-logic deadlocks;
- a direct door bonus admitted reward hacking;
- an unbounded/overpowering route prior could be degraded or could dominate PPO learning;
- Action Memory and TabuX were not additive in controlled Local Transport success;
- structural solvability of a generated task does not imply policy solvability under stochastic hazards.

Keeping these results is important because they narrow the causal story. TFP should not evolve by deleting every failed design from the record.

## 7. Suggested publication-style experiment sequence

A compact research paper based on TFP could organize experiments in three stages. First, establish Local Transport mechanisms with multiple training seeds: action memory, Tabu/TabuX, and reward ablations. Second, test transfer of the "match representation to failure" principle on room and moving tasks. Third, use Keyed Logistics as a safety-focused stress test, reporting predator failure and near-miss rates in addition to task completion.

For statistical reporting, use multiple independent training seeds, confidence intervals or bootstrap intervals over per-seed evaluation means, and effect sizes rather than a single best checkpoint. Keep architecture parameter count and inference latency visible because the project specifically targets compact robotics.

## 8. Open research questions

1. Can compact event memory outperform recurrent image memory at equal parameter and latency budgets?
2. Does actionable/potential-style reward shaping retain its benefit when route priors are removed?
3. At what hazard density does a soft safety prior become insufficient and require a hard constraint or shield?
4. Can a single small world model support both moving-target prediction and hazard avoidance without losing interpretability?
5. Do policies trained with 5×5 perception generalize more robustly than 7×7 policies because they rely less on layout-specific visual context, or simply learn more slowly?
6. Which TFP mechanisms transfer to a physical differential-drive platform with noisy depth/perception and actuator latency?
