# Color-Matched Sorting — Multi-Object Goal Switching

## Problem

Color-Matched Sorting extends single-cargo transport to 2–5 colored objects and matching destinations with capacity one. The geometric layouts are intentionally simpler than the large-room tasks so that the main difficulty is not maze solving: the policy must maintain which object is active, condition delivery on object color, switch goals after each successful delivery, and avoid repeatedly servicing an already completed pair.

There is not yet a complete controlled benchmark for the three registered models; all performance claims in this document are therefore hypotheses or development observations unless explicitly stated otherwise.

## Research hypothesis

The task is a small test of **goal-conditioned memory under repeated phase changes**. A visual policy that only recognizes nearby objects can perform well locally yet fail globally because identical-looking corridors correspond to different current goals. The minimum useful state is not a full map; it is a compact representation of carrying state, active color, completed colors, and recent task events.

## Model rationale

`001_color_cnn` provides a feed-forward baseline and measures how much of the task is solvable from immediate visual/context features. `002_goal_conditioned_cnn` explicitly conditions visual features on task state and should reduce confusion after pickup. `003_goal_gru_action_memory` adds temporal memory for target switching and interaction history.

The comparison is scientifically useful only if the reward and action-mask protocol remain fixed. Otherwise a more permissive mask or denser target signal can masquerade as a memory improvement.

## Reward design questions

Dense progress should be calculated relative to the **current valid subgoal**: before pickup, the selected cargo; after pickup, its same-color destination. At a delivery boundary the potential function changes, so the reward should treat delivery as an explicit milestone rather than subtracting two incompatible distance potentials. A poor implementation can create an artificial negative jump after a successful delivery or a positive jump when the active target changes.

An informative ablation would compare (1) sparse milestone-only reward, (2) goal-conditioned geodesic shaping, and (3) the same shaping with repeated-target penalties. The central question is whether dense shaping improves sample efficiency without teaching a brittle handcrafted target-selection policy.

## Proposed analysis

Formal training should report full-completion rate in addition to binary success, because delivering four of five objects is meaningfully different from delivering none. Per-color or per-item completion order can reveal whether the policy develops a stable sorting strategy or simply succeeds on whichever object it encounters first. Additional diagnostics should include target switches, duplicate interaction attempts, time-to-first-delivery, and residual undelivered count.

## Robotics relevance

The environment abstracts multi-object fetch-and-place tasks where a robot repeatedly updates its mission state after each completed delivery. The core mechanism—conditioning local perception on a compact goal state—is directly relevant to resource-constrained mobile manipulation even though TFP omits continuous grasping and perception uncertainty.
