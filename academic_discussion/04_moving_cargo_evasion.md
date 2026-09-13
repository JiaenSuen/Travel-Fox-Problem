# Moving Cargo & Predator Avoidance — Dynamic Interception Under Local Risk

## Problem

The moving-cargo task introduces two coupled nonstationary elements: cargo is carried along a cyclic rail route and a roaming wolf creates terminal collision risk. Track topology includes rings, serpentine paths, figure-eight structures, clover patterns, switchyards, and nested loops. The task is therefore not ordinary point-goal navigation. A successful policy must reason about **where the cargo will be when the robot arrives**, not only where it is now.

## Interception formulation

A naive policy can chase the carrier's current position forever. TFP instead derives a compact feasible-interception cue from observed carrier phase and route dynamics. The cue is low-bandwidth and does not expose the full route program. This transforms the learning problem from unstructured moving-target pursuit into residual control around a meaningful rendezvous estimate.

The current three baselines separate dynamic reasoning in different ways: multi-horizon FiLM conditioning, spatial cross-attention from recent telemetry, and factorized recurrent state for periodic carrier dynamics versus stochastic hazard dynamics. These are intended as architectural hypotheses rather than claims of superiority until controlled results are available.

## Counterfactual safety shaping

The current reward `001_counterfactual_intercept_risk` separates agent-caused progress from environment-caused motion. If the carrier moves closer after the robot acts, that change alone should not be interpreted as robot competence. Similarly, a wolf wandering away should not create free safety reward.

A local safety-regret term compares the chosen action with feasible alternatives under the same observed pre-motion hazard state. This provides a gradient for avoidable risk before a collision occurs. It is still a heuristic and not a formal safety guarantee.

## Development history

Earlier versions used simpler rectangular cyclic tracks and models whose primary difference was action memory versus GRU. Those designs underrepresented the real problem: junctions, route phase, and separate carrier/hazard dynamics. The later track-network generator deliberately introduces self-intersections and repeated junction visits, while temporal models represent motion histories directly.

Short pipeline tests have demonstrated that the models can train and preserve a usable interception prior on held-out layouts, but these tests are explicitly **not controlled benchmark results**.

## Proposed analysis

Formal evaluation should stratify results by track topology and spatial map family. Useful outcome variables include interception success, time-to-pickup, delivery success after pickup, predator failure rate, near misses, waits, and path efficiency. A model that improves success only by waiting excessively would be exposed by pickup latency and episode length.

The central research question is whether explicit factorization of target dynamics and hazard dynamics provides a better sample/parameter trade-off than generic temporal memory. This is particularly relevant to compact robotics where the full image sequence may be too expensive to retain.
