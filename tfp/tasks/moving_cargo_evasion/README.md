# Moving Cargo & Predator Avoidance

`MOVING-CARGO-EVASION` is a dynamic-interception POMDP. Cargo remains on a moving carrier until pickup; the carrier follows a deterministic cyclic route program over a visible rail network, while a roaming wolf introduces stochastic collision risk. Route topology and scene geometry vary independently across 72 training and 24 held-out maps at 12×12, 16×16, 20×20, and 24×24 scales.

The environment includes ring, serpentine, figure-eight, clover, switchyard, and nested routes. Junction reuse and self-intersections make carrier position alone insufficient to infer motion phase. The policy therefore receives local perception plus compact motion telemetry rather than a global map or full route. `WAIT` is available for interception timing but remains costly.

## Models

| Model | Research role |
|---|---|
| `001_horizon_film_shield` | Multi-horizon carrier guidance and predictive hazard shielding FiLM-conditioned into local visual perception. |
| `002_cross_attention_dynamics` | Eight-step compact telemetry Transformer whose temporal state cross-attends local spatial tokens at junctions and hazards. |
| `003_phase_world_gru` | Factorized recurrent carrier/hazard state estimators with action memory and phase-conditioned residual control. |

## Reward

`001_counterfactual_intercept_risk` combines courier-caused interception/delivery progress with symmetric predictive safety shaping and **local safety regret**. Safety regret compares the selected action with the safest locally feasible action under the same pre-motion hazard state; exogenous wolf motion is excluded. This distinguishes unavoidable risk from avoidable unsafe control. Pickup and delivery provide sparse milestones, missed interaction windows are penalized, waiting is only partially refunded, and wolf contact remains the dominant terminal cost.

## Evaluation

Controlled evaluation uses 24 held-out layouts × 5 default seeds = 120 episodes per model with local 7×7 perception, task action masking, and the shared reward above.
