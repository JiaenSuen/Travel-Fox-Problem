# TFP Plugin Guide

TFP discovers Python plugins directly from `tfp/models`, `tfp/policies`, `tfp/rewards`, and `tfp/intrinsic`. Numeric prefixes define experiment ordering, for example `001_simple_cnn.py`, `009_ppo_gru_action_memory_tabux.py`, and `003_valid_interaction_transport.py`.

## Models

A model plugin defines `MODEL_SPEC` and `create_model(observation_shape, action_count)`. Stateless models may implement only `forward`. Stateful models can additionally expose rollout/training context hooks. Recurrent PPO preserves rollout chronology with per-environment truncated BPTT.

`006_ppo_gru_bootstrap` initializes from the packaged Simple CNN checkpoint. `009_ppo_gru_action_memory_tabux` combines recurrent GRU memory, explicit action history, and inference-only TabuX control. Active model IDs are contiguous `001`–`010`.

## Policies

`001_ppo_categorical` is the default masked PPO policy. `002_ppo_cycle_guard` changes the effective legal-action mask during sampling and returns that exact mask to the rollout buffer so PPO log-probability reconstruction remains on-policy.

Model-owned Tabu/TabuX controllers are deployment/evaluation mechanisms: they rewrite greedy inference actions only and do not alter PPO training rollouts.

## Rewards

- `001_dense_transport` — dense transport baseline.
- `001_sparse_transport` — sparse task events.
- `002_cycle_safe_transport` — symmetric geodesic movement shaping.
- `003_valid_interaction_transport` — valid-mask reward with cycle-safe movement shaping and a small penalty for ignoring an available `PICKUP`/`DELIVER` opportunity.

## Intrinsic modules

Two compact intrinsic modules are included:

- `001_episodic_count`
- `001_gobi_compact`

## Action-mask contract

Only `task` and `valid` are supported. `task` forces `PICKUP` on cargo and `DELIVER` on the goal while carrying. `valid` keeps legal movement alternatives available on those cells, so interaction timing remains a learned policy decision. Early off-goal `DROP` is invalid and is not exposed by the valid mask.

Training protocol is immutable once a run begins. Studio disables protocol-critical selectors while the worker is active and the trainer logs the frozen mask mode.

## Observation contract

`TFP-FoxTransport-Local` is local-only and emits a fixed `10×5×5` observation by default.
