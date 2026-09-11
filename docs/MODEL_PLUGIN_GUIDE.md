# TFP Task Plugin Guide

TFP now scopes **models and rewards by task**. Shared policies and intrinsic modules remain framework-level components.

```text
tfp/tasks/<task_package>/
├── __init__.py      # TaskSpec registration
├── environment.py   # optional task-owned environment
├── maps/
├── models/
└── rewards/
```

## Register a task

A `TaskSpec` declares the task ID/code, environment class, train/test maps, supported local-view sizes, default model, and default reward. The PPO trainer calls `create_task_env(...)`; it does not import a concrete task environment.

## Models

Place experiment-facing model plugins in `tfp/tasks/<task>/models/`. A plugin defines `MODEL_SPEC` and `create_model(observation_shape, action_count)`. Stateless models may implement `forward`; stateful models can use the existing rollout/training memory hooks.

Shared neural building blocks may remain under `tfp/models/`, but model choices exposed to Studio are discovered only from the selected task.

## Rewards

Place task rewards in `tfp/tasks/<task>/rewards/`. A plugin defines `REWARD_SPEC` and `create_reward()`. Environments load rewards with their own task ID, preventing an unrelated task reward from appearing in the selector.

## Policies / intrinsic modules

`tfp/policies/` and `tfp/intrinsic/` remain shared because they operate on the generic action/model interfaces. The default PPO policy is `001_ppo_categorical`.

## Action-mask contract

All current tasks support:

- `task`: required `PICKUP` / valid delivery interaction is forced when available.
- `valid`: movement alternatives remain available, so interaction timing must be learned.

## Observation contract

Both packaged tasks support **5×5 and 7×7** local observations. Channel count is task-specific and is stored in checkpoints.

## Results

Evaluation writes to `results/<TASK-CODE>/`. Each evaluation regenerates `README.md` and `summary.png` in that folder from its JSON records, so root documentation can embed a stable path without manually copying metrics.
