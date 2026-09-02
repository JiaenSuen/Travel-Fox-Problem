# TFP Model Plugin Guide

A TFP model plugin is a Python file placed in `tfp/models/` that defines:

```python
MODEL_SPEC = {...}

def create_model(observation_shape, action_count):
    return MyModel(observation_shape, action_count)
```

The environment observation contract stays unchanged. A model may remain stateless, own learnable action memory, or attach an inference-only controller.

## 1. Stateless model

Implement ordinary:

```python
def forward(self, x):
    return logits, value
```

`simple_cnn_001.py` is the minimal reference.

## 2. Learnable action-memory model

A model can keep compact action history without changing the environment tensor. `action_memory_cnn_003.py` demonstrates the supported hooks:

```python
act_forward(x, env_ids, update=True)
training_forward(x, context)
observe_action_memory(env_ids, actions, rewards, dones)
reset_action_memory(env_ids)
```

`act_forward` returns the exact context used at interaction time. PPO stores that context and supplies it to `training_forward` after minibatch shuffling, preventing temporal misalignment.

## 3. Inference-only controller

A deployment controller can rewrite greedy actions without contaminating PPO rollout data:

```python
rewrite_actions(proposed, logits, mask, env_ids)
observe_behavior(env_ids, actions, rewards, dones)
reset_behavior_memory(env_ids)
```

`simple_cnn_tabu_002.py` is the original periodic-action example. `simple_cnn_tabux_004.py` and `tabux_controller.py` demonstrate a stronger Tabu Search interpretation with short-term behavior-state/transition tenure and aspiration.

## 4. Learnable memory + inference controller

`action_memory_tabux_cnn_005.py` composes both mechanisms. The neural action-memory branch is trained normally; TabuX only affects greedy evaluation/deployment. This separation is useful for controlled ablations between learned temporal context and handcrafted deployment priors.

## Checkpoints

TFP writes:

```text
<model>.pt       # best validation checkpoint; default export/evaluation
<model>.last.pt  # final training endpoint
```

The unseen test suite does not select the best checkpoint.
