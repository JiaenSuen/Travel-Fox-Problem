# TFP Experiment Checklist

Before reporting a TFP result:

1. Record the **Task ID**, model, action policy, reward function, observation mode, action mask, curriculum, training seed, and training steps.
2. Use a short **Experiment Tag** such as `baseline-seed7`, `tabux-seed7`, or `action-memory-tabux-seed7`.
3. Evaluate every compared method on the **same fixed map × seed suite**. The Fox Transport default is 18 unseen maps × 10 seeds = 180 episodes.
4. Compare more than success rate. TFP records path efficiency, collisions, invalid actions, periodic behavior cycles, interaction cycles, state revisit rate, model parameter count, and inference latency.
5. Report the canonical `<model>.pt` **best validation checkpoint**. Retain `<model>.last.pt` only for learning-dynamics/debug analysis.
6. Treat inference latency as **hardware-specific**. Compare timings only on the same machine/runtime and report the device.
7. For statistical claims, repeat training with multiple independent training seeds and report dispersion or confidence intervals.
8. Keep inference-only controllers, such as Tabu/TabuX, separate from PPO rollout training unless the experiment explicitly studies controller-assisted exploration.

Evaluation JSON files contain runtime, PyTorch, CUDA, checkpoint, configuration, and experiment metadata automatically.
