# TFP Experiment Checklist

Before reporting a TFP result:

1. Record the **Task ID**, model, action policy, training reward, observation mode, action mask, curriculum, training seed, and training steps.
2. Use a short **Experiment Tag** such as `baseline-seed7`, `tabux-seed7`, or `action-memory-tabux-seed7`.
3. Evaluate every compared method on the **same fixed map × seed suite**. The default Fox Transport benchmark is 18 unseen maps × 10 seeds = 180 episodes.
4. Record both the **checkpoint protocol** and the **evaluation protocol**. If reward or action mask is overridden during evaluation, report the override explicitly.
5. Compare more than success rate. TFP records path efficiency, collisions, invalid actions, periodic behavior cycles, interaction cycles, state revisit rate, model parameter count, and inference latency.
6. Use `<model>.pt` for the final training endpoint and `<model>.best.pt` for validation-selected weights. State which checkpoint role was evaluated.
7. Treat inference latency as **hardware-specific**. Compare timings only on the same machine/runtime and report the device.
8. For statistical claims, repeat training with multiple independent training seeds and report dispersion or confidence intervals.
9. Keep inference-only controllers such as Tabu/TabuX separate from PPO rollout training unless the experiment explicitly studies controller-assisted exploration.

Evaluation JSON files store runtime, checkpoint, reward, mask, benchmark, and experiment metadata automatically.
