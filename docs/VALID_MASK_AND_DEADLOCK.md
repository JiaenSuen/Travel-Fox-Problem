# Valid-Mask and Deadlock Notes

## Action-mask semantics

`task` applies geometric validity and forces the task-critical interaction when the agent is on cargo or, while carrying, on the goal. `valid` exposes all semantically valid choices without forcing interaction: the policy may move away from cargo instead of picking it up, or move away from the goal instead of delivering. Early off-goal `DROP` is not exposed because the environment classifies it as invalid.

This distinction matters experimentally. `task` supplies interaction supervision through the mask, whereas `valid` requires the policy to learn both navigation and interaction timing. Results from the two protocols should therefore be reported separately unless mask choice is the explicit independent variable.

## Deadlock hypothesis

Short LEFT↔RIGHT or UP↔DOWN cycles can arise from several interacting causes: local-observation aliasing, deterministic greedy action selection, reward shaping, and insufficient temporal context. The cycle-safe reward removes asymmetric progress/regress incentives; action memory and GRU state address temporal ambiguity; Tabu and TabuX act as inference-time cycle breakers. A useful anti-deadlock improvement should increase task success or path efficiency while also reducing cycle events and state revisit rate. A lower cycle count alone is not sufficient evidence of better task reasoning.
