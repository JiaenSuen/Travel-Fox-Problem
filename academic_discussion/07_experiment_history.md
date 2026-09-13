# Experiment History and Negative-Result Ledger

This document preserves development observations that motivated the current TFP design. Unless a result is explicitly identified as controlled, the numbers and outcomes below are **historical development observations** from intermediate versions and should not be used as final benchmark comparisons.

## 1. Early Local Transport / Traveling-Fox behavior

The initial local-POMDP experiments established that a policy could learn basic transport competence on small maps, but test behavior was unstable. In several runs the agent reached the pickup region and then alternated forward/backward or left/right actions instead of executing or completing the interaction. Corner and corridor deadlocks were also common.

This changed the research question. The main bottleneck was not simply visual recognition or global exploration; it was **behavioral recurrence under aliased local observations**.

## 2. Recurrent memory was not an automatic solution

Multiple recurrent variants were explored:

- staged CNN → GRU/LSTM training;
- PPO with action memory plus LSTM/GRU;
- GRU action memory;
- GRU with episodic-count exploration.

Some intermediate runs reached reasonable training or quick-test success while still oscillating at pickup in deterministic rollout. For example, historical GRU/action-memory experiments reached roughly 0.75 train/quick success in one configuration but still failed through pickup-zone oscillation; an episodic-count GRU reached roughly 0.77 training and 0.44 quick-test success while preserving the same qualitative failure. A later GRU action-memory configuration produced about 0.617 test success.

These are not comparable controlled results, but they establish an important negative finding: **adding recurrent capacity did not directly encode the task logic needed to escape short interaction loops**. Memory capacity and useful memory representation are different problems.

## 3. Intrinsic exploration did not fix task-logic deadlocks

RND, ICM, DEIR, episodic count, and a GoBI-inspired direction were all considered or prototyped during exploration studies. Some improved visitation or training activity, but none reliably removed the dominant pickup/deadlock behavior in the configurations tested. Several were later removed from the main comparison surface to keep the project focused; a compact GoBI-style module remains as an explicit exploration ablation.

The failure is conceptually consistent: novelty bonuses encourage reaching unfamiliar states, but a pickup-area oscillation can occur **after the correct region has already been discovered**. Exploration reward is therefore poorly matched to an interaction-control failure.

## 4. Action Memory as targeted state augmentation

Action Memory was introduced to make recent control history explicit. This provided the policy with the phase of a short behavioral loop without requiring the visual encoder to reconstruct it from repeated local frames. The current controlled Local Transport study supports this direction: Action-Memory CNN improves success by 20.0 percentage points over the plain CNN.

However, it does not eliminate all loops. The controlled results show that TabuX provides a substantially larger anti-deadlock improvement, and combining Action Memory with TabuX is not additive. The interpretation is therefore targeted rather than universal: action history is useful state augmentation, while explicit runtime cycle handling can still be more effective for a narrow recurring failure.

## 5. Masking instability and protocol control

Historical experiments compared `task`, `valid`, and unmasked behavior. Unmasked control performed poorly, while some `valid`-mask runs also produced unexpectedly low success. During development there was concern that the displayed or active mask mode could change across training stages, making comparisons difficult to interpret.

The current system resolves this as a reproducibility issue rather than a model issue. `PPOConfig` is frozen, the action-mask mode is copied into a run-local snapshot, the Studio form is locked while training is active, and checkpoints store both the full configuration and a protocol fingerprint. Pause/Resume cannot mutate the configuration. Future mask comparisons should therefore be intentionally separate runs.

## 6. Door reward hacking in Multi-Room transport

An early door reward paid a positive opening bonus and a smaller closing penalty. Because structural distance also ignored the additional action required by a closed door, a policy could obtain misleading return by toggling doors. This was replaced by actionable geodesic cost: a closed doorway is priced as `TOGGLE + MOVE`, allowing a necessary opening to be valued through reduced task cost rather than an independent bonus.

This was one of the clearest examples in TFP where a nonzero training reward concealed zero task success.

## 7. Route-prior strength and PPO residual learning

During Multi-Room development, a strong route prior quickly produced useful unseen-small behavior but PPO updates could later overwrite the prior and reduce deterministic evaluation success. This showed that a good heuristic prior is not automatically stable when added to an unconstrained learned residual.

The current design bounds the residual/prior relationship. The research interpretation is not that the prior should dominate the policy, but that **optimization should not destroy a useful structured initialization before the learned component has evidence to replace it**.

## 8. Moving Cargo: from chase behavior to interception

The first moving-cargo formulation risked teaching the agent to chase the carrier's current cell. On cyclic tracks this can produce perpetual pursuit. The task was reframed around earliest feasible interception and later extended with nontrivial track networks, carrier phase, and explicit short-horizon dynamics.

Model design correspondingly moved away from a simple CNN / action-memory / GRU sequence toward architectures that factor target dynamics and hazard dynamics. This is another instance of failure-driven representation design.

## 9. Keyed Logistics: structural success but hazard-control failure

The fifth task introduced access-frontier key dependencies, multiple cargo trips, and two wolves. Structural validation consistently shows that the symbolic task can be completed if hazards are ignored. Development observation, however, identified wolf collision as the dominant real policy failure.

The current revision therefore adds observed wolf velocity, closing rate, predictive local risk, safety regret, near-miss accounting, and explicit predator/timeout failure rates. This is a shift from "penalize the terminal collision more" to **make dangerous approach observable and assign credit before contact**.

## 10. Research lessons from failed approaches

The project history supports five working lessons:

1. More memory is not equivalent to the right state representation.
2. Exploration bonuses cannot repair every control or interaction failure.
3. Dense reward must be tested for profitable zero-progress cycles.
4. Heuristic priors need bounded interaction with learned residual policies.
5. Benchmark infrastructure—immutable masks, seeds, task namespace, and failure-specific metrics—is part of the scientific method, not only software engineering.

These lessons should guide future TFP experiments and should remain visible even when later models outperform the historical baselines.
