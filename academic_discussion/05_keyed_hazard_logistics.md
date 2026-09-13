# Keyed Multi-Cargo Logistics — Access-Constrained Planning Under Dynamic Hazards

## Abstract

Keyed Multi-Cargo Logistics is the most demanding current TFP task. An agent with local perception must acquire persistent colored keycards, unlock room-graph access frontiers, transport 1–3 cargo items one at a time to a shared destination, and avoid two independently moving wolves. The environment combines long-horizon symbolic dependency, repeated route execution, local visual aliasing, and stochastic safety risk. The current development observation is that **wolf contact is the dominant practical failure mode**, motivating a redesign of hazard observation, reward, and evaluation rather than another increase in collision penalty alone.

## Why structural solvability is insufficient

The map generator is validated so that key, door, cargo, and goal dependencies are structurally reachable when stochastic wolves are ignored. This answers only one question: does a legal task plan exist? It does not imply that a learned local policy can execute the plan safely while two hazards move.

This distinction matters academically. A 100% structural-solvability diagnostic can coexist with a low policy success rate if the policy repeatedly collides with predators, times out while avoiding them, or fails to remember access state. The revised evaluator therefore records hazard collisions, near misses, timeout rate, and explicit failure reason in addition to success.

## Failure hypothesis: collision penalty is too late

A terminal collision penalty supplies a strong signal only after the critical mistake has already occurred. Under partial observability, the causal action may have been taken several steps earlier—for example entering a narrow doorway while a wolf is approaching from the opposite room. Repeated terminal penalties can teach general conservatism, but they do not explain which locally available action was safer at the decision point.

The v9 safety revision therefore changes **information and credit assignment**, not just reward magnitude.

## Observed hazard motion

The observation now retains per-wolf observed velocity and a closing-rate feature derived from consecutive visible/telemetry states. The policy receives no future RNG, wolf target, or global path. A one-step linear extrapolation is used only as a compact predictive feature for reward/prior calculations.

This makes two local scenes distinguishable: a wolf at distance three moving away and a wolf at distance three rapidly closing. Position-only telemetry cannot represent that distinction.

## Predictive hazard reward

`002_predictive_hazard_potential` retains dependency-aware mission progress but adds three pre-collision terms:

1. **Predictive risk improvement** — rewards an agent action that moves the robot toward a lower estimated one-step hazard risk before stochastic wolf motion occurs.
2. **Local safety regret** — penalizes the selected destination when a locally feasible alternative was substantially safer under the same observed hazard state.
3. **Near-miss penalty** — records and penalizes transitions that end adjacent to a wolf even when no collision occurs.

The environment's subsequent wolf movement cannot produce free dense safety credit. This is important because otherwise a robot could receive positive reward simply because the hazard randomly moved away.

## Motion-aware soft prior

All three keyed-logistics models now share a stronger bounded local safety prior using wolf bearing, observed velocity, and closing rate. It remains a **soft action-logit bias**, not a hard action mask: PPO can override it. This is deliberate because an always-forbidden action can make narrow passages impossible to traverse and can hide the true capability of the policy.

The three architectures still test distinct memory hypotheses: FiLM-conditioned feed-forward control, compact event-memory attention, and dual-timescale recurrent state. In the dual-GRU model, persistent mission/access state and fast hazard dynamics are represented separately because their useful time constants are different.

## Safety metrics and proposed experiments

The next controlled study should report at minimum:

- task success and cargo completion rate;
- predator failure rate and mean hazard collisions;
- near misses per episode;
- timeout failure rate;
- key acquisition and first-cargo pickup rate;
- mean steps and path efficiency;
- results stratified by map scale and cargo count.

The critical ablation is **position-only hazard state vs velocity/closing telemetry vs predictive reward vs both**. A genuine safety improvement should reduce predator failures without collapsing task completion or producing extreme waiting behavior.

## Limitations

The one-step velocity extrapolation assumes locally consistent motion and cannot predict abrupt random-waypoint changes. Risk is defined on a discrete grid and does not model braking distance or robot footprint. The soft prior has no formal safety certificate. Reward weights may trade efficiency for caution differently across map sizes. These limitations should be treated as experimental variables rather than hidden behind aggregate success.

## Robotics interpretation

Despite its abstraction, the task reflects a common small-robot decision problem: a robot must complete logistics objectives while access state and moving hazards evolve faster than a global planner can be blindly followed. The intended research contribution is not a new safe-RL theorem; it is an interpretable benchmark for studying how compact temporal state, predictive local risk, and long-horizon task memory interact under limited compute.
