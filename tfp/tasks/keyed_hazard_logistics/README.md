# Keyed Multi-Cargo Logistics

`KEYED-HAZARD-LOGISTICS` studies long-horizon logistics under **partial observability, symbolic access constraints, multi-goal ordering, and dynamic safety risk**. An agent acquires persistent red/green/blue keycards to unlock colored room-graph frontiers, transports 1–3 cargo items one at a time to a shared destination, and avoids two independently roaming wolf hazards. Maps span Medium through Large+++ and vary room geometry, door placement, access depth, cargo placement, and hazard trajectories across seeds.

Colored locks control complete access frontiers, preventing trivial corridor bypasses. Capacity-one transport forces repeated planning through already unlocked regions, while the moving hazards make the shortest path potentially unsafe. The policy remains local (5×5 or 7×7) and receives compact mission state, a next-subgoal waypoint, and observed hazard telemetry rather than a global map or future hazard path.

## Models

| Model | Research role |
|---|---|
| `001_dependency_film_shield` | FiLM-conditioned local perception with symbolic dependency state and a bounded motion-aware safety/navigation prior. |
| `002_event_memory_transformer` | Transformer memory over compact mission events and hazard telemetry instead of recurrent image history. |
| `003_dual_timescale_gru_shield` | Separate long-timescale task memory and short-timescale hazard memory, plus explicit action history. |

## Reward and hazard observation

The default `002_predictive_hazard_potential` retains executable subgoal progress and sparse key/cargo/delivery milestones, then adds **observed-motion safety shaping**. The 43-channel observation includes current wolf bearing/distance, observed velocity, and closing rate. Reward uses agent-caused predictive-risk improvement, local safety regret, and a near-miss penalty before terminal collision. Wolf motion after the action cannot generate free safety credit. `001_dependency_risk_potential` remains available for ablation.

Evaluation exports `hazard_collisions`, `near_misses`, and `failure_reason` so predator-driven failures can be separated from timeout, navigation, and task-logic failures.

## Environment

- **Scale:** Medium, Large, Large+, Large++, Large+++ (`19×19` to `35×35`).
- **Mission:** 1–3 cargo items, capacity one, one shared destination.
- **Access:** persistent colored keycards and locked room-graph frontiers.
- **Hazards:** two seeded roaming wolves; contact terminates the episode.
- **Actions:** move, pickup, drop, toggle door, wait.
- **Observation:** local 5×5 or 7×7, 43 channels.
- **Evaluation:** 20 held-out layouts × 5 default seeds = 100 episodes per controlled model evaluation.

See [`academic_discussion/05_keyed_hazard_logistics.md`](../../../academic_discussion/05_keyed_hazard_logistics.md) for the complete safety-failure analysis and proposed ablations.
