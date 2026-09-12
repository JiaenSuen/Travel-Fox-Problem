# Keyed Multi-Cargo Logistics

`KEYED-HAZARD-LOGISTICS` studies long-horizon logistics under **partial observability, symbolic access constraints, multi-goal ordering, and dynamic safety risk**. An agent operates in a multi-room environment, acquires persistent red/green/blue keycards to unlock colored access frontiers, transports 1–3 cargo items one at a time to a shared destination, and avoids two independently roaming wolf hazards. Maps span Medium through Large+++ and vary room geometry, door placement, access depth, cargo placement, and hazard trajectories across seeds.

The task is intentionally stronger than a sequential key-door puzzle. Colored locks control complete room-graph frontiers, so the dependency cannot be bypassed through another corridor. Capacity-one transport then forces repeated planning through already unlocked regions, while two stochastic hazards make the shortest path potentially unsafe. The policy remains local (5×5 or 7×7); it receives only compact progress, key inventory, hazard telemetry, and a next-subgoal waypoint rather than a global map or full plan.

## Models

| Model | Research role |
|---|---|
| `001_dependency_film_shield` | FiLM-conditioned local perception with symbolic dependency state and a bounded soft safety/navigation prior. |
| `002_event_memory_transformer` | Transformer memory over compact task events and hazard telemetry instead of expensive image-history recurrence. |
| `003_dual_timescale_gru_shield` | Separate long-timescale task memory and short-timescale hazard memory, plus explicit action history. |

## Reward

`001_dependency_risk_potential` uses executable subgoal-distance shaping, sparse key/pickup/delivery milestones, symmetric action-caused risk improvement, and anti-cycle penalties. Wolf motion after the agent action is excluded from dense safety credit, preventing passive hazard movement from generating free reward. Door interaction receives only a small acknowledgement; useful door opening is rewarded primarily by reducing executable cost, so repeated toggle cycles are net-negative.

## Environment

- **Scale:** Medium, Large, Large+, Large++, Large+++ (`19×19` to `35×35`).
- **Mission:** 1–3 cargo items, capacity one, one shared destination.
- **Access:** persistent colored keycards and locked room-graph frontiers.
- **Hazards:** two seeded roaming wolves; contact terminates the episode.
- **Actions:** move, pickup, drop, toggle door, wait.
- **Observation:** local 5×5 or 7×7, 37 channels.
- **Evaluation:** 20 held-out layouts × 5 default seeds = 100 episodes per controlled model evaluation.
