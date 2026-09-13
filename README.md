# TFP — A Lightweight POMDP Benchmark for Memory-Aware Reinforcement Learning in Partially Observable Robotic Tasks

**A lightweight PyTorch benchmark for reinforcement learning under local perception, memory, interaction, exploration, moving targets, and dynamic hazards.**

![TFP Studio](assets/demo.png)

TFP is organized as task-scoped research environments sharing one PPO/evaluation stack. Each task owns its maps, models, rewards, experiment presets, checkpoints, and result namespace. Experiments use deterministic map × seed evaluation and support 5×5 / 7×7 local perception.

## Tasks

| Code | Task | Train / test maps | View | Research focus |
|---|---|---:|---|---|
| `LOCAL-TRANSPORT` | **Local Transport** | 54 / 18 | 5×5, 7×7 | exploration, short cycles, pickup/delivery timing, compact memory |
| `COLOR-SORT` | **Color-Matched Sorting** | 45 / 15 | 5×5, 7×7 | multi-object state, goal conditioning, target switching |
| `ROOM-DOOR-TRANSPORT` | **Multi-Room Door Transport** | 50 / 20 | 5×5, 7×7 | long-horizon navigation, room memory, stateful doors |
| `MOVING-CARGO-EVASION` | **Moving Cargo & Predator Avoidance** | 72 / 24 | 5×5, 7×7 | predictive interception, complex route topology, temporal dynamics, hazard avoidance |
| `KEYED-HAZARD-LOGISTICS` | **Keyed Multi-Cargo Logistics** | 50 / 20 | 5×5, 7×7 | access dependencies, multi-goal logistics, long-horizon memory, dynamic safety |

### Local Transport

Single-cargo transport across 10×10, 15×15, and 20×20 layouts. The agent observes only a local crop and must locate cargo, execute `PICKUP`, and deliver it to the destination. The task is intentionally compact enough for controlled ablations while still exposing perceptual aliasing, exploration failure, interaction loops, and deployment-time deadlocks. It is the primary environment for Action Memory, GRU policies, Tabu/TabuX, GoBI-style exploration, bootstrap recurrent training, and reward-shaping studies. The controlled benchmark retains the historical default reward; `004_phase_consistent_transport` is provided as a new reward-study variant that separates navigation shaping from pickup/drop phase transitions and repeated-state penalties.

### Color-Matched Sorting

Each episode contains 2–5 colored objects and matching destinations. Capacity is one, so the policy repeatedly switches between object selection, pickup, color-conditioned routing, and delivery. Layouts span 8×8, 10×10, and 12×12 maps. The task isolates multi-stage completion, target conditioning, interaction timing, and recurrent memory without the topological complexity of the larger room environment.

### Multi-Room Door Transport

A long-horizon transport problem over 15×15 to 31×31 layouts with 4–16 rooms. Cargo and destination locations vary by seed and doors may begin open or closed. Closed doors require `TOGGLE_DOOR`. The policy receives local perception plus a compact next-doorway route cue; it does not receive the global map or complete route. The default reward uses executable action cost, so crossing a closed door counts as `TOGGLE + MOVE` and irrelevant door toggles receive no standalone benefit.

### Moving Cargo & Predator Avoidance

A dynamic transport POMDP in which cargo remains on a moving carrier until interception. The carrier follows an explicit cyclic route program over a visible rail network, while a white wolf independently roams the traversable map; contact is terminal. The packaged benchmark contains 72 training and 24 test layouts across 12×12, 16×16, 20×20, and 24×24 scales. Six motion topologies—ring, serpentine, figure-eight, clover, switchyard, and nested-loop—introduce turns, shared junctions, repeated junction visits, and route self-intersections. Spatial structures vary independently through open, slalom, room, block, corridor, and island families.

Local 5×5/7×7 perception is augmented only with compact motion telemetry: interception/delivery waypoint, carrier velocity and cadence phase, short-horizon carrier position, interception slack, upcoming turn proximity, and wolf bearing/observed velocity. `WAIT` supports interception timing. The policy does not receive the global map or complete route program.

### Keyed Multi-Cargo Logistics

A partial-observable multi-room logistics problem over Medium through Large+++ layouts (19×19 to 35×35). Colored doors form one-to-three access frontiers; matching persistent keycards must be acquired before deeper regions become reachable. Each episode contains 1–3 cargo items, a shared destination, capacity-one transport, and two independently roaming wolves. Cargo/key/goal locations and hazard trajectories vary by seed. `WAIT` enables risk-aware timing, while local 5×5/7×7 perception prevents direct global planning.

The task is designed to study **hierarchical dependency reasoning, multi-goal ordering, event memory, and safe long-horizon control** in one compact grid benchmark. A low-bandwidth waypoint exposes only the next required doorway or active subgoal; the policy never receives the room graph, complete route, global map, or future hazard paths.

## Task-specific advanced baselines

### Dynamic interception

The moving-cargo task shares one counterfactual reward across three architectures, isolating how temporal state and spatial-temporal fusion affect interception and hazard avoidance.

| Model | Architecture | Research role |
|---|---|---|
| `001_horizon_film_shield` | gated FiLM visual encoder + multi-horizon residual safety prior | strong feed-forward baseline using current/future carrier geometry, route phase, interception slack, and predictive hazard state |
| `002_cross_attention_dynamics` | eight-step telemetry Transformer + spatial cross-attention | lets recent carrier/wolf dynamics query local map features directly, targeting junction ambiguity and hazard-conditioned route choice |
| `003_phase_world_gru` | carrier GRU + hazard GRU + action memory + phase gating | factorizes periodic target dynamics from stochastic hazard dynamics and retains longer temporal state |

**Counterfactual Intercept-Risk Potential.** `001_counterfactual_intercept_risk` uses courier-caused mission progress, symmetric action-caused safety change, and local safety regret. The regret term compares the chosen action with the safest locally feasible alternative under the same pre-motion hazard state. Carrier/wolf motion after the action cannot create free dense reward; `WAIT` remains net-costly, missed interaction windows are penalized, and predator contact is terminal.

### Access-constrained multi-goal logistics

`KEYED-HAZARD-LOGISTICS` turns multi-room transport into a long-horizon planning problem. Colored keycards unlock complete room-graph access frontiers rather than isolated doors, preventing trivial bypasses. Each episode contains 1–3 cargo items with capacity one and two roaming hazards, so successful policies must coordinate symbolic access, repeated delivery trips, memory, and risk-aware route selection.

| Model | Architecture | Research role |
|---|---|---|
| `001_dependency_film_shield` | local CNN + symbolic FiLM + bounded dependency/safety prior | feed-forward baseline that explicitly conditions perception on key inventory, active subgoal, cargo progress, and hazard state |
| `002_event_memory_transformer` | compact ten-step event/state Transformer | retains access, delivery, and hazard transitions without recurrently storing image frames |
| `003_dual_timescale_gru_shield` | long-timescale task GRU + short-timescale hazard GRU + action memory | separates persistent mission state from rapidly changing safety state for long-horizon POMDP control |

**Predictive Dependency-Risk Potential.** `002_predictive_hazard_potential` combines executable subgoal progress with sparse key, cargo, delivery, and completion milestones, then adds motion-aware safety credit assignment. Observed wolf velocity and closing rate support one-step predictive risk, local safety regret penalizes avoidably dangerous choices, and near misses provide a pre-collision learning signal. Exogenous wolf motion cannot create free dense safety reward. The earlier `001_dependency_risk_potential` remains available for ablation.

## Benchmark results

### LOCAL-TRANSPORT

<!-- TFP-BENCHMARK:LOCAL-TRANSPORT:START -->
| Model | Params | Runs | N | Success ↑ | Completion ↑ | Pickup ↑ | Steps ↓ | Return ↑ | Path Eff. ↑ | Collision ↓ | Invalid ↓ | Cycles ↓ | Interact cycles ↓ | Revisit ↓ | Infer ms ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `001_simple_cnn` | 121.8k | 1 | 180 | 37.8% | 37.8% | 73.9% | 125.3 | 5.42 | 0.996 | 0.00 | 0.00 | 107.61 | 0.00 | 57.1% | 0.277 |
| `002_simple_cnn_tabu` | 121.8k | 1 | 180 | 55.6% | 55.6% | 73.9% | 102.0 | 6.74 | 0.835 | 0.00 | 0.00 | 16.19 | 0.00 | 47.2% | 0.328 |
| `003_action_memory_cnn` | 132.4k | 1 | 180 | 57.8% | 57.8% | 75.6% | 93.3 | 6.94 | 0.988 | 0.00 | 0.00 | 73.88 | 0.00 | 39.6% | 0.731 |
| `004_simple_cnn_tabux` | 121.8k | 1 | 180 | 87.8% | 87.8% | 93.3% | 58.1 | 9.66 | 0.828 | 0.00 | 0.00 | 7.09 | 0.00 | 20.0% | 0.458 |
| `005_action_memory_tabux_cnn` | 132.4k | 1 | 180 | 86.1% | 86.1% | 91.7% | 57.2 | 9.49 | 0.881 | 0.00 | 0.00 | 6.67 | 0.00 | 19.0% | 0.905 |
| `006_ppo_gru_bootstrap` | 167.4k | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `007_ppo_gru_action_memory` | 141.5k | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `008_ppo_gru_episodic_count` | 133.3k | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `009_ppo_gru_action_memory_tabux` | 141.5k | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `010_ppo_gru_gobi` | 133.3k | — | — | — | — | — | — | — | — | — | — | — | — | — | — |

*Evaluation: 18 test maps × 10 seeds = 180 episodes/run · `local` 5×5 · `001_dense_transport` · `task` mask · `001_ppo_categorical`. `—` = pending.*
<!-- TFP-BENCHMARK:LOCAL-TRANSPORT:END -->

![LOCAL-TRANSPORT benchmark](results/LOCAL-TRANSPORT/summary.png)

### COLOR-SORT

<!-- TFP-BENCHMARK:COLOR-SORT:START -->
| Model | Params | Runs | N | Success ↑ | Completion ↑ | Pickup ↑ | Steps ↓ | Return ↑ | Path Eff. ↑ | Collision ↓ | Invalid ↓ | Cycles ↓ | Interact cycles ↓ | Revisit ↓ | Infer ms ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `001_color_cnn` | 125.3k | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `002_goal_conditioned_cnn` | 233.8k | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `003_goal_gru_action_memory` | 144.9k | — | — | — | — | — | — | — | — | — | — | — | — | — | — |

*Evaluation: 15 test maps × 5 seeds = 75 episodes/run · `local` 5×5 · `001_dense_color_sort` · `task` mask · `001_ppo_categorical`. `—` = pending.*
<!-- TFP-BENCHMARK:COLOR-SORT:END -->

![COLOR-SORT benchmark](results/COLOR-SORT/summary.png)

### ROOM-DOOR-TRANSPORT

<!-- TFP-BENCHMARK:ROOM-DOOR-TRANSPORT:START -->
| Model | Params | Runs | N | Success ↑ | Completion ↑ | Pickup ↑ | Steps ↓ | Return ↑ | Path Eff. ↑ | Collision ↓ | Invalid ↓ | Cycles ↓ | Interact cycles ↓ | Revisit ↓ | Infer ms ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `001_route_prior_cnn` | 393.2k | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `002_route_prior_action_memory` | 422.3k | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `003_route_prior_gru_memory` | 520.8k | — | — | — | — | — | — | — | — | — | — | — | — | — | — |

*Evaluation: 20 test maps × 5 seeds = 100 episodes/run · `local` 7×7 · `001_actionable_geodesic` · `task` mask · `001_ppo_categorical`. `—` = pending.*
<!-- TFP-BENCHMARK:ROOM-DOOR-TRANSPORT:END -->

![ROOM-DOOR-TRANSPORT benchmark](results/ROOM-DOOR-TRANSPORT/summary.png)

### MOVING-CARGO-EVASION

<!-- TFP-BENCHMARK:MOVING-CARGO-EVASION:START -->
| Model | Params | Runs | N | Success ↑ | Completion ↑ | Pickup ↑ | Steps ↓ | Return ↑ | Path Eff. ↑ | Collision ↓ | Invalid ↓ | Cycles ↓ | Interact cycles ↓ | Revisit ↓ | Infer ms ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `001_horizon_film_shield` | 821.0k | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `002_cross_attention_dynamics` | 225.4k | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `003_phase_world_gru` | 975.0k | — | — | — | — | — | — | — | — | — | — | — | — | — | — |

*Evaluation: 24 test maps × 5 seeds = 120 episodes/run · `local` 7×7 · `001_counterfactual_intercept_risk` · `task` mask · `001_ppo_categorical`. `—` = pending.*
<!-- TFP-BENCHMARK:MOVING-CARGO-EVASION:END -->

![MOVING-CARGO-EVASION benchmark](results/MOVING-CARGO-EVASION/summary.png)

### KEYED-HAZARD-LOGISTICS

<!-- TFP-BENCHMARK:KEYED-HAZARD-LOGISTICS:START -->
| Model | Params | Runs | N | Success ↑ | Completion ↑ | Pickup ↑ | Steps ↓ | Return ↑ | Path Eff. ↑ | Collision ↓ | Invalid ↓ | Cycles ↓ | Interact cycles ↓ | Revisit ↓ | Infer ms ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `001_dependency_film_shield` | 808.5k | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `002_event_memory_transformer` | 929.8k | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `003_dual_timescale_gru_shield` | 992.0k | — | — | — | — | — | — | — | — | — | — | — | — | — | — |

*Evaluation: 20 test maps × 5 seeds = 100 episodes/run · `local` 7×7 · `002_predictive_hazard_potential` · `task` mask · `001_ppo_categorical`. `—` = pending.*
<!-- TFP-BENCHMARK:KEYED-HAZARD-LOGISTICS:END -->

![KEYED-HAZARD-LOGISTICS benchmark](results/KEYED-HAZARD-LOGISTICS/summary.png)

## Key research techniques

**Action memory.** TFP encodes a short history of executed actions separately from visual features. For local POMDPs, many failures are defined by behavior sequence rather than appearance: left/right oscillation, repeated backtracking, missed interaction timing, or repeated waiting. A compact action history therefore supplies loop phase and recent control context directly, without requiring a recurrent network to reconstruct those signals from image features. In the packaged Local Transport reference, the Action-Memory CNN improves controlled success over the plain CNN baseline while remaining much smaller than a general recurrent visual policy.

**Tabu / TabuX.** Tabu controllers are inference-time anti-deadlock mechanisms. Basic Tabu detects short repeated action cycles and suppresses the action that would continue them. TabuX extends this to repeated local state-action basins, allowing the agent to escape persistent corner, corridor, or pickup-area loops. Because the controller operates after neural inference and is disabled during PPO rollout collection, learning quality and deployment-time cycle handling can be evaluated separately.

**Reward design.** Task-owned rewards make assumptions explicit. Dense rewards use symmetric geodesic or actionable progress so reversing a move does not create net positive shaping. Multi-Room Door Transport prices closed-door traversal as an additional action rather than paying a direct door bonus. Moving Cargo adds counterfactual local safety regret so avoidable unsafe actions are distinguishable from unavoidable risk. Keyed Logistics extends this idea with observed wolf velocity, closing rate, predictive risk improvement, near-miss penalties, and explicit failure-cause export. Local Transport also includes a phase-consistent reward-study variant that separates navigation progress from pickup/drop phase transitions.

**Structured residual priors and soft safety shields.** The larger room, moving-target, and keyed-logistics tasks use low-bandwidth task cues as bounded action-logit priors while PPO learns residual corrections. Moving-target priors combine multi-horizon interception with predictive hazard suppression; keyed logistics combines the next access/subgoal waypoint with a soft two-hazard shield. This shifts capacity toward interaction, temporal state, dependency reasoning, and recovery without exposing a global map, complete path, or oracle action sequence.

**GoBI-style exploration.** `GoBI Compact` combines episodic novelty with a small latent reachability model. It evaluates whether limited model-based imagination can improve local exploration while remaining practical for resource-constrained agents. The imagination budget is deliberately small so the intrinsic module stays an ablation component rather than becoming the dominant world model.

**Bootstrap recurrent training.** `PPO-GRU Bootstrap` initializes a recurrent policy from a mature feed-forward encoder and actor/value heads. The recurrent residual is introduced gradually before full fine-tuning. This reduces destructive optimization when temporal capacity is added to an already useful visual policy and provides a controlled baseline for recurrent-memory studies.

## Academic discussion

`academic_discussion/` preserves the research record behind the benchmark: controlled findings, failed approaches, reward-design revisions, model rationale, threats to validity, and proposed ablations. Claims are labeled as **controlled results**, **development observations**, or **hypotheses** so short debugging runs are not confused with formal benchmark evidence.

Start with [`academic_discussion/README.md`](academic_discussion/README.md). The Local Transport discussion analyzes the existing 180-episode references; the Keyed Logistics discussion focuses on predator-driven failures and safety-specific evaluation.

## Task-scoped architecture

```text
tfp/tasks/<task>/
├── maps/train + maps/test
├── models/
├── rewards/
└── environment.py

experiments/<TASK-CODE>/
results/<TASK-CODE>/
checkpoints/<TASK-CODE>/
videos/<TASK-CODE>/
```

The trainer and evaluator instantiate environments through the task registry, so new tasks do not require task-specific PPO code.

## TFP Studio

```bash
python tfp_studio.py
```

Select **Task → Model → Reward → view size → action mask**, then train or evaluate. Once training starts, the complete protocol form is locked and the frozen configuration is identified by a protocol fingerprint. **Pause / Resume** suspends rollout collection without changing the run configuration; **Graceful Stop** writes a separate interrupted checkpoint. **Compare** can filter records by task and benchmark scope.

Evaluation display supports windowed or fullscreen rendering (`F11` toggles fullscreen, `Esc` returns to windowed mode). Video export supports selectable FPS and writes constant-frame-rate H.264 MP4 with a reset frame, terminal hold, fade, and explicit `EPISODE COMPLETE` outro so the final simulator state is shown before container EOF.

To rebuild result tables and summary PNGs:

```bash
python tools/rebuild_reports.py
```
