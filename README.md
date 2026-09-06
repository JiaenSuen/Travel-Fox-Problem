# TFP — Traveling Fox Problems

**A compact PyTorch benchmark for partially observed transport, PPO memory, exploration, and anti-deadlock research.**

TFP is a small embodied-RL testbed for rapid controlled experiments before moving to larger simulators or physical robots. The agent receives a local observation, must locate cargo, execute `PICKUP`, navigate to the goal, and execute `DELIVER`. The benchmark is designed to expose perceptual aliasing, recurrent-memory limits, interaction-learning difficulty, exploration failures, and deterministic local policy cycles.

## Demo

![TFP transport demo](assets/demo.png)

*Example local-view transport episode rendered by the built-in visualizer.*

## Benchmark

| Component | Setting |
|---|---:|
| Task | `TFP-FoxTransport-Local` (`FOX-TR-L1`) |
| Observation | local `10×5×5` tensor |
| Actions | four moves + `PICKUP` + `DELIVER` |
| Training maps | 54 |
| Unseen test maps | 18 |
| Map sizes | 10×10, 15×15, 20×20 |
| Topology families | aisle, partition, blocks, mixed, rooms, zigzag |
| Formal seeds | 10 per map |
| Formal episodes | **180** |

Formal comparisons should use the same map × seed matrix, checkpoint role, action-mask protocol, and evaluation reward unless one of those variables is the intended ablation.

## Action masks

TFP supports two explicit mask protocols:

- **`task`** — geometric validity plus task supervision. On the cargo cell, `PICKUP` becomes the only available action. While carrying cargo on the goal, `DELIVER` becomes the only available action.
- **`valid`** — semantically valid actions without forced interaction. On cargo, the policy may `PICKUP` or move away. On the goal while carrying, the policy may `DELIVER` or move away. Early off-goal `DROP` is not exposed because the environment classifies it as invalid.

`task` and `valid` answer different research questions. `task` isolates navigation and memory under supervised interaction timing; `valid` requires the learned policy to solve both navigation and interaction timing. Training configuration is frozen at run start, and checkpoints preserve the mask used during training.

## Rewards

- `001_dense_transport` — dense transport baseline.
- `001_sparse_transport` — task-event reward with minimal shaping.
- `002_cycle_safe_transport` — symmetric geodesic shaping that makes immediate progress/regress loops net negative after step cost.
- `003_valid_interaction_transport` — valid-mask study reward with a small opportunity cost for leaving an available `PICKUP` or `DELIVER` interaction while preserving policy choice.

The current `valid` mask is semantically aligned with `step()`: off-goal drop is not exposed. This prevents a pickup/drop reward loop in which repeated pickup bonuses could compete with actual delivery.

## Active models

| ID | Model | Research role |
|---|---|---|
| 001 | `001_simple_cnn` | feed-forward visual PPO baseline |
| 002 | `002_simple_cnn_tabu` | baseline + short-cycle inference filter |
| 003 | `003_action_memory_cnn` | explicit finite action history |
| 004 | `004_simple_cnn_tabux` | baseline + local-basin TabuX escape |
| 005 | `005_action_memory_tabux_cnn` | action memory + TabuX |
| 006 | `006_ppo_gru_bootstrap` | CNN checkpoint → GRU residual fine-tuning |
| 007 | `007_ppo_gru_action_memory` | recurrent action-memory baseline |
| 008 | `008_ppo_gru_episodic_count` | GRU + episodic novelty |
| 009 | `009_ppo_gru_action_memory_tabux` | GRU action memory + TabuX |
| 010 | `010_ppo_gru_gobi` | GRU + compact GoBI-style exploration |

## Key model design notes

### `001_simple_cnn`

`001_simple_cnn` is the minimum learned visual policy and the main architectural control. Three lightweight convolutional stages encode the fixed local tensor into a compact feature, followed by independent PPO actor and value heads. The model has no recurrent state, explicit action history, intrinsic reward, or inference-time search controller. This makes it useful for separating representation capacity from protocol effects. Under `task`, interaction timing is effectively supplied by the mask; under `valid`, the same network must learn when to execute `PICKUP` and `DELIVER` from observation channels that indicate interaction availability. The architecture should therefore remain fixed when studying mask, reward, checkpoint, or evaluation changes. Any large change in its behavior is easier to attribute to the experimental protocol than to hidden temporal state. It also serves as the first regression model for training/evaluation consistency and the reference point for memory and anti-deadlock extensions.

### `002_simple_cnn_tabu`

`002_simple_cnn_tabu` keeps the trainable neural network identical to the Simple CNN and adds a small inference-only tabu mechanism. The controller tracks recent actions and detects short periodic motifs such as LEFT↔RIGHT or UP↔DOWN. When the greedy policy proposes an action that would continue a detected short cycle, that action is temporarily suppressed and the best remaining legal action is selected. PPO training is unchanged, so this branch isolates the effect of a deterministic deployment-time cycle breaker from improvements in representation learning. The mechanism is intentionally narrow: it can interrupt obvious two-action oscillations but has limited understanding of a broader local decision basin and may escape for one step only to return immediately. Tabu is therefore a low-complexity anti-deadlock baseline. If cycle count decreases without a corresponding improvement in success or path efficiency, the result suggests that the controller is suppressing a visible symptom rather than solving navigation or interaction reasoning.

### `003_action_memory_cnn`

`003_action_memory_cnn` tests whether a compact finite behavioral history can reduce local-observation aliasing without using a recurrent network. The local tensor is encoded by the lightweight CNN, while recent executed actions are embedded and summarized into a small history feature. Visual and action-history features are fused before the PPO actor and value heads. The exact history context used during rollout is stored and replayed during optimization, so shuffled PPO minibatches do not reconstruct a different temporal context from the one that generated the original action. This model is useful when two local scenes look similar but were reached through different recent trajectories. Its memory remains finite and explicit, making it easier to inspect than a GRU and preventing hidden state from accumulating arbitrarily long context. The branch therefore separates the question “does recent behavior provide useful state information?” from the stronger question “is learned recurrent state required for the task?”

### `004_simple_cnn_tabux`

`004_simple_cnn_tabux` combines the unchanged Simple CNN with a stronger inference-only escape controller. TabuX maintains a compact trace of recent behavior, detects repeated local basins rather than only immediate ABAB motifs, and applies temporary tabu tenure to actions or transitions that would continue or rapidly re-enter the basin. An aspiration rule can release a tabu choice when it becomes sufficiently preferable, preventing the controller from becoming permanently rigid. Because TabuX is applied during greedy evaluation rather than PPO optimization, the learned CNN and training objective remain directly comparable to the baseline. This makes the model useful both as an engineering safeguard and as a diagnostic probe. Improvement over `001_simple_cnn` indicates that part of the failure set is caused by deterministic local-policy traps rather than missing visual capacity. TabuX should not be interpreted as learned exploration: it can redirect an action sequence, but it does not learn task semantics, long-horizon planning, or a global navigation model.

### `007_ppo_gru_action_memory`

`007_ppo_gru_action_memory` is the primary learned temporal model. A compact CNN extracts the local visual feature, a GRU maintains recurrent state, and an explicit embedding of recent actions provides a direct short-term behavioral record. The two memory paths are complementary: the GRU can compress longer temporal context, while the action sequence does not need to be reconstructed implicitly inside hidden state. PPO optimization uses rollout-length truncated backpropagation through time and preserves the recurrent context that actually generated each action, with state reset at episode boundaries. The architecture directly targets perceptual aliasing caused by the 5×5 local view. However, recurrent memory does not guarantee escape from deterministic local cycles. A GRU can itself settle into a stable repeated trajectory if policy logits and reward make the loop locally self-consistent. `009_ppo_gru_action_memory_tabux` therefore keeps the learned recurrent architecture while adding an external TabuX controller to isolate residual inference-time deadlocks.

### `008_ppo_gru_episodic_count`

`008_ppo_gru_episodic_count` augments compact recurrent PPO with a lightweight episodic-count novelty signal. Within each episode, local observation signatures are counted, and rarely visited signatures receive a larger intrinsic bonus using an inverse-count style schedule. The intrinsic coefficient is annealed so that exploration pressure is strongest early and gradually yields to the external transport objective. This design targets repeated-state and local-basin behavior without introducing a separate trainable curiosity network, keeping the experiment small and interpretable. Its limitation is deliberate: novelty does not imply useful task progress. An agent may discover new local observations while still failing to execute `PICKUP`, reach the goal, or resolve observation aliasing. Counts may also merge globally different states that appear identical inside the local view. Episodic Count should therefore be evaluated jointly with success, path efficiency, cycle events, and state revisit rate rather than treating intrinsic reward magnitude or raw novelty as evidence of effective exploration.

## Deadlock analysis

Persistent LEFT↔RIGHT or UP↔DOWN behavior is best treated as a combined partial-observation, reward, and deterministic-policy problem. The 5×5 tensor can alias globally different locations, while greedy action selection turns small logit preferences into stable limit cycles. Asymmetric distance shaping can also make some progress/regress sequences artificially attractive; `002_cycle_safe_transport` removes that incentive through symmetric distance deltas. Memory addresses temporal ambiguity, whereas Tabu and TabuX directly target residual deterministic cycles. A credible anti-deadlock improvement should increase task success or path efficiency while also reducing cycle events and state revisit rate. A lower cycle count by itself is insufficient: a controller may merely replace one repeated behavior with another inefficient path. The benchmark therefore records task and behavior metrics together.

## Experimental results template

Fill the table only with results generated under a declared common protocol.

| Model | Train Reward | Train Mask | Eval Reward | Eval Mask | Success | Pickup | Mean Steps | Mean Return | Path Efficiency | Cycle Events | State Revisit | Inference ms |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `001_simple_cnn` |  |  |  |  |  |  |  |  |  |  |  |  |
| `002_simple_cnn_tabu` |  |  |  |  |  |  |  |  |  |  |  |  |
| `003_action_memory_cnn` |  |  |  |  |  |  |  |  |  |  |  |  |
| `004_simple_cnn_tabux` |  |  |  |  |  |  |  |  |  |  |  |  |
| `005_action_memory_tabux_cnn` |  |  |  |  |  |  |  |  |  |  |  |  |
| `006_ppo_gru_bootstrap` |  |  |  |  |  |  |  |  |  |  |  |  |
| `007_ppo_gru_action_memory` |  |  |  |  |  |  |  |  |  |  |  |  |
| `008_ppo_gru_episodic_count` |  |  |  |  |  |  |  |  |  |  |  |  |
| `009_ppo_gru_action_memory_tabux` |  |  |  |  |  |  |  |  |  |  |  |  |
| `010_ppo_gru_gobi` |  |  |  |  |  |  |  |  |  |  |  |  |

## TFP Studio

![TFP Studio](assets/tfp_studio.png)

TFP Studio provides **Train**, **Evaluate**, **Compare**, and **Research Plugins** views. Compare records the checkpoint protocol and the actual evaluation protocol separately:

- **PT Reward** / **Eval Reward**
- **PT Mask** / **Eval Mask**

This makes deliberate reward or mask overrides visible instead of silently mixing incompatible runs. Evaluation supports the final training endpoint (`last`) and validation-selected (`best`) checkpoint roles.

## Reproducibility

Checkpoints preserve task, model, policy, reward, action mask, seed, observation configuration, checkpoint role, and training configuration. Evaluation JSON files preserve the checkpoint protocol, evaluation protocol, hardware/runtime information, map × seed benchmark definition, per-episode records, and aggregate behavior metrics.

Large learned weights are ignored by Git through `*.pt`, `*.pth`, and `*.ckpt`. Only the five small packaged baseline reference checkpoints are explicitly allow-listed.

Experiment templates are available under:

- `experiments/anti_deadlock/`
- `experiments/valid_mask/`

## Repository structure

```text
Traveling-Fox-Problems/
├─ README.md
├─ tfp_studio.py
├─ assets/
├─ checkpoints/
├─ docs/
├─ experiments/
├─ reference_results/
├─ results/
├─ tests/
├─ tools/
└─ tfp/
   ├─ envs/
   ├─ evaluation/
   ├─ intrinsic/
   ├─ models/
   ├─ policies/
   ├─ rewards/
   ├─ tasks/
   ├─ training/
   └─ visualization/
```

## License

MIT. See [`LICENSE`](LICENSE).
