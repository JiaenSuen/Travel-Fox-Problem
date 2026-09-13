# Local Transport — Small-POMDP Failure Analysis

## Abstract

Local Transport is the smallest TFP environment and the most mature controlled study. An agent with only local perception must find one cargo item, execute pickup, and deliver it to a destination. Although the task is geometrically simple, it exposes a recurring problem in compact RL: policies can learn useful navigation while still fail through short action loops, missed interaction timing, and state revisitation. The controlled reference results show that explicit behavioral memory and inference-time cycle control can materially change task success without increasing visual model capacity dramatically.

## Controlled evidence

The packaged controlled evaluation contains 180 episodes per completed model (18 test maps × 10 seeds).

| Model | Success | Pickup | Mean steps | Cycles | Revisit |
|---|---:|---:|---:|---:|---:|
| Simple CNN | 37.8% | 73.9% | 125.3 | 107.61 | 57.1% |
| Simple CNN + Tabu | 55.6% | 73.9% | 102.0 | 16.19 | 47.2% |
| Action-Memory CNN | 57.8% | 75.6% | 93.3 | 73.88 | 39.6% |
| Simple CNN + TabuX | **87.8%** | **93.3%** | 58.1 | 7.09 | 20.0% |
| Action-Memory CNN + TabuX | 86.1% | 91.7% | **57.2** | **6.67** | **19.0%** |

### Finding 1 — recent action history is a high-value compact memory signal

Action Memory improves success from 37.8% to 57.8%, a **+20.0 percentage-point** gain over the plain CNN. The important interpretation is not that action history is universally superior to recurrent visual memory. Rather, many observed Local Transport failures are themselves action-sequence phenomena: alternating directions, repeated backtracking, repeated pickup attempts, and cyclic local behavior. Encoding executed actions exposes loop phase directly and avoids asking a recurrent visual encoder to infer it indirectly from aliased local images.

### Finding 2 — a lightweight deployment controller can recover competence the policy already partly possesses

Basic Tabu raises the same feed-forward CNN from 37.8% to 55.6% and reduces mean cycle events from 107.61 to 16.19. TabuX produces the largest controlled gain: 87.8% success, **+50.0 points** over the plain CNN. This suggests that a large portion of the original failure rate was not inability to identify the task objective, but inability to exit locally recurrent behavior once the policy entered a bad basin.

For robotics, this is relevant because a tiny runtime recovery mechanism can sometimes be cheaper and easier to validate than enlarging the policy network. However, deployment control and learning quality must remain reported separately.

### Finding 3 — Action Memory and TabuX are not additive

Action Memory + TabuX reaches 86.1%, slightly below TabuX alone at 87.8%. This negative interaction is useful: once TabuX suppresses the dominant short-cycle failure mode, explicit action memory provides little additional success. The combination does reduce cycles/revisit marginally and has the lowest mean steps, but the success metric does not improve. This argues against stacking mechanisms simply because each works in isolation.

## Reward failure analysis

Early dense shaping treated progress and regression asymmetrically. That creates a generic reward-hacking risk: a two-step forward/backward loop can have non-negative or positive return even though net task progress is zero. Local Transport therefore evolved toward symmetric geodesic shaping and explicit interaction penalties.

The new experimental reward `004_phase_consistent_transport` is a **hypothesis**, not yet a controlled result. It separates four phases/events that were previously conflated: navigation progress, pickup phase transition, early drop, and repeated-state deadlock. Progress is rewarded symmetrically only for movement; milestone actions receive their own terms. Repeated state visits receive a small capped penalty rather than an ever-growing intrinsic signal.

### Proposed reward ablation

Compare the original default reward, cycle-safe shaping, interaction-aware shaping, and `004_phase_consistent_transport` using the same Action-Memory CNN, fixed mask/view, and multiple training seeds. Primary outcomes should be success and completion; diagnostic outcomes should include pickup rate, interaction cycles, revisit rate, return, and path efficiency. A successful reward revision should improve task success without merely increasing average return.

## Broader interpretation

Local Transport supports a compact robotics lesson: **failure representation should match failure timescale**. A five-to-ten-action history is sufficient for many oscillations; storing a long sequence of image embeddings may be unnecessary. Conversely, action memory does not solve long-distance spatial aliasing, which becomes dominant in larger room tasks. TFP therefore treats Action Memory as a targeted inductive bias rather than a universal memory architecture.
