# FOX-CS-L2 — Fox Color Sort · Local

A lower-topology-complexity multi-object extension of TFP. Each episode contains **2–5 colored objects** and the same number of color-matched destinations. The fox has capacity one and must repeatedly `PICKUP` an object and `DROP` it on the matching goal under a 5×5 or 7×7 local observation.

## Experimental strategies

| Strategy | Model | Reward / mask | Main question |
|---|---|---|---|
| S1 | `001_color_cnn` | `001_dense_color_sort` + `task` | Can color-separated local features + target cues solve the task without memory? |
| S2 | `002_goal_conditioned_cnn` | `001_dense_color_sort` + `task` | Does explicit pooled target/carry context improve multi-color routing? |
| S3 | `003_goal_gru_action_memory` | `003_valid_interaction_color_sort` + `valid` | Can recurrent/action memory learn both routing and interaction timing under stronger partial observability? |

The training curriculum exposes 8×8 → 10×10 → 12×12 layouts; object count expands from 2–3 to 2–4 to 2–5. Train/test layouts are disjoint: **45 / 15**.

## Reward design

`001_dense_color_sort` uses a step cost, symmetric geodesic progress shaping, pickup reward, per-item delivery reward, completion bonus, and invalid-action penalty. `002_milestone_color_sort` reduces dense shaping to test reward dependence. `003_valid_interaction_color_sort` adds opportunity costs for ignoring available pickup/delivery actions under the `valid` mask.

## Solvability check

`tools/validate_color_sort_task.py` runs a full-state shortest-path diagnostic only to verify environment correctness. Current packaged validation: **75/75 successful episodes**, 2–5 items represented, mean path efficiency **0.973**. This oracle is not used by PPO and is not a learning baseline.

A short end-to-end PPO integration smoke run (`002_goal_conditioned_cnn`, 32,768 steps, 30 training layouts) reached **0.333 success** and **0.569 mean completion** on a 6-map × 2-seed unseen subset. The record is packaged under `reference_results/FOX-CS-L2/` and is explicitly marked non-formal.
