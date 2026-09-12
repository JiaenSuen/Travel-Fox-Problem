# Color-Matched Sorting

A multi-object local-POMDP transport task with **2–5 colored objects** and matched destinations. The agent carries one item at a time and must repeatedly select the correct pickup, navigate under 5×5/7×7 perception, and deliver to the color-consistent goal. Packaged layouts span 8×8, 10×10, and 12×12 maps with disjoint 45 / 15 train-test layouts.

| Model | Research role |
|---|---|
| `001_color_cnn` | compact color-aware visual baseline |
| `002_goal_conditioned_cnn` | explicit target/carry conditioning for multi-stage routing |
| `003_goal_gru_action_memory` | recurrent + action-history model for target switching and interaction timing |

The default `001_dense_color_sort` reward combines symmetric geodesic progress, pickup/delivery milestones, completion reward, step cost, and invalid-action cost. Additional reward plugins support milestone and valid-interaction ablations.
