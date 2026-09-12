# Local Transport

A compact single-cargo local-POMDP for studying exploration, interaction timing, short behavior cycles, and memory under 5×5/7×7 perception. Maps span 10×10, 15×15, and 20×20 structural families with 54 training and 18 test layouts.

The task is the primary ablation environment for feed-forward CNNs, executed-action memory, GRU policies, Tabu/TabuX inference control, GoBI-style intrinsic exploration, bootstrap recurrent training, and dense/cycle-safe/interaction-aware reward variants. Six actions are used: four-neighbor movement, `PICKUP`, and `DROP`.
