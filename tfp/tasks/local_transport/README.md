# Local Transport

A compact single-cargo local-POMDP for studying exploration, interaction timing, behavioral cycles, and low-cost memory under 5×5/7×7 perception. Maps span 10×10, 15×15, and 20×20 structural families with 54 training and 18 test layouts.

The task is the primary controlled ablation environment for feed-forward CNNs, executed-action memory, GRU policies, Tabu/TabuX inference control, GoBI-style intrinsic exploration, bootstrap recurrent training, and reward shaping. Six actions are used: four-neighbor movement, `PICKUP`, and `DROP`.

## Reward studies

The controlled historical benchmark retains `001_dense_transport` so previous results remain comparable. New experiments can use `004_phase_consistent_transport`, which separates movement progress from pickup/drop phase changes, applies symmetric geodesic shaping, penalizes early drop and missed interaction, and adds a capped repeated-state penalty. This variant is intentionally an ablation candidate rather than a replacement for historical results.

See [`academic_discussion/01_local_transport.md`](../../../academic_discussion/01_local_transport.md) for quantitative findings and reward-design analysis.
