# Multi-Room Door Transport

A long-horizon local-POMDP in which one seeded cargo item and one destination are placed in different rooms. Layouts span **15×15 to 31×31**, from 4 to 16 rooms, with variable room geometry, door positions, and seeded door states. Closed doors require the explicit `TOGGLE_DOOR` action.

The policy receives a 5×5/7×7 crop plus a compact 2-D **route doorway waypoint**. It does not receive the global map, room graph, complete route, or next action. Three models isolate structured route fusion, compact action memory, and recurrent room-state memory: `001_route_prior_cnn`, `002_route_prior_action_memory`, and `003_route_prior_gru_memory`.

The single `001_actionable_geodesic` reward measures executable action cost: crossing a closed door costs `TOGGLE + MOVE`. Useful door operations therefore gain progress only by reducing path cost; irrelevant toggles receive no standalone reward. This keeps navigation, interaction, and anti-cycle behavior aligned under one reward definition.
