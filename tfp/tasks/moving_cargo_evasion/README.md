# Moving Cargo & Predator Avoidance

A dynamic local-POMDP transport task. The cargo begins on a carrier that circulates continuously on a closed cyclic track spanning most of the map; the agent must plan an interception, execute `PICKUP`, and deliver the cargo to a fixed destination. A white wolf independently roams the traversable map using random-waypoint motion, and any contact terminates the episode.

The observation remains 5×5 or 7×7 local vision, augmented with compact motion context: a two-value interception/delivery waypoint, carrier velocity and phase, and low-bandwidth wolf bearing/proximity telemetry. The task therefore isolates moving-target interception, interaction timing, temporal memory, and risk-aware navigation without exposing a global map.

**Models:** `001_intercept_safety_cnn`, `002_intercept_action_memory`, `003_intercept_gru_memory`  
**Reward:** `001_intercept_safety_potential`  
**Maps:** 45 train / 15 test across 12×12, 16×16, and 20×20 layouts.
