# Multi-Room Door Transport — Long-Horizon Navigation and Reward Alignment

## Problem

Multi-Room Door Transport expands the transport task to 4–16 rooms and introduces stateful doors. The policy must navigate under local perception, open a closed door when required, remember enough topological context to avoid revisiting rooms indefinitely, find cargo, and deliver it across a potentially different set of rooms.

## Historical failure: reward-positive door behavior

A major early development failure was a reward design in which opening a door received a positive bonus while closing it received a smaller penalty. This made repeated door toggling potentially profitable. At the same time, structural distance treated a closed doorway as if it were already traversable, so opening a genuinely necessary door did not consistently appear as task progress. The policy could therefore optimize return without learning the intended interaction semantics.

This was a **development observation**, not a controlled benchmark finding, but it produced an important design rule: interaction rewards should be derived from changes in executable task cost whenever possible.

## Actionable geodesic potential

The revised `001_actionable_geodesic` reward prices movement in action space. Traversing an open doorway costs a movement action; traversing a closed but unlockable doorway requires `TOGGLE + MOVE`. Opening the correct door therefore reduces remaining executable cost naturally. Opening an irrelevant door provides little or no progress, and closing a useful door reverses that gain. This greatly reduces the need for standalone door bonuses.

## Route prior as an inductive bias

A second difficulty was that a vector toward the final cargo or destination can point directly through walls. For a local policy this cue is ambiguous and often encourages wall-following or oscillation. The current task instead provides a low-bandwidth next-doorway/route cue. It does not expose the global map or complete action sequence; it identifies the next topological transition that matters.

The registered models use this cue as a bounded residual prior rather than a hard planner. The neural policy can override it when local obstacles or interaction state require a different action.

## Development evidence

During the route-prior redesign, a short Small-only PPO sanity run was able to produce a policy that succeeded on 99/100 packaged test-map × seed episodes across the five map scales, with mean episode length around 61 steps. This is **development evidence only**: it was not a multi-training-seed controlled benchmark and must not be compared numerically with formal Local Transport results.

The more important conclusion is mechanistic: after reward and route-cue alignment were corrected, the environment ceased to be structurally stuck at zero success. That makes subsequent controlled model comparisons scientifically meaningful.

## Proposed controlled experiments

Three questions remain open. First, how much does the route prior contribute relative to the learned visual policy? Second, does action memory still help after topological guidance removes many short deadlocks? Third, does GRU memory improve generalization on Large++ layouts, or merely add capacity?

A useful factorial study would vary route prior {off, bounded}, memory {none, action history, GRU+action history}, and reward {sparse, actionable geodesic}, while measuring success by size tier, door interactions, cycle events, and path efficiency. Such a study can distinguish a navigation-prior effect from a memory effect.
