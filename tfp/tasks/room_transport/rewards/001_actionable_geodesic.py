from __future__ import annotations

REWARD_SPEC = {
    "key": "001_actionable_geodesic",
    "display_name": "001 · Actionable Geodesic",
    "description": (
        "Dense shortest-action progress shaping that counts closed-door traversal as "
        "TOGGLE+MOVE, with pickup/delivery milestones and no standalone door bonus."
    ),
}


class ActionableGeodesicReward:
    """Learning-oriented reward for the long-horizon room task.

    V4 rewarded opening any door directly, which could make open/close behavior locally
    profitable even when the door was irrelevant. V5 instead measures progress in exact
    *action cost* to the current object/delivery target. Opening a closed door only earns
    progress when it lowers that cost. Closing it back cancels the potential gain.
    """

    STEP_COST = 0.010
    PROGRESS_SCALE = 0.22
    INVALID_PENALTY = 0.18
    REPEAT_PENALTY = 0.025
    PICKUP_BONUS = 3.0
    DELIVERY_BONUS = 12.0
    EARLY_DROP_PENALTY = 1.0

    def compute(self, transition: dict[str, object]) -> float:
        reward = -self.STEP_COST

        # Symmetric action-distance shaping. Positive and negative deltas use the same
        # scale, preventing two-step movement/toggle cycles from farming shaping reward.
        reward += self.PROGRESS_SCALE * int(transition.get("actionable_delta", 0))

        if bool(transition.get("invalid", False)):
            reward -= self.INVALID_PENALTY
        if bool(transition.get("early_drop", False)):
            reward -= self.EARLY_DROP_PENALTY
        if bool(transition.get("repeat_visit", False)):
            reward -= self.REPEAT_PENALTY

        if bool(transition.get("pickup", False)):
            reward += self.PICKUP_BONUS
        if bool(transition.get("delivered", False)):
            reward += self.DELIVERY_BONUS

        return float(reward)


def create_reward() -> ActionableGeodesicReward:
    return ActionableGeodesicReward()
