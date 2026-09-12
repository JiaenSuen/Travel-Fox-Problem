from __future__ import annotations

REWARD_SPEC = {
    "key": "001_intercept_safety_potential",
    "display_name": "001 · Intercept-Safety Potential",
    "description": (
        "Symmetric progress toward the earliest feasible carrier interception or delivery goal, "
        "combined with local predator-risk shaping and terminal collision cost."
    ),
}


class InterceptSafetyPotentialReward:
    STEP_COST = 0.010
    PROGRESS_SCALE = 0.20
    SAFETY_SCALE = 0.10
    NEAR_WOLF_PENALTY = 0.10
    INVALID_PENALTY = 0.18
    MISSED_INTERACTION_PENALTY = 0.30
    PICKUP_BONUS = 4.0
    DELIVERY_BONUS = 15.0
    WOLF_COLLISION_PENALTY = 15.0

    def compute(self, transition: dict[str, object]) -> float:
        reward = -self.STEP_COST
        # Symmetric potential shaping: reversing a move cannot farm reward.
        reward += self.PROGRESS_SCALE * int(transition.get("navigation_delta", 0))

        # Safety shaping is only supplied while the wolf is already near the agent.
        safety_delta = int(transition.get("safety_delta", 0))
        reward += self.SAFETY_SCALE * safety_delta
        wolf_distance = int(transition.get("wolf_distance", 999))
        if wolf_distance <= 2 and not bool(transition.get("wolf_collision", False)):
            reward -= self.NEAR_WOLF_PENALTY

        if bool(transition.get("invalid", False)):
            reward -= self.INVALID_PENALTY
        if bool(transition.get("missed_pickup", False)) or bool(transition.get("missed_delivery", False)):
            reward -= self.MISSED_INTERACTION_PENALTY
        if bool(transition.get("pickup", False)):
            reward += self.PICKUP_BONUS
        if bool(transition.get("delivered", False)):
            reward += self.DELIVERY_BONUS
        if bool(transition.get("wolf_collision", False)):
            reward -= self.WOLF_COLLISION_PENALTY
        return float(reward)


def create_reward() -> InterceptSafetyPotentialReward:
    return InterceptSafetyPotentialReward()
