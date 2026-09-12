from __future__ import annotations

REWARD_SPEC = {
    "key": "001_counterfactual_intercept_risk",
    "display_name": "001 · Counterfactual Intercept-Risk Potential",
    "description": (
        "Courier-caused interception/delivery progress plus symmetric predictive safety shaping and local safety-regret, "
        "with timing-aware WAIT and dominant collision cost."
    ),
}


class CounterfactualInterceptRiskReward:
    STEP_COST = 0.012
    NAV_SCALE = 0.28
    SAFETY_DELTA_SCALE = 0.14
    SAFETY_REGRET_SCALE = 0.12
    INVALID = 0.22
    MISSED = 0.45
    NEAR_WOLF = 0.10
    GOOD_WAIT_REFUND = 0.010
    BAD_WAIT = 0.010
    PICKUP = 5.5
    DELIVERY = 20.0
    WOLF_COLLISION = 25.0

    def compute(self, t: dict[str, object]) -> float:
        reward = -self.STEP_COST
        reward += self.NAV_SCALE * float(t.get("navigation_delta", 0.0) or 0.0)
        reward += self.SAFETY_DELTA_SCALE * float(t.get("safety_delta", 0.0) or 0.0)
        reward -= self.SAFETY_REGRET_SCALE * float(t.get("safety_regret", 0.0) or 0.0)

        wolf_distance = int(t.get("wolf_distance", 999) or 999)
        wolf_collision = bool(t.get("wolf_collision", False))
        if wolf_distance <= 2 and not wolf_collision:
            reward -= self.NEAR_WOLF
        if bool(t.get("well_timed_wait", False)):
            reward += self.GOOD_WAIT_REFUND  # still net-negative
        elif bool(t.get("waited", False)):
            reward -= self.BAD_WAIT
        if bool(t.get("invalid", False)):
            reward -= self.INVALID
        if bool(t.get("missed_pickup", False)) or bool(t.get("missed_delivery", False)):
            reward -= self.MISSED
        if bool(t.get("pickup", False)):
            reward += self.PICKUP
        if bool(t.get("delivered", False)):
            reward += self.DELIVERY
        if wolf_collision:
            reward -= self.WOLF_COLLISION
        return float(reward)


def create_reward() -> CounterfactualInterceptRiskReward:
    return CounterfactualInterceptRiskReward()
