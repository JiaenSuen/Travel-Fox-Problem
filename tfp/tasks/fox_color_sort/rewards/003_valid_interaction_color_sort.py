from __future__ import annotations

REWARD_SPEC = {
    "key": "003_valid_interaction_color_sort",
    "display_name": "003 · Valid-Interaction Color Sort",
    "description": "Dense color-sort shaping with opportunity costs when valid-mask policies ignore available pickup/delivery interactions.",
}


class ValidInteractionColorSortReward:
    def compute(self, transition: dict[str, object]) -> float:
        reward = -0.01
        if bool(transition.get("invalid", False)):
            reward -= 0.12
        pickup = bool(transition.get("pickup", False))
        delivered = bool(transition.get("item_delivered", False))
        if pickup:
            reward += 0.35
        if delivered:
            reward += 2.2
        if bool(transition.get("task_complete", False)):
            reward += 6.0
        if bool(transition.get("missed_pickup", False)):
            reward -= 0.15
        if bool(transition.get("missed_delivery", False)):
            reward -= 0.25
        if bool(transition.get("moved", False)) and not pickup and not delivered:
            reward += 0.07 * int(transition.get("distance_delta", 0))
        return float(reward)


def create_reward() -> ValidInteractionColorSortReward:
    return ValidInteractionColorSortReward()
