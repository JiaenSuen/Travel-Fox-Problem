from __future__ import annotations

REWARD_SPEC = {
    "key": "001_dense_color_sort",
    "display_name": "001 · Dense Color Sort",
    "description": "Symmetric geodesic shaping plus pickup, per-item delivery, completion, and invalid-action terms.",
}


class DenseColorSortReward:
    def __init__(self) -> None:
        self.step_cost = -0.01
        self.progress_scale = 0.07
        self.pickup_bonus = 0.35
        self.delivery_bonus = 2.0
        self.complete_bonus = 6.0
        self.invalid_penalty = -0.08

    def compute(self, transition: dict[str, object]) -> float:
        reward = self.step_cost
        if bool(transition.get("invalid", False)):
            reward += self.invalid_penalty
        pickup = bool(transition.get("pickup", False))
        delivered = bool(transition.get("item_delivered", False))
        if pickup:
            reward += self.pickup_bonus
        if delivered:
            reward += self.delivery_bonus
        if bool(transition.get("task_complete", False)):
            reward += self.complete_bonus
        if bool(transition.get("moved", False)) and not pickup and not delivered:
            reward += self.progress_scale * int(transition.get("distance_delta", 0))
        return float(reward)


def create_reward() -> DenseColorSortReward:
    return DenseColorSortReward()
