from __future__ import annotations

REWARD_SPEC = {
    "key": "dense_transport_001",
    "display_name": "Dense Transport 001",
    "description": "Task baseline with step cost, geodesic progress shaping, pickup bonus, delivery reward, and invalid-action penalty.",
}


class DenseTransportReward001:
    def __init__(self) -> None:
        self.step_cost = -0.01
        self.progress_bonus = 0.06
        self.pickup_bonus = 1.5
        self.delivery_bonus = 8.0
        self.invalid_penalty = -0.08

    def compute(self, transition: dict[str, object]) -> float:
        reward = self.step_cost
        if bool(transition.get("invalid", False)):
            reward += self.invalid_penalty
        if bool(transition.get("pickup", False)):
            reward += self.pickup_bonus
        if bool(transition.get("delivered", False)):
            reward += self.delivery_bonus
        if not bool(transition.get("delivered", False)):
            delta = int(transition.get("distance_delta", 0))
            if delta > 0:
                reward += self.progress_bonus
            elif delta < 0:
                reward -= self.progress_bonus * 0.5
        return float(reward)


def create_reward() -> DenseTransportReward001:
    return DenseTransportReward001()
