from __future__ import annotations

REWARD_SPEC = {
    "key": "003_valid_interaction_transport",
    "display_name": "003 · Valid-Interaction Transport",
    "description": (
        "Cycle-safe dense shaping for valid-mask training. It preserves optional PICKUP/DELIVER decisions "
        "while penalizing leaving a cell where the required interaction is currently available."
    ),
}


class ValidInteractionTransportReward003:
    """Reward shaping for learning interactions without forcing them via the mask.

    Under the ``valid`` mask the policy may move away from cargo or from the delivery
    goal instead of taking PICKUP/DELIVER.  The observation already exposes explicit
    interaction-opportunity channels, so this reward adds a modest opportunity-cost
    penalty when the agent ignores those channels.  It does not force an action and
    therefore keeps valid-mask evaluation a genuine policy decision.

    Distance shaping is symmetric, so d->d-1->d cannot earn positive shaping reward.
    """

    def __init__(self) -> None:
        self.step_cost = -0.01
        self.progress_scale = 0.06
        self.pickup_bonus = 1.5
        self.delivery_bonus = 8.0
        self.invalid_penalty = -0.12
        self.missed_pickup_penalty = -0.20
        self.missed_delivery_penalty = -0.30

    def compute(self, transition: dict[str, object]) -> float:
        reward = self.step_cost
        if bool(transition.get("invalid", False)):
            reward += self.invalid_penalty

        pickup = bool(transition.get("pickup", False))
        delivered = bool(transition.get("delivered", False))
        if pickup:
            reward += self.pickup_bonus
        if delivered:
            reward += self.delivery_bonus

        if bool(transition.get("missed_pickup", False)):
            reward += self.missed_pickup_penalty
        if bool(transition.get("missed_delivery", False)):
            reward += self.missed_delivery_penalty

        if bool(transition.get("moved", False)) and not pickup and not delivered:
            reward += self.progress_scale * int(transition.get("distance_delta", 0))
        return float(reward)


def create_reward() -> ValidInteractionTransportReward003:
    return ValidInteractionTransportReward003()
