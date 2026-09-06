from __future__ import annotations

REWARD_SPEC = {
    "key": "002_cycle_safe_transport",
    "display_name": "002 · Cycle-Safe Transport",
    "description": (
        "Dense transport reward with symmetric geodesic progress shaping so a progress/regress pair cannot earn net shaping reward. "
        "Pickup/delivery bonuses remain event-based and interaction transitions are excluded from distance shaping."
    ),
}


class CycleSafeTransportReward002:
    """Dense reward designed to remove a specific short-cycle incentive.

    The original dense reward paid +0.06 for progress but only -0.03 for regression.
    With the -0.01 step cost, a two-step d->d-1->d oscillation could receive
    (+0.05) + (-0.04) = +0.01 total reward. This variant makes distance shaping
    symmetric: +0.06 * distance_delta for successful movement transitions. Therefore
    a progress/regress pair contributes zero shaping and still pays two step costs.

    Interaction transitions are not distance-shaped because PICKUP switches the target
    from cargo to goal; mixing those two distance fields can create an unrelated jump
    in reward. The plugin remains intentionally simple and stateless so it is easy to
    ablate against 001_dense_transport.
    """

    def __init__(self) -> None:
        self.step_cost = -0.01
        self.progress_scale = 0.06
        self.pickup_bonus = 1.5
        self.delivery_bonus = 8.0
        self.invalid_penalty = -0.08

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

        # Shape only genuine movement. PICKUP/DROP can switch the active target and
        # should not compare distances from two different task phases.
        if bool(transition.get("moved", False)) and not pickup and not delivered:
            reward += self.progress_scale * int(transition.get("distance_delta", 0))
        return float(reward)


def create_reward() -> CycleSafeTransportReward002:
    return CycleSafeTransportReward002()
