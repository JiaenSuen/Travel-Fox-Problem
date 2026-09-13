from __future__ import annotations

REWARD_SPEC = {
    "key": "004_phase_consistent_transport",
    "display_name": "004 · Phase-Consistent Transport",
    "description": (
        "Interaction-aware symmetric geodesic shaping with explicit early-drop, missed-interaction, "
        "and repeated-state penalties for Local Transport reward studies."
    ),
}


class PhaseConsistentTransportReward004:
    """Reward candidate for Local Transport ablations.

    The dense term is applied only to genuine movement inside one task phase. Pickup
    switches the active potential from cargo to goal, so it is handled as a milestone
    instead of comparing unrelated distance fields. Symmetric +/− progress removes the
    historical two-step shaping exploit; small repeated-state penalties target persistent
    deadlocks without making novelty the main objective.
    """

    def __init__(self) -> None:
        self.step_cost = -0.008
        self.progress_scale = 0.075
        self.pickup_bonus = 2.0
        self.delivery_bonus = 10.0
        self.invalid_penalty = -0.12
        self.early_drop_penalty = -1.5
        self.missed_interaction_penalty = -0.16

    def compute(self, t: dict[str, object]) -> float:
        reward = self.step_cost
        pickup = bool(t.get("pickup", False))
        delivered = bool(t.get("delivered", False))
        early_drop = bool(t.get("early_drop", False))

        if bool(t.get("moved", False)) and not pickup and not delivered and not early_drop:
            reward += self.progress_scale * float(t.get("distance_delta", 0.0) or 0.0)
        if pickup:
            reward += self.pickup_bonus
        if delivered:
            reward += self.delivery_bonus
        if early_drop:
            reward += self.early_drop_penalty
        elif bool(t.get("invalid", False)):
            reward += self.invalid_penalty

        # These terms are inactive under the task mask (interaction is forced), but
        # make valid-mask ablations semantically meaningful instead of allowing the
        # policy to repeatedly walk away from an available pickup/delivery for free.
        if bool(t.get("missed_pickup", False)) or bool(t.get("missed_delivery", False)):
            reward += self.missed_interaction_penalty

        if bool(t.get("repeat_visit", False)):
            visits = int(t.get("visit_count", 3) or 3)
            reward -= min(0.10, 0.015 * max(1, visits - 2))
        return float(reward)


def create_reward() -> PhaseConsistentTransportReward004:
    return PhaseConsistentTransportReward004()
