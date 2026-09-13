from __future__ import annotations

from tfp.rewards.reward_api import RewardSpec


REWARD_SPEC = RewardSpec(
    key="002_predictive_hazard_potential",
    display_name="Predictive Hazard-Aware Dependency Potential",
    description=(
        "Dependency progress plus one-step motion-aware wolf risk, local safety-regret, near-miss, "
        "milestone, and anti-cycle terms for long-horizon multi-cargo logistics."
    ),
)


class PredictiveHazardPotential:
    """Risk-aware reward targeting the dominant wolf-collision failure mode.

    Mission progress stays symmetric. Safety terms are deliberately action-relative:
    current/predicted risk is compared before and immediately after the courier action,
    before stochastic wolf motion. A local regret term penalizes choosing a materially
    riskier feasible move when a safer alternative was available. Actual near misses and
    collisions remain sparse safety events.
    """

    def compute(self, t: dict[str, object]) -> float:
        reward = -0.010
        if bool(t.get("wait_action")):
            reward -= 0.004

        reward += 0.18 * float(t.get("progress_delta", 0.0) or 0.0)
        reward += 0.35 * float(t.get("risk_improvement", 0.0) or 0.0)
        reward += 0.90 * float(t.get("predictive_risk_improvement", 0.0) or 0.0)
        reward -= 1.10 * float(t.get("safety_regret", 0.0) or 0.0)

        if bool(t.get("near_miss")):
            reward -= 0.65
        if bool(t.get("entered_new_room")):
            reward += 0.025
        if bool(t.get("key_pickup")):
            reward += 2.4
        if bool(t.get("cargo_pickup")):
            reward += 3.2
        if bool(t.get("door_opened")):
            reward += 0.03
        if bool(t.get("door_closed")):
            reward -= 0.10
        if bool(t.get("delivered")):
            reward += 8.5
        if bool(t.get("task_complete")):
            reward += 18.0

        if bool(t.get("invalid")):
            reward -= 0.24
        if bool(t.get("locked_without_key")):
            reward -= 0.24
        if bool(t.get("early_drop")):
            reward -= 1.4
        if bool(t.get("repeat_visit")):
            visits = int(t.get("visit_count", 3) or 3)
            reward -= min(0.20, 0.03 * max(1, visits - 2))
        if bool(t.get("predator_collision")):
            reward -= 30.0
        return float(reward)


def create_reward() -> PredictiveHazardPotential:
    return PredictiveHazardPotential()
