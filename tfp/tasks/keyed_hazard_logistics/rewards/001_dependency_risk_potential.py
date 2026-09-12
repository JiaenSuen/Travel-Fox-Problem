from __future__ import annotations

from tfp.rewards.reward_api import RewardSpec


REWARD_SPEC = RewardSpec(
    key="001_dependency_risk_potential",
    display_name="Dependency-Aware Risk Potential",
    description=(
        "Symmetric executable-progress shaping over the current symbolic subgoal, sparse key/cargo/delivery milestones, "
        "agent-caused wolf-risk shaping, and anti-cycle penalties."
    ),
)


class DependencyRiskPotential:
    """Dense but non-exploitable shaping for long-horizon access logistics.

    Progress compares the same active symbolic target before/after the courier action.
    Exogenous wolf motion is intentionally excluded from the risk potential, so the
    agent cannot receive free reward because a predator happens to move away.
    """

    def compute(self, t: dict[str, object]) -> float:
        reward = -0.010
        if bool(t.get("wait_action")):
            reward -= 0.006

        delta = float(t.get("progress_delta", 0.0) or 0.0)
        reward += 0.20 * delta

        # Safety shaping is symmetric and only measures the action-caused change
        # against the same predator positions.
        risk_improvement = float(t.get("risk_improvement", 0.0) or 0.0)
        reward += 0.55 * risk_improvement

        if bool(t.get("entered_new_room")):
            reward += 0.035
        if bool(t.get("key_pickup")):
            reward += 2.25
        if bool(t.get("cargo_pickup")):
            reward += 3.00
        if bool(t.get("door_opened")):
            # Small interaction acknowledgement only; useful doors are rewarded mainly
            # through executable potential reduction, so open/close cycling is negative.
            reward += 0.04
        if bool(t.get("door_closed")):
            reward -= 0.08
        if bool(t.get("delivered")):
            reward += 8.00
        if bool(t.get("task_complete")):
            reward += 16.00

        if bool(t.get("invalid")):
            reward -= 0.22
        if bool(t.get("locked_without_key")):
            reward -= 0.22
        if bool(t.get("early_drop")):
            reward -= 1.25
        if bool(t.get("repeat_visit")):
            visits = int(t.get("visit_count", 3) or 3)
            reward -= min(0.22, 0.035 * max(1, visits - 2))
        if bool(t.get("predator_collision")):
            reward -= 22.0
        return float(reward)


def create_reward() -> DependencyRiskPotential:
    return DependencyRiskPotential()
