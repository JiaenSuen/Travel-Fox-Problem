from __future__ import annotations

REWARD_SPEC = {
    "key": "002_milestone_color_sort",
    "display_name": "002 · Milestone Color Sort",
    "description": "Lower shaping, stronger item-completion milestones; useful for testing dependence on dense navigation reward.",
}


class MilestoneColorSortReward:
    def compute(self, transition: dict[str, object]) -> float:
        reward = -0.012
        if bool(transition.get("invalid", False)):
            reward -= 0.10
        if bool(transition.get("pickup", False)):
            reward += 0.20
        if bool(transition.get("item_delivered", False)):
            reward += 3.0
        if bool(transition.get("task_complete", False)):
            reward += 8.0
        if bool(transition.get("moved", False)) and not bool(transition.get("item_delivered", False)):
            reward += 0.035 * int(transition.get("distance_delta", 0))
        return float(reward)


def create_reward() -> MilestoneColorSortReward:
    return MilestoneColorSortReward()
