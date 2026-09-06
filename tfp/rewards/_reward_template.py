"""Copy this file to e.g. ``002_my_reward.py`` and edit it."""

REWARD_SPEC = {
    "key": "002_my_reward",
    "display_name": "002 · My Reward",
    "description": "Describe the hypothesis tested by this reward function.",
}


class MyReward002:
    def compute(self, transition: dict[str, object]) -> float:
        reward = -0.01
        if bool(transition["pickup"]):
            reward += 1.0
        if bool(transition["delivered"]):
            reward += 8.0
        return reward


def create_reward() -> MyReward002:
    return MyReward002()
