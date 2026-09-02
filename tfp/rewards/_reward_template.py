"""Copy this file to e.g. ``my_reward_001.py`` and edit it."""

REWARD_SPEC = {
    "key": "my_reward_001",
    "display_name": "My Reward 001",
    "description": "Describe the hypothesis tested by this reward function.",
}


class MyReward001:
    def compute(self, transition: dict[str, object]) -> float:
        reward = -0.01
        if bool(transition["pickup"]):
            reward += 1.0
        if bool(transition["delivered"]):
            reward += 8.0
        return reward


def create_reward() -> MyReward001:
    return MyReward001()
