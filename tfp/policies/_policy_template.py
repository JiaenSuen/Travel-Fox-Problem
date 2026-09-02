"""Copy this file to e.g. ``my_policy_001.py`` and edit it."""

from tfp.policies.ppo_categorical_001 import PPOCategoricalPolicy001

POLICY_SPEC = {
    "key": "my_policy_001",
    "display_name": "My Policy 001",
    "algorithm": "ppo",
    "description": "Describe the action-selection hypothesis tested here.",
}


class MyPolicy001(PPOCategoricalPolicy001):
    pass


def create_policy() -> MyPolicy001:
    return MyPolicy001()
