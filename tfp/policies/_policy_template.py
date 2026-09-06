"""Copy this file to e.g. ``003_my_policy.py`` and edit it."""

from importlib import import_module

PPOCategoricalPolicy001 = import_module("tfp.policies.001_ppo_categorical").PPOCategoricalPolicy001

POLICY_SPEC = {
    "key": "003_my_policy",
    "display_name": "003 · My Policy",
    "algorithm": "ppo",
    "description": "Describe the action-selection hypothesis tested here.",
}


class MyPolicy003(PPOCategoricalPolicy001):
    pass


def create_policy() -> MyPolicy003:
    return MyPolicy003()
