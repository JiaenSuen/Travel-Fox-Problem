from pathlib import Path
import numpy as np

from tfp.envs.transport_env import TransportEnv
from tfp.rewards import load_reward_plugin


def _env() -> TransportEnv:
    maps = sorted((Path('tfp/tasks/local_transport/maps/train')).glob('*.txt'))
    assert maps
    return TransportEnv(maps[:1], reward_module='003_valid_interaction_transport')


def test_valid_mask_never_exposes_early_drop():
    env = _env()
    env.reset(seed=3)
    env.carrying = True
    env.object_pos = None
    if env.agent_pos == env.goal_pos:
        # select any reachable free non-goal cell
        for a, (dr, dc) in env.ACTIONS.items():
            q = (env.agent_pos[0] + dr, env.agent_pos[1] + dc)
            if env._is_free(q) and q != env.goal_pos:
                env.agent_pos = q
                break
    mask = env.valid_action_mask('valid')
    assert not bool(mask[5]), 'valid mask must not expose early DROP outside the goal'


def test_valid_mask_keeps_interaction_optional_but_semantically_valid():
    env = _env()
    env.reset(seed=4)
    env.agent_pos = env.object_pos
    mask = env.valid_action_mask('valid')
    assert bool(mask[4])
    assert int(mask.sum()) >= 2, 'valid should allow PICKUP or leaving the cargo cell'
    env.carrying = True
    env.object_pos = None
    env.agent_pos = env.goal_pos
    mask = env.valid_action_mask('valid')
    assert bool(mask[5])
    assert int(mask.sum()) >= 2, 'valid should allow DELIVER or leaving the goal cell'


def test_valid_interaction_reward_penalizes_ignored_interaction():
    _, factory = load_reward_plugin('003_valid_interaction_transport')
    reward = factory()
    base = {
        'invalid': False, 'pickup': False, 'delivered': False,
        'moved': True, 'distance_delta': 0,
        'missed_pickup': False, 'missed_delivery': False,
    }
    ordinary = reward.compute(dict(base))
    missed = reward.compute({**base, 'missed_pickup': True})
    assert missed < ordinary
