from __future__ import annotations

import unittest
from pathlib import Path

from tfp.envs.transport_env import TransportEnv
from tfp.rewards import load_reward_plugin
from tfp.tasks import get_task
from tfp.utils import discover_maps


class TFPCycleSafeRewardTests(unittest.TestCase):
    def test_progress_regress_cycle_is_negative(self):
        reward = load_reward_plugin('002_cycle_safe_transport')[1]()
        progress = reward.compute({
            'moved': True, 'invalid': False, 'pickup': False, 'delivered': False,
            'distance_delta': 1,
        })
        regress = reward.compute({
            'moved': True, 'invalid': False, 'pickup': False, 'delivered': False,
            'distance_delta': -1,
        })
        self.assertAlmostEqual(progress, 0.05, places=6)
        self.assertAlmostEqual(regress, -0.07, places=6)
        self.assertLess(progress + regress, 0.0)

    def test_pickup_ignores_target_switch_distance_jump(self):
        reward = load_reward_plugin('002_cycle_safe_transport')[1]()
        pickup = reward.compute({
            'moved': False, 'invalid': False, 'pickup': True, 'delivered': False,
            'distance_delta': -17,
        })
        self.assertAlmostEqual(pickup, 1.49, places=6)

    def test_phase_consistent_reward_penalizes_early_drop_and_repeat_loops(self):
        reward = load_reward_plugin('004_phase_consistent_transport', task_id='TFP-LocalTransport')[1]()
        fwd = reward.compute({'moved': True, 'distance_delta': 1})
        back = reward.compute({'moved': True, 'distance_delta': -1})
        self.assertLess(fwd + back, 0.0)
        self.assertLess(reward.compute({'early_drop': True, 'invalid': True}), -1.0)
        self.assertLess(reward.compute({'repeat_visit': True, 'visit_count': 6}), 0.0)

    def test_none_action_mask_is_removed(self):
        task = get_task('TFP-LocalTransport')
        one_map = discover_maps(task.train_map_dir)[:1]
        env = TransportEnv(one_map, observation_mode='local', view_size=5)
        env.reset(seed=1)
        with self.assertRaisesRegex(ValueError, 'unmasked action selection is not supported'):
            env.valid_action_mask('none')

    def test_task_and_valid_masks_have_distinct_interaction_semantics(self):
        task = get_task('TFP-LocalTransport')
        one_map = discover_maps(task.train_map_dir)[:1]
        env = TransportEnv(one_map, observation_mode='local', view_size=5)
        env.reset(seed=1)
        env.agent_pos = env.object_pos
        task_mask = env.valid_action_mask('task')
        valid_mask = env.valid_action_mask('valid')
        self.assertEqual(task_mask.tolist(), [False, False, False, False, True, False])
        self.assertTrue(bool(valid_mask[4]))
        self.assertTrue(any(bool(v) for v in valid_mask[:4]))

    def test_global_observation_is_removed(self):
        task = get_task('TFP-LocalTransport')
        one_map = discover_maps(task.train_map_dir)[:1]
        with self.assertRaisesRegex(ValueError, 'supports local observation only'):
            TransportEnv(one_map, observation_mode='global')
        env = TransportEnv(one_map, observation_mode='local', view_size=5)
        obs, _ = env.reset(seed=1)
        self.assertEqual(obs.shape, (10, 5, 5))


if __name__ == '__main__':
    unittest.main()
