from __future__ import annotations

import unittest

import torch

from tfp.models import discover_model_plugins
from tfp.models.action_memory_cnn_003 import ActionMemoryCNN003
from tfp.models.simple_cnn_tabu_002 import SimpleCNNTabu002
from tfp.policies.ppo_categorical_001 import PPOCategoricalPolicy001
from tfp.tasks import get_task
from tfp.utils import discover_maps


class TFPCoreModelTests(unittest.TestCase):
    def test_benchmark_counts(self):
        task = get_task("TFP-FoxTransport-Local")
        self.assertEqual(len(discover_maps(task.train_map_dir)), 54)
        self.assertEqual(len(discover_maps(task.test_map_dir)), 18)

    def test_plugins_include_builtin_models(self):
        plugins = discover_model_plugins()
        self.assertIn("simple_cnn_001", plugins)
        self.assertIn("simple_cnn_tabu_002", plugins)
        self.assertIn("action_memory_cnn_003", plugins)

    def test_tabu_breaks_forced_left_right_cycle(self):
        model = SimpleCNNTabu002((10, 5, 5), 6)
        mask = torch.tensor([[1, 1, 1, 1, 0, 0]], dtype=torch.bool)
        env_ids = (0,)
        executed = []
        for step in range(10):
            proposed = torch.tensor([2 if step % 2 == 0 else 3])
            logits = torch.tensor([[0.0, 0.1, 5.0 if step % 2 == 0 else 1.0, 5.0 if step % 2 else 1.0, -5.0, -5.0]])
            action = int(model.rewrite_actions(proposed, logits, mask, env_ids).item())
            executed.append(action)
            model.observe_behavior(env_ids, (action,), (0.0,), (False,))
        self.assertTrue(any(action in (0, 1) for action in executed[4:]))

    def test_tabu_is_inference_only_in_default_ppo_policy(self):
        model = SimpleCNNTabu002((10, 5, 5), 6)
        policy = PPOCategoricalPolicy001()
        env_ids = (0,)
        for action in [4, 5, 4, 5]:
            model.observe_behavior(env_ids, (action,), (0.0,), (False,))
        logits = torch.tensor([[-100.0, -100.0, -100.0, -100.0, 100.0, -100.0]])
        mask = torch.ones((1, 6), dtype=torch.bool)

        # Greedy inference uses the Tabu rewrite and must avoid PICKUP/DROP.
        greedy = int(policy.greedy(model, logits, mask, env_ids).item())
        self.assertNotIn(greedy, (4, 5))

        # PPO sampling does not call rewrite_actions. With an almost-deterministic
        # distribution this remains PICKUP, proving training data is unmodified.
        sampled, _, _ = policy.sample(model, logits, mask, env_ids)
        self.assertEqual(int(sampled.item()), 4)

    def test_action_memory_context_is_internal_and_learnable(self):
        model = ActionMemoryCNN003((10, 5, 5), 6)
        x = torch.zeros((2, 10, 5, 5))
        logits, value, context = model.act_forward(x, (0, 1), update=True)
        self.assertEqual(tuple(logits.shape), (2, 6))
        self.assertEqual(tuple(value.shape), (2,))
        self.assertEqual(tuple(context.shape), (2, 8))
        self.assertEqual(context.dtype, torch.long)

        model.observe_action_memory((0, 1), (2, 4), (0.0, 0.0), (False, False))
        _, _, context2 = model.act_forward(x, (0, 1), update=True)
        self.assertEqual(int(context2[0, -1]), 2)
        self.assertEqual(int(context2[1, -1]), 4)

        logits2, value2 = model.training_forward(x, context2)
        self.assertEqual(tuple(logits2.shape), (2, 6))
        self.assertEqual(tuple(value2.shape), (2,))

    def test_action_memory_resets_on_episode_end(self):
        model = ActionMemoryCNN003((10, 5, 5), 6)
        x = torch.zeros((1, 10, 5, 5))
        model.observe_action_memory((0,), (3,), (0.0,), (False,))
        _, _, before = model.act_forward(x, (0,), update=True)
        self.assertEqual(int(before[0, -1]), 3)
        model.observe_action_memory((0,), (4,), (0.0,), (True,))
        _, _, after = model.act_forward(x, (0,), update=True)
        self.assertTrue(torch.all(after == 6))


if __name__ == "__main__":
    unittest.main()
