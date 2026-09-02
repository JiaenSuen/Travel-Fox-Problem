from __future__ import annotations

import unittest

import torch

from tfp.models import discover_model_plugins
from tfp.models.action_memory_tabux_cnn_005 import ActionMemoryTabuXCNN005
from tfp.models.simple_cnn_tabux_004 import SimpleCNNTabuX004
from tfp.policies.ppo_categorical_001 import PPOCategoricalPolicy001
from tfp.training.ppo import _validation_rank


class TFPTabuXModelTests(unittest.TestCase):
    def test_tabux_plugins_are_discoverable(self):
        plugins = discover_model_plugins()
        self.assertIn("simple_cnn_tabux_004", plugins)
        self.assertIn("action_memory_tabux_cnn_005", plugins)

    def test_tabux_escapes_three_cell_basin(self):
        model = SimpleCNNTabuX004((10, 5, 5), 6)
        env_ids = (0,)
        # Relative positions repeatedly visit x={0,1,2}.
        for action in [3, 3, 2, 2, 3, 3, 2, 2]:
            model.observe_behavior(env_ids, (action,), (0.0,), (False,))
        mask = torch.tensor([[1, 1, 1, 1, 0, 0]], dtype=torch.bool)
        logits = torch.tensor([[4.0, 3.0, 1.0, 8.0, -9.0, -9.0]])
        action = int(model.rewrite_actions(torch.tensor([3]), logits, mask, env_ids).item())
        self.assertIn(action, (0, 1))

    def test_tabux_prevents_immediate_reentry_after_escape(self):
        model = SimpleCNNTabuX004((10, 5, 5), 6)
        env_ids = (0,)
        for action in [3, 3, 2, 2, 3, 3, 2, 2]:
            model.observe_behavior(env_ids, (action,), (0.0,), (False,))
        # Escape upward from the repeated x={0,1,2} basin.
        model.observe_behavior(env_ids, (0,), (0.0,), (False,))
        mask = torch.tensor([[1, 1, 1, 1, 0, 0]], dtype=torch.bool)
        # DOWN has the highest neural logit but returns directly to the recent basin.
        logits = torch.tensor([[1.0, 9.0, 3.0, 4.0, -9.0, -9.0]])
        action = int(model.rewrite_actions(torch.tensor([1]), logits, mask, env_ids).item())
        self.assertNotEqual(action, 1)

    def test_tabux_breaks_pickup_drop_local_state_cycle(self):
        model = SimpleCNNTabuX004((10, 5, 5), 6)
        env_ids = (0,)
        for action, reward in [(4, 1.5), (5, -0.1), (4, 1.5), (5, -0.1), (4, 1.5), (5, -0.1)]:
            model.observe_behavior(env_ids, (action,), (reward,), (False,))
        mask = torch.ones((1, 6), dtype=torch.bool)
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 9.0, 8.0]])
        action = int(model.rewrite_actions(torch.tensor([4]), logits, mask, env_ids).item())
        self.assertNotIn(action, (4, 5))

    def test_tabux_remains_inference_only(self):
        model = SimpleCNNTabuX004((10, 5, 5), 6)
        policy = PPOCategoricalPolicy001()
        env_ids = (0,)
        for action in [3, 3, 2, 2, 3, 3, 2, 2]:
            model.observe_behavior(env_ids, (action,), (0.0,), (False,))
        logits = torch.tensor([[-100.0, -100.0, -100.0, 100.0, -100.0, -100.0]])
        mask = torch.ones((1, 6), dtype=torch.bool)
        sampled, _, _ = policy.sample(model, logits, mask, env_ids)
        self.assertEqual(int(sampled.item()), 3)

    def test_action_memory_tabux_keeps_learnable_context(self):
        model = ActionMemoryTabuXCNN005((10, 5, 5), 6)
        x = torch.zeros((1, 10, 5, 5))
        model.observe_action_memory((0,), (3,), (0.0,), (False,))
        logits, value, context = model.act_forward(x, (0,), update=True)
        self.assertEqual(tuple(logits.shape), (1, 6))
        self.assertEqual(tuple(value.shape), (1,))
        self.assertEqual(int(context[0, -1]), 3)
        logits2, value2 = model.training_forward(x, context)
        self.assertEqual(tuple(logits2.shape), (1, 6))
        self.assertEqual(tuple(value2.shape), (1,))

    def test_best_checkpoint_rank_prefers_success_then_efficiency_then_steps(self):
        self.assertGreater(_validation_rank(0.8, 0.6, 50), _validation_rank(0.7, 1.0, 10))
        self.assertGreater(_validation_rank(0.8, 0.7, 80), _validation_rank(0.8, 0.6, 20))
        self.assertGreater(_validation_rank(0.8, 0.7, 40), _validation_rank(0.8, 0.7, 60))


if __name__ == "__main__":
    unittest.main()
