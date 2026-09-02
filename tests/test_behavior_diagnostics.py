from __future__ import annotations

import unittest

import torch

from tfp.evaluation.diagnostics import BehaviorDiagnostics, periodic_suffix
from tfp.models.simple_cnn_tabu_002 import SimpleCNNTabu002


class TFPBehaviorDiagnosticsTests(unittest.TestCase):
    def test_periodic_suffix_detects_generic_short_cycles(self):
        self.assertEqual(periodic_suffix([2, 3, 2, 3]), (2, 3))
        self.assertEqual(periodic_suffix([4, 5, 4, 5]), (4, 5))
        self.assertEqual(periodic_suffix([0, 2, 3, 0, 2, 3]), (0, 2, 3))
        self.assertEqual(periodic_suffix([1, 1, 1, 1]), (1,))
        self.assertIsNone(periodic_suffix([0, 0, 1, 2]))

    def test_tabu_breaks_pickup_drop_cycle_even_with_positive_pickup_reward(self):
        model = SimpleCNNTabu002((10, 5, 5), 6)
        env_ids = (0,)
        for action, reward in [(4, 1.49), (5, -0.09), (4, 1.49), (5, -0.09)]:
            model.observe_behavior(env_ids, (action,), (reward,), (False,))
        mask = torch.ones((1, 6), dtype=torch.bool)
        logits = torch.tensor([[0.2, 0.3, 0.4, 0.5, 5.0, 4.0]])
        action = int(model.rewrite_actions(torch.tensor([4]), logits, mask, env_ids).item())
        self.assertNotIn(action, (4, 5))

    def test_tabu_commits_in_two_action_corridor_instead_of_deadlocking(self):
        model = SimpleCNNTabu002((10, 5, 5), 6)
        env_ids = (0,)
        for action in [2, 3, 2, 3]:
            model.observe_behavior(env_ids, (action,), (0.0,), (False,))
        mask = torch.tensor([[0, 0, 1, 1, 0, 0]], dtype=torch.bool)
        logits = torch.tensor([[0.0, 0.0, 5.0, 4.0, -5.0, -5.0]])
        action = int(model.rewrite_actions(torch.tensor([2]), logits, mask, env_ids).item())
        self.assertEqual(action, 3)

    def test_behavior_diagnostics_count_interaction_cycles_and_revisits(self):
        diag = BehaviorDiagnostics()
        state_a = ((1, 1), False, (1, 1))
        state_b = ((1, 1), True, None)
        for action, state in [(4, state_b), (5, state_a), (4, state_b), (5, state_a)]:
            diag.observe(action, state)
        self.assertGreaterEqual(diag.cycle_events, 1)
        self.assertGreaterEqual(diag.interaction_cycle_events, 1)
        self.assertGreater(diag.state_revisit_rate, 0.0)


if __name__ == "__main__":
    unittest.main()
