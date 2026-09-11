from __future__ import annotations

import unittest
import numpy as np
import torch
from pathlib import Path

from tfp.intrinsic import discover_intrinsic_plugins, load_intrinsic_plugin
from tfp.models import discover_model_plugins
from tfp.models.model_api import load_model_plugin
from tfp.policies import load_policy_plugin

ROOT = Path(__file__).resolve().parents[1]


class TFPExplorationModelTests(unittest.TestCase):
    def test_active_model_order_and_restored_branches(self):
        models = discover_model_plugins()
        expected = {
            '001_simple_cnn', '002_simple_cnn_tabu', '003_action_memory_cnn',
            '004_simple_cnn_tabux', '005_action_memory_tabux_cnn',
            '006_ppo_gru_bootstrap', '007_ppo_gru_action_memory',
            '008_ppo_gru_episodic_count', '009_ppo_gru_action_memory_tabux',
            '010_ppo_gru_gobi',
        }
        self.assertEqual(set(models), expected)

    def test_intrinsic_plugins_restored(self):
        self.assertEqual(set(discover_intrinsic_plugins()), {
            '001_episodic_count', '001_gobi_compact'
        })
        obs = np.zeros((4, 10, 5, 5), dtype=np.float32)
        nxt = obs.copy()
        nxt[:, 6, 2, 2] = np.linspace(-1, 1, 4)
        actions = np.arange(4, dtype=np.int64)
        for name in discover_intrinsic_plugins():
            module = load_intrinsic_plugin(name)[1]((10, 5, 5), 6, torch.device('cpu'))
            bonus = module.compute(obs, nxt, actions, (0, 1, 2, 3), (False, False, False, True))
            self.assertEqual(bonus.shape, (4,))
            self.assertTrue(np.isfinite(bonus).all())
            self.assertTrue(np.isfinite(module.update(obs, nxt, actions)))

    def test_bootstrap_exactly_matches_cnn_at_zero_progress(self):
        # Weight files are intentionally excluded from Git. Use an in-memory mature
        # feed-forward state to verify exact bootstrap transfer semantics.
        base = load_model_plugin('001_simple_cnn')[1]((10, 5, 5), 6)
        state = base.state_dict()
        base.eval()
        gru = load_model_plugin('006_ppo_gru_bootstrap')[1]((10, 5, 5), 6)
        gru.initialize_from_feedforward_state(state)
        gru.set_training_progress(0.0)
        gru.eval()
        x = torch.randn((3, 10, 5, 5))
        bl, bv = base(x)
        gl, gv, _ = gru.act_forward(x, (0, 1, 2), update=False)
        self.assertTrue(torch.allclose(bl, gl, atol=1e-6))
        self.assertTrue(torch.allclose(bv, gv, atol=1e-6))

    def test_gru_action_memory_shape_preserved_after_reindex(self):
        m = load_model_plugin('007_ppo_gru_action_memory')[1]((10, 5, 5), 6)
        x = torch.zeros((1, 10, 5, 5))
        m.observe_action_memory((0,), (3,), (0.0,), (False,))
        logits, value, ctx = m.act_forward(x, (0,), update=True)
        self.assertEqual(tuple(ctx.shape), (1, 70))
        self.assertEqual(int(round(float(ctx[0, -1]))), 3)
        l2, v2 = m.training_forward(x, ctx)
        self.assertEqual(tuple(l2.shape), (1, 6))
        self.assertEqual(tuple(v2.shape), (1,))

    def test_legacy_recurrent_aliases_follow_new_indices(self):
        self.assertEqual(load_model_plugin('008_ppo_gru_action_memory')[0].key, '007_ppo_gru_action_memory')
        self.assertEqual(load_model_plugin('010_ppo_gru_episodic_count')[0].key, '008_ppo_gru_episodic_count')
        self.assertEqual(load_model_plugin('014_ppo_gru_gobi')[0].key, '010_ppo_gru_gobi')
        self.assertEqual(load_model_plugin('012_ppo_gru_gobi')[0].key, '010_ppo_gru_gobi')

    def test_gru_action_memory_tabux_has_both_memories(self):
        m = load_model_plugin('009_ppo_gru_action_memory_tabux')[1]((10, 5, 5), 6)
        x = torch.zeros((1, 10, 5, 5))
        m.observe_action_memory((0,), (3,), (0.0,), (False,))
        logits, value, ctx = m.act_forward(x, (0,), update=True)
        self.assertEqual(tuple(logits.shape), (1, 6))
        self.assertEqual(tuple(value.shape), (1,))
        self.assertEqual(tuple(ctx.shape), (1, 70))
        for action in [3, 3, 2, 2, 3, 3, 2, 2]:
            m.observe_behavior((0,), (action,), (0.0,), (False,))
        mask = torch.tensor([[1, 1, 1, 1, 0, 0]], dtype=torch.bool)
        logits = torch.tensor([[4.0, 3.0, 1.0, 8.0, -9.0, -9.0]])
        rewritten = int(m.rewrite_actions(torch.tensor([3]), logits, mask, (0,)).item())
        self.assertIn(rewritten, (0, 1))

    def test_cycle_guard_forces_pickup_and_returns_effective_mask(self):
        m = load_model_plugin('001_simple_cnn')[1]((10, 5, 5), 6)
        p = load_policy_plugin('002_ppo_cycle_guard')[1]()
        logits = torch.tensor([[5.0, 4.0, 3.0, 2.0, -10.0, 1.0]])
        mask = torch.ones((1, 6), dtype=torch.bool)
        actions, _, _, effective = p.sample(m, logits, mask, (0,))
        self.assertEqual(int(actions.item()), 4)
        self.assertEqual(effective[0].tolist(), [False, False, False, False, True, False])

    def test_cycle_guard_breaks_abab(self):
        m = load_model_plugin('001_simple_cnn')[1]((10, 5, 5), 6)
        p = load_policy_plugin('002_ppo_cycle_guard')[1]()
        p._last_mode = 'train'
        for a in [2, 3, 2, 3]:
            p.observe(m, (0,), (a,), (0.0,), (False,))
        logits = torch.tensor([[0.0, 0.0, 9.0, 8.0, -9.0, -9.0]])
        mask = torch.tensor([[1, 1, 1, 1, 0, 0]], dtype=torch.bool)
        action, _, _, effective = p.sample(m, logits, mask, (0,))
        self.assertFalse(bool(effective[0, 2]))
        self.assertNotEqual(int(action.item()), 2)


if __name__ == '__main__':
    unittest.main()
