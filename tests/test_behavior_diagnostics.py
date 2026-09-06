from __future__ import annotations
import unittest
import torch
from tfp.evaluation.diagnostics import BehaviorDiagnostics, periodic_suffix
from tfp.models.model_api import load_model_plugin

class TFPBehaviorDiagnosticsTests(unittest.TestCase):
    def setUp(self): self.model=load_model_plugin('002_simple_cnn_tabu')[1]((10,5,5),6)
    def test_periodic_suffix(self):
        self.assertEqual(periodic_suffix([2,3,2,3]),(2,3)); self.assertEqual(periodic_suffix([4,5,4,5]),(4,5)); self.assertIsNone(periodic_suffix([0,0,1,2]))
    def test_tabu_breaks_pickup_drop(self):
        for a,r in [(4,1.49),(5,-.09),(4,1.49),(5,-.09)]: self.model.observe_behavior((0,),(a,),(r,),(False,))
        mask=torch.ones((1,6),dtype=torch.bool); logits=torch.tensor([[.2,.3,.4,.5,5.,4.]])
        self.assertNotIn(int(self.model.rewrite_actions(torch.tensor([4]),logits,mask,(0,)).item()),(4,5))
    def test_diagnostics(self):
        d=BehaviorDiagnostics(); a=((1,1),False,(1,1)); b=((1,1),True,None)
        for action,state in [(4,b),(5,a),(4,b),(5,a)]: d.observe(action,state)
        self.assertGreaterEqual(d.cycle_events,1); self.assertGreaterEqual(d.interaction_cycle_events,1); self.assertGreater(d.state_revisit_rate,0)
if __name__=='__main__': unittest.main()
