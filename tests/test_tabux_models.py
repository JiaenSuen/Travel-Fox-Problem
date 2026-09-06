from __future__ import annotations
import unittest, torch
from tfp.models.model_api import load_model_plugin
from tfp.policies import load_policy_plugin
from tfp.training.ppo import _validation_rank

def model(n): return load_model_plugin(n)[1]((10,5,5),6)
class TFPTabuXModelTests(unittest.TestCase):
    def test_tabux_escape(self):
        m=model('004_simple_cnn_tabux'); env=(0,)
        for a in [3,3,2,2,3,3,2,2]: m.observe_behavior(env,(a,),(0.,),(False,))
        mask=torch.tensor([[1,1,1,1,0,0]],dtype=torch.bool); logits=torch.tensor([[4.,3.,1.,8.,-9.,-9.]])
        self.assertIn(int(m.rewrite_actions(torch.tensor([3]),logits,mask,env).item()),(0,1))
    def test_tabux_inference_only(self):
        m=model('004_simple_cnn_tabux'); p=load_policy_plugin('001_ppo_categorical')[1](); env=(0,)
        for a in [3,3,2,2,3,3,2,2]: m.observe_behavior(env,(a,),(0.,),(False,))
        logits=torch.tensor([[-100.,-100.,-100.,100.,-100.,-100.]]); mask=torch.ones((1,6),dtype=torch.bool)
        sampled,_,_=p.sample(m,logits,mask,env); self.assertEqual(int(sampled.item()),3)
    def test_action_memory_tabux(self):
        m=model('005_action_memory_tabux_cnn'); x=torch.zeros((1,10,5,5)); m.observe_action_memory((0,),(3,),(0.,),(False,))
        logits,value,ctx=m.act_forward(x,(0,),update=True); self.assertEqual(int(ctx[0,-1]),3); self.assertEqual(tuple(logits.shape),(1,6))
        logits2,value2=m.training_forward(x,ctx); self.assertEqual(tuple(logits2.shape),(1,6)); self.assertEqual(tuple(value2.shape),(1,))
    def test_validation_rank(self):
        self.assertGreater(_validation_rank(.8,.6,50),_validation_rank(.7,1.,10)); self.assertGreater(_validation_rank(.8,.7,40),_validation_rank(.8,.7,60))
if __name__=='__main__': unittest.main()
