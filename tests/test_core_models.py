from __future__ import annotations
import unittest
import torch
from tfp.models import discover_model_plugins
from tfp.models.model_api import load_model_plugin
from tfp.policies import load_policy_plugin
from tfp.tasks import get_task
from tfp.utils import discover_maps


def model(name):
    return load_model_plugin(name)[1]((10, 5, 5), 6)

class TFPCoreModelTests(unittest.TestCase):
    def test_benchmark_counts(self):
        task=get_task('TFP-LocalTransport')
        self.assertEqual(len(discover_maps(task.train_map_dir)),54)
        self.assertEqual(len(discover_maps(task.test_map_dir)),18)

    def test_plugins_use_prefix_ids(self):
        plugins=discover_model_plugins()
        for name in ('001_simple_cnn','002_simple_cnn_tabu','003_action_memory_cnn','004_simple_cnn_tabux','005_action_memory_tabux_cnn'):
            self.assertIn(name,plugins)
        self.assertNotIn('simple_cnn_001',plugins)

    def test_legacy_alias_loads(self):
        spec,_=load_model_plugin('simple_cnn_001')
        self.assertEqual(spec.key,'001_simple_cnn')
        pspec,_=load_policy_plugin('ppo_categorical_001')
        self.assertEqual(pspec.key,'001_ppo_categorical')

    def test_tabu_breaks_forced_left_right_cycle(self):
        m=model('002_simple_cnn_tabu'); mask=torch.tensor([[1,1,1,1,0,0]],dtype=torch.bool); env_ids=(0,); executed=[]
        for step in range(10):
            proposed=torch.tensor([2 if step%2==0 else 3])
            logits=torch.tensor([[0.,.1,5. if step%2==0 else 1.,5. if step%2 else 1.,-5.,-5.]])
            action=int(m.rewrite_actions(proposed,logits,mask,env_ids).item()); executed.append(action)
            m.observe_behavior(env_ids,(action,),(0.,),(False,))
        self.assertTrue(any(a in (0,1) for a in executed[4:]))

    def test_default_policy_keeps_tabu_inference_only(self):
        m=model('002_simple_cnn_tabu'); policy=load_policy_plugin('001_ppo_categorical')[1](); env_ids=(0,)
        for a in [4,5,4,5]: m.observe_behavior(env_ids,(a,),(0.,),(False,))
        logits=torch.tensor([[-100.,-100.,-100.,-100.,100.,-100.]]); mask=torch.ones((1,6),dtype=torch.bool)
        self.assertNotIn(int(policy.greedy(m,logits,mask,env_ids).item()),(4,5))
        sampled,_,_=policy.sample(m,logits,mask,env_ids)
        self.assertEqual(int(sampled.item()),4)

    def test_action_memory_context_and_reset(self):
        m=model('003_action_memory_cnn'); x=torch.zeros((2,10,5,5))
        logits,value,context=m.act_forward(x,(0,1),update=True)
        self.assertEqual(tuple(logits.shape),(2,6)); self.assertEqual(tuple(value.shape),(2,)); self.assertEqual(tuple(context.shape),(2,8))
        m.observe_action_memory((0,1),(2,4),(0.,0.),(False,False))
        _,_,ctx=m.act_forward(x,(0,1),update=True); self.assertEqual(int(ctx[0,-1]),2); self.assertEqual(int(ctx[1,-1]),4)
        m.observe_action_memory((0,),(4,),(0.,),(True,)); _,_,after=m.act_forward(x[:1],(0,),update=True)
        self.assertTrue(torch.all(after==6))

if __name__=='__main__': unittest.main()
