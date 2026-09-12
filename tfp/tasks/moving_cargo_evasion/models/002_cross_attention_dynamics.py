from torch import nn
from ._advanced_core import CrossAttentionDynamicsActorCritic
MODEL_SPEC={"key":"002_cross_attention_dynamics","display_name":"002 · Cross-Attention Dynamics Transformer","algorithm":"ppo","recurrent":False,"memory_type":"telemetry_transformer+spatial_cross_attention","description":"Eight-step motion telemetry attends directly to local spatial tokens, improving junction and hazard-conditioned interception decisions."}
def create_model(observation_shape:tuple[int,int,int],action_count:int)->nn.Module:return CrossAttentionDynamicsActorCritic(observation_shape,action_count)
