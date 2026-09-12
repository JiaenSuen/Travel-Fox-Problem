from torch import nn
from ._advanced_core import HorizonFiLMShieldActorCritic
MODEL_SPEC={"key":"001_horizon_film_shield","display_name":"001 · Horizon-FiLM Safety Shield","algorithm":"ppo","recurrent":False,"memory_type":"predictive_context","description":"Multi-horizon carrier guidance, gated FiLM fusion, and bounded PPO residuals over a predictive hazard shield."}
def create_model(observation_shape:tuple[int,int,int],action_count:int)->nn.Module:return HorizonFiLMShieldActorCritic(observation_shape,action_count)
