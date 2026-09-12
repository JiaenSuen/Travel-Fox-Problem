from torch import nn
from ._advanced_core import PhaseWorldGRUActorCritic
MODEL_SPEC={"key":"003_phase_world_gru","display_name":"003 · Phase-Gated World-State GRU","algorithm":"ppo","recurrent":True,"memory_type":"carrier_gru+hazard_gru+action_memory","description":"Factorized carrier/hazard recurrent state, action memory, and phase-gated residual control for long route programs."}
def create_model(observation_shape:tuple[int,int,int],action_count:int)->nn.Module:return PhaseWorldGRUActorCritic(observation_shape,action_count)
