from __future__ import annotations

from collections import defaultdict, deque
from typing import Sequence

import torch
from torch import nn

from tfp.models._action_memory_core import LearnableActionMemory

OBS_CHANNELS = 24
DYNAMIC_START = 6
DYNAMIC_DIM = OBS_CHANNELS - DYNAMIC_START


def _context(x: torch.Tensor) -> torch.Tensor:
    return x[:, DYNAMIC_START:OBS_CHANNELS].mean(dim=(-2, -1))


def horizon_safety_prior_logits(x: torch.Tensor, action_count: int, view_size: int) -> torch.Tensor:
    """Multi-horizon interception prior with a bounded soft hazard shield."""
    b = x.shape[0]
    bias = torch.zeros((b, action_count), dtype=x.dtype, device=x.device)
    carrying = x[:, 6].mean(dim=(-2, -1)) > 0.5
    dy = x[:, 7].mean(dim=(-2, -1)); dx = x[:, 8].mean(dim=(-2, -1))
    future_dy = x[:, 18].mean(dim=(-2, -1)); future_dx = x[:, 19].mean(dim=(-2, -1))
    eta = x[:, 16].mean(dim=(-2, -1)); phase = x[:, 17].mean(dim=(-2, -1))
    turn = x[:, 23].mean(dim=(-2, -1)); slack = x[:, 22].mean(dim=(-2, -1))

    # Future carrier position matters most far from interception and around upcoming turns.
    horizon_mix = (~carrying).to(x.dtype) * torch.clamp(0.18 + 0.55 * eta + 0.35 * turn, 0.0, 0.80)
    guide_y = dy + horizon_mix * future_dy
    guide_x = dx + horizon_mix * future_dx
    denom = torch.maximum(guide_y.abs(), guide_x.abs()).clamp_min(1e-4)
    gy, gx = guide_y / denom, guide_x / denom
    bias[:, 0] += 3.2 * torch.relu(-gy)
    bias[:, 1] += 3.2 * torch.relu(gy)
    bias[:, 2] += 3.2 * torch.relu(-gx)
    bias[:, 3] += 3.2 * torch.relu(gx)

    pickup = x[:, 14].mean(dim=(-2, -1)) > 0.5
    delivery = x[:, 15].mean(dim=(-2, -1)) > 0.5
    bias[:, 4] += pickup.to(x.dtype) * 4.5
    bias[:, 5] += delivery.to(x.dtype) * 5.0
    at_waypoint = (dy.abs() + dx.abs()) < (0.28 / max(1, view_size))
    if action_count > 6:
        wait = at_waypoint & (~carrying) & (~pickup) & ((eta > 0.005) | (slack > 0.01) | (phase < 0.8))
        bias[:, 6] += wait.to(x.dtype) * (2.2 + 0.8 * torch.clamp(slack, 0, 1))

    # Two-step hazard projection from observed relative position + velocity.
    wy = x[:, 11].mean(dim=(-2, -1)); wx = x[:, 12].mean(dim=(-2, -1))
    wvy = x[:, 20].mean(dim=(-2, -1)); wvx = x[:, 21].mean(dim=(-2, -1))
    prox = x[:, 13].mean(dim=(-2, -1))
    py = wy + 0.22 * wvy; px = wx + 0.22 * wvx
    risk = torch.clamp((prox - 0.07) * 5.5, 0.0, 1.0)
    bias[:, 0] -= 4.0 * risk * (py < -1e-5).to(x.dtype)
    bias[:, 1] -= 4.0 * risk * (py > 1e-5).to(x.dtype)
    bias[:, 2] -= 4.0 * risk * (px < -1e-5).to(x.dtype)
    bias[:, 3] -= 4.0 * risk * (px > 1e-5).to(x.dtype)
    if action_count > 6:
        bias[:, 6] += 0.9 * torch.clamp((prox - 0.20) * 4.0, 0.0, 1.0)

    # Hard-looking but still soft logit suppression for an adjacent visible wolf.
    r = x.shape[-1] // 2; wolf = x[:, 4]
    if r >= 1:
        bias[:, 0] -= wolf[:, r-1, r] * 7.0
        bias[:, 1] -= wolf[:, r+1, r] * 7.0
        bias[:, 2] -= wolf[:, r, r-1] * 7.0
        bias[:, 3] -= wolf[:, r, r+1] * 7.0
    return bias


class GatedFiLMEncoder(nn.Module):
    def __init__(self, observation_shape: tuple[int, int, int], feature_dim: int = 208) -> None:
        super().__init__()
        channels, h, w = observation_shape
        if channels < OBS_CHANNELS:
            raise ValueError(f"Moving-cargo models require {OBS_CHANNELS} channels.")
        self.conv1 = nn.Sequential(nn.Conv2d(6, 40, 3, padding=1), nn.GELU())
        self.conv2 = nn.Sequential(nn.Conv2d(40, 64, 3, padding=1), nn.GELU())
        self.context = nn.Sequential(nn.Linear(DYNAMIC_DIM, 112), nn.LayerNorm(112), nn.GELU())
        self.film = nn.Linear(112, 128)
        self.gate = nn.Linear(112, 64)
        self.visual = nn.Sequential(nn.Conv2d(64,64,3,padding=1),nn.GELU(),nn.Flatten(),nn.Linear(64*h*w,192),nn.LayerNorm(192),nn.GELU())
        self.fusion = nn.Sequential(nn.Linear(192+112, feature_dim), nn.LayerNorm(feature_dim), nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx = self.context(_context(x))
        h = self.conv2(self.conv1(x[:, :6]))
        gamma,beta = self.film(ctx).chunk(2,1)
        gate = torch.sigmoid(self.gate(ctx)).unsqueeze(-1).unsqueeze(-1)
        h = h * (1 + 0.30*torch.tanh(gamma).unsqueeze(-1).unsqueeze(-1)) + 0.30*beta.unsqueeze(-1).unsqueeze(-1)
        h = h * (0.65 + 0.70*gate)
        return self.fusion(torch.cat([self.visual(h),ctx],1))


class HorizonFiLMShieldActorCritic(nn.Module):
    def __init__(self, observation_shape, action_count):
        super().__init__(); self.action_count=int(action_count); self.view_size=int(observation_shape[1])
        self.encoder=GatedFiLMEncoder(observation_shape,208)
        self.policy=nn.Sequential(nn.Linear(208,160),nn.GELU(),nn.Linear(160,action_count))
        self.value=nn.Sequential(nn.Linear(208,160),nn.GELU(),nn.Linear(160,1))
        nn.init.normal_(self.policy[-1].weight,0,0.006); nn.init.zeros_(self.policy[-1].bias)
    def forward(self,x):
        z=self.encoder(x); logits=horizon_safety_prior_logits(x,self.action_count,self.view_size)+0.52*torch.tanh(self.policy(z))
        return logits,self.value(z).squeeze(-1)


class CrossAttentionDynamicsActorCritic(nn.Module):
    """Telemetry history queries local spatial tokens instead of only pooled visual state."""
    def __init__(self, observation_shape, action_count):
        super().__init__(); self.action_count=int(action_count); self.view_size=int(observation_shape[1]); self.history_length=8
        _,h,w=observation_shape; self.h=h; self.w=w
        self.spatial=nn.Sequential(nn.Conv2d(6,48,3,padding=1),nn.GELU(),nn.Conv2d(48,72,3,padding=1),nn.GELU())
        self.telemetry_token=nn.Linear(DYNAMIC_DIM+1,72); self.pos=nn.Parameter(torch.zeros(1,self.history_length,72))
        enc=nn.TransformerEncoderLayer(d_model=72,nhead=4,dim_feedforward=176,dropout=0.0,activation='gelu',batch_first=True,norm_first=True)
        self.temporal=nn.TransformerEncoder(enc,num_layers=2,enable_nested_tensor=False)
        self.cross=nn.MultiheadAttention(72,4,batch_first=True,dropout=0.0)
        self.fusion=nn.Sequential(nn.Linear(72+72+DYNAMIC_DIM,220),nn.LayerNorm(220),nn.GELU(),nn.Linear(220,168),nn.GELU())
        self.policy=nn.Linear(168,action_count); self.value=nn.Linear(168,1)
        nn.init.normal_(self.policy.weight,0,0.006); nn.init.zeros_(self.policy.bias)
        self._history:dict[int,deque[torch.Tensor]]=defaultdict(lambda:deque(maxlen=self.history_length))
    def _empty(self,batch,device): return torch.zeros((batch,self.history_length,DYNAMIC_DIM+1),device=device)
    def _hist(self,env_ids,device):
        out=self._empty(len(env_ids),device)
        for i,raw in enumerate(env_ids):
            vals=list(self._history[int(raw)])
            if vals:
                start=self.history_length-len(vals); out[i,start:,:DYNAMIC_DIM]=torch.stack([v.to(device) for v in vals]); out[i,start:,DYNAMIC_DIM]=1
        return out
    def _heads(self,x,hist):
        fmap=self.spatial(x[:,:6]); tokens=fmap.flatten(2).transpose(1,2)
        present=hist[...,DYNAMIC_DIM]; temp=self.temporal(self.telemetry_token(hist)+self.pos)
        denom=present.sum(1,keepdim=True).clamp_min(1); summary=(temp*present.unsqueeze(-1)).sum(1)/denom
        query=summary.unsqueeze(1); attended,_=self.cross(query,tokens,tokens,need_weights=False); attended=attended.squeeze(1)
        ctx=_context(x); z=self.fusion(torch.cat([summary,attended,ctx],1))
        logits=horizon_safety_prior_logits(x,self.action_count,self.view_size)+0.52*torch.tanh(self.policy(z))
        return logits,self.value(z).squeeze(-1)
    def forward(self,x): return self._heads(x,self._empty(x.shape[0],x.device))
    def act_forward(self,x,env_ids:Sequence[int],update=True):
        hist=self._hist(env_ids,x.device); logits,value=self._heads(x,hist)
        if update:
            c=_context(x).detach().cpu()
            for i,raw in enumerate(env_ids): self._history[int(raw)].append(c[i])
        return logits,value,hist.detach()
    def training_forward(self,x,context): return self._heads(x,context.float())
    def observe_action_memory(self,env_ids,actions,rewards,dones):
        del actions,rewards
        for raw,done in zip(env_ids,dones):
            if done:self._history.pop(int(raw),None)
    def reset_sequence_memory(self,env_ids):
        for raw in env_ids:self._history.pop(int(raw),None)


class PhaseWorldGRUActorCritic(nn.Module):
    """Factorized carrier/hazard recurrent state with explicit action history and phase gate."""
    CARRIER=(7,8,9,10,16,17,18,19,22,23); HAZARD=(11,12,13,20,21)
    def __init__(self,observation_shape,action_count):
        super().__init__(); self.action_count=int(action_count); self.view_size=int(observation_shape[1]); self.history_length=10; self.pad_token=self.action_count
        self.carrier_hidden=88; self.hazard_hidden=64
        self.encoder=GatedFiLMEncoder(observation_shape,176); self.amem=LearnableActionMemory(action_count,self.history_length,12,36)
        self.carrier_gru=nn.GRUCell(len(self.CARRIER)+36,self.carrier_hidden); self.hazard_gru=nn.GRUCell(len(self.HAZARD)+36,self.hazard_hidden)
        self.phase_gate=nn.Sequential(nn.Linear(3,48),nn.GELU(),nn.Linear(48,3),nn.Softmax(dim=-1))
        self.fusion=nn.Sequential(nn.Linear(176+self.carrier_hidden+self.hazard_hidden+36,304),nn.LayerNorm(304),nn.GELU(),nn.Linear(304,192),nn.GELU())
        self.policy=nn.Linear(192,action_count); self.value=nn.Linear(192,1)
        nn.init.normal_(self.policy.weight,0,0.006); nn.init.zeros_(self.policy.bias)
        self._carrier={}; self._hazard={}; self._actions=defaultdict(lambda:deque(maxlen=self.history_length))
    def _empty_actions(self,b,d): return torch.full((b,self.history_length),self.pad_token,dtype=torch.long,device=d)
    def _actions_for(self,ids,d):
        rows=[]
        for raw in ids:
            vals=list(self._actions[int(raw)]); rows.append([self.pad_token]*(self.history_length-len(vals))+vals)
        return torch.tensor(rows,dtype=torch.long,device=d) if rows else self._empty_actions(0,d)
    def _hidden(self,store,dim,ids,d):
        rows=[store.get(int(raw),torch.zeros(dim,device=d)).to(d) for raw in ids]; return torch.stack(rows) if rows else torch.zeros((0,dim),device=d)
    @staticmethod
    def _sel(x,idx): return torch.stack([x[:,i].mean((-2,-1)) for i in idx],1)
    def _pack(self,c,h,a): return torch.cat([c,h,a.float()],1).detach()
    def _unpack(self,ctx):
        a=self.carrier_hidden;b=a+self.hazard_hidden;return ctx[:,:a].float(),ctx[:,a:b].float(),ctx[:,b:b+self.history_length].round().long()
    def _heads(self,x,c,h,a):
        vis=self.encoder(x); am=self.amem(a); cn=self.carrier_gru(torch.cat([self._sel(x,self.CARRIER),am],1),c); hn=self.hazard_gru(torch.cat([self._sel(x,self.HAZARD),am],1),h)
        z=self.fusion(torch.cat([vis,cn,hn,am],1)); residual=self.policy(z)
        carrying=x[:,6].mean((-2,-1)); pickup=x[:,14].mean((-2,-1)); delivery=x[:,15].mean((-2,-1)); gate=self.phase_gate(torch.stack([1-carrying,carrying,pickup+delivery],1))
        scale=(0.44+0.12*gate[:,2]).unsqueeze(1)
        logits=horizon_safety_prior_logits(x,self.action_count,self.view_size)+scale*torch.tanh(residual)
        return logits,self.value(z).squeeze(-1),cn,hn
    def forward(self,x):
        c=torch.zeros((x.shape[0],self.carrier_hidden),device=x.device);h=torch.zeros((x.shape[0],self.hazard_hidden),device=x.device);a=self._empty_actions(x.shape[0],x.device);log,v,_,_=self._heads(x,c,h,a);return log,v
    def act_forward(self,x,env_ids,update=True):
        c=self._hidden(self._carrier,self.carrier_hidden,env_ids,x.device);h=self._hidden(self._hazard,self.hazard_hidden,env_ids,x.device);a=self._actions_for(env_ids,x.device);ctx=self._pack(c,h,a);log,v,cn,hn=self._heads(x,c,h,a)
        if update:
            for i,raw in enumerate(env_ids): self._carrier[int(raw)]=cn[i].detach();self._hazard[int(raw)]=hn[i].detach()
        return log,v,ctx
    def training_forward(self,x,context): c,h,a=self._unpack(context);log,v,_,_=self._heads(x,c,h,a);return log,v
    def training_sequence_forward(self,x_seq,initial_context,actions_seq,dones_seq):
        c,h,a=self._unpack(initial_context);logs=[];vals=[]
        for t in range(x_seq.shape[0]):
            log,v,cn,hn=self._heads(x_seq[t],c,h,a);logs.append(log);vals.append(v);a=torch.cat([a[:,1:],actions_seq[t].long().unsqueeze(1)],1);done=dones_seq[t].bool().unsqueeze(1);c=torch.where(done,torch.zeros_like(cn),cn);h=torch.where(done,torch.zeros_like(hn),hn);a=torch.where(done.expand_as(a),torch.full_like(a,self.pad_token),a)
        return torch.stack(logs),torch.stack(vals)
    def observe_action_memory(self,env_ids,actions,rewards,dones):
        del rewards
        for raw,action,done in zip(env_ids,actions,dones):
            k=int(raw);self._actions[k].append(int(action))
            if done:self._actions.pop(k,None);self._carrier.pop(k,None);self._hazard.pop(k,None)
    def reset_action_memory(self,env_ids):
        for raw in env_ids:self._actions.pop(int(raw),None)
    def reset_sequence_memory(self,env_ids):
        for raw in env_ids:self._carrier.pop(int(raw),None);self._hazard.pop(int(raw),None)
