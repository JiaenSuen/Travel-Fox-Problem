from pathlib import Path

import torch

from tfp.models.model_api import load_model_plugin, rollout_forward
from tfp.tasks import create_task_env, get_task

TASK_ID = "TFP-FoxRoomTransport-Local"


def test_route_prior_prefers_observed_waypoint_direction():
    torch.manual_seed(0)
    task = get_task(TASK_ID)
    map_path = sorted(Path(task.test_map_dir).glob("test_small_*.txt"))[0]
    env = create_task_env(TASK_ID, [map_path], view_size=7, seed=107, reward_module="001_actionable_geodesic")
    obs, _ = env.reset(seed=107, map_path=map_path)
    _, factory = load_model_plugin("001_route_prior_cnn", task_id=TASK_ID)
    model = factory(tuple(obs.shape), env.action_space_n)
    x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    logits, _, _ = rollout_forward(model, x, (0,))
    mask = torch.as_tensor(env.valid_action_mask("task"), dtype=torch.bool)
    action = int(logits[0].masked_fill(~mask, -1e9).argmax().item())
    waypoint = env._route_waypoint()
    dr = waypoint[0] - env.agent_pos[0]
    dc = waypoint[1] - env.agent_pos[1]
    preferred = set()
    if dr < 0: preferred.add(0)
    if dr > 0: preferred.add(1)
    if dc < 0: preferred.add(2)
    if dc > 0: preferred.add(3)
    # If the direct preferred movement is locally valid, the residual initialization
    # must not erase the route prior before any PPO update.
    valid_preferred = {a for a in preferred if bool(mask[a])}
    if valid_preferred:
        assert action in valid_preferred
