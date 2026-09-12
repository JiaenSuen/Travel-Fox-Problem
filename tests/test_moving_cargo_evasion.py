from pathlib import Path

import numpy as np
import torch

from tfp.models import discover_model_plugins
from tfp.models.model_api import load_model_plugin, rollout_forward
from tfp.rewards import discover_reward_plugins
from tfp.tasks import create_task_env, discover_tasks, get_task
from tfp.utils import discover_maps

TASK_ID = "TFP-MovingCargoEvasion"


def _env(seed: int = 109, view_size: int = 7):
    task = get_task(TASK_ID)
    maps = discover_maps(task.test_map_dir)
    env = create_task_env(
        task.env_id,
        maps,
        view_size=view_size,
        seed=seed,
        reward_module=task.default_reward,
    )
    env.reset(seed=seed, map_path=maps[0])
    return env


def test_task_names_have_no_fox_or_order_marker():
    for task in discover_tasks().values():
        for name in (task.env_id, task.code, task.display_name):
            assert "fox" not in name.lower()
        assert not task.code.endswith(("-L1", "-L2", "-L3", "-L4"))


def test_moving_cargo_assets_models_reward_and_views():
    task = get_task(TASK_ID)
    train = discover_maps(task.train_map_dir)
    test = discover_maps(task.test_map_dir)
    assert len(train) == 45
    assert len(test) == 15
    assert {tuple(map(int, p.stem.rsplit("_", 1)[-1].split("x"))) for p in train} == {
        (12, 12), (16, 16), (20, 20)
    }
    assert set(discover_model_plugins(TASK_ID)) == {
        "001_intercept_safety_cnn",
        "002_intercept_action_memory",
        "003_intercept_gru_memory",
    }
    assert set(discover_reward_plugins(TASK_ID)) == {"001_intercept_safety_potential"}

    for view_size in (5, 7):
        env = _env(view_size=view_size)
        obs = env._observation()
        assert obs.shape == (18, view_size, view_size)
        assert env.action_space_n == 7
        assert env.valid_action_mask("task")[6]  # WAIT


def test_track_is_one_closed_branch_free_cycle():
    env = _env()
    track = set(env.track_cells)
    assert len(track) >= 8
    for r, c in track:
        degree = sum((r + dr, c + dc) in track for dr, dc in env.ACTIONS.values())
        assert degree == 2
    assert len(env.track_cells) == len(track)


def test_carrier_moves_with_defined_cadence():
    env = _env()
    start = env.vehicle_pos
    # Keep the wolf out of the way so this test isolates carrier dynamics.
    env.wolf_pos = env.goal_pos
    env.wolf_target = env.goal_pos
    env._wolf_target_field = env._distance_field(env.goal_pos)
    env._advance_wolf = lambda: None
    for _ in range(env.VEHICLE_CADENCE):
        env.step(6)
    assert env.vehicle_pos != start
    assert env.vehicle_moves == 1


def test_contact_with_wolf_terminates_as_failure():
    env = _env()
    ar, ac = env.agent_pos
    chosen = None
    for action, (dr, dc) in env.ACTIONS.items():
        pos = (ar + dr, ac + dc)
        if env._is_free(pos):
            chosen = (action, pos)
            break
    assert chosen is not None
    action, pos = chosen
    env.wolf_pos = pos
    _, reward, terminated, truncated, info = env.step(action)
    assert terminated and not truncated
    assert not info["success"]
    assert info["failure_reason"] == "wolf_collision"
    assert info["hazard_collisions"] == 1
    assert reward < -10.0


def test_intercept_reward_is_symmetric_and_collision_dominates():
    reward = _env().reward_function
    forward = reward.compute({"navigation_delta": 1, "wolf_distance": 99})
    reverse = reward.compute({"navigation_delta": -1, "wolf_distance": 99})
    assert forward + reverse < 0.0  # two step costs remain; no progress loop profit
    collision = reward.compute({"wolf_collision": True, "wolf_distance": 0})
    assert collision <= -15.0


def test_untrained_residual_model_obeys_visible_intercept_prior():
    torch.manual_seed(0)
    env = _env()
    obs = env._observation()
    _, factory = load_model_plugin("001_intercept_safety_cnn", task_id=TASK_ID)
    model = factory(tuple(obs.shape), env.action_space_n)
    x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    logits, _, _ = rollout_forward(model, x, (0,))
    mask = torch.as_tensor(env.valid_action_mask("task"), dtype=torch.bool)
    action = int(logits[0].masked_fill(~mask, -1e9).argmax().item())
    waypoint = env._route_waypoint()
    dy = waypoint[0] - env.agent_pos[0]
    dx = waypoint[1] - env.agent_pos[1]
    preferred = set()
    if dy < 0: preferred.add(0)
    if dy > 0: preferred.add(1)
    if dx < 0: preferred.add(2)
    if dx > 0: preferred.add(3)
    valid_preferred = {a for a in preferred if bool(mask[a])}
    if valid_preferred:
        assert action in valid_preferred
    else:
        assert action == 6 or bool(mask[action])
