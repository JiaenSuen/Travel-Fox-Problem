from pathlib import Path

from tfp.tasks import create_task_env, get_task


TASK_ID = "TFP-FoxRoomTransport-Local"


def _env(seed: int = 107):
    task = get_task(TASK_ID)
    maps = sorted(Path(task.test_map_dir).glob("*.txt"))
    env = create_task_env(
        TASK_ID,
        maps,
        observation_mode="local",
        view_size=7,
        seed=seed,
        reward_module="001_actionable_geodesic",
    )
    env.reset(seed=seed, map_path=maps[0])
    return env


def test_v5_default_reward_and_models_are_replaced():
    from tfp.models import discover_model_plugins
    from tfp.rewards import discover_reward_plugins

    task = get_task(TASK_ID)
    assert task.default_reward == "001_actionable_geodesic"
    assert task.default_model == "001_route_prior_cnn"
    assert set(discover_reward_plugins(TASK_ID)) == {"001_actionable_geodesic"}
    assert set(discover_model_plugins(TASK_ID)) == {
        "001_route_prior_cnn",
        "002_route_prior_action_memory",
        "003_route_prior_gru_memory",
    }


def test_actionable_reward_has_no_profitable_open_close_loop():
    env = _env()
    reward = env.reward_function
    open_step = reward.compute({"actionable_delta": 1, "door_opened": True})
    close_step = reward.compute({"actionable_delta": -1, "door_closed": True})
    assert open_step + close_step < 0.0


def test_irrelevant_door_open_has_no_standalone_bonus():
    env = _env()
    reward = env.reward_function
    ordinary = reward.compute({"actionable_delta": 0})
    irrelevant_open = reward.compute({"actionable_delta": 0, "door_opened": True})
    assert irrelevant_open == ordinary


def test_room_observation_contains_nonzero_hierarchical_waypoint():
    env = _env()
    obs = env._observation()
    # Object is intentionally sampled outside the spawn room; at least one waypoint
    # coordinate should normally be non-zero and remains bounded for stable learning.
    assert abs(float(obs[7, 0, 0])) <= 1.0
    assert abs(float(obs[8, 0, 0])) <= 1.0
    assert abs(float(obs[7, 0, 0])) + abs(float(obs[8, 0, 0])) > 0.0
