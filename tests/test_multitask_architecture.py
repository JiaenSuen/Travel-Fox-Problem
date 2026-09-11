from __future__ import annotations

from pathlib import Path

from tfp.models import discover_model_plugins
from tfp.rewards import discover_reward_plugins
from tfp.reporting import build_task_report
from tfp.tasks import create_task_env, get_task
from tfp.utils import discover_maps


def test_task_scoped_plugins_are_disjoint():
    transport = discover_model_plugins("TFP-FoxTransport-Local")
    color = discover_model_plugins("TFP-FoxColorSort-Local")
    assert "001_simple_cnn" in transport
    assert "001_color_cnn" in color
    assert "001_color_cnn" not in transport
    assert "001_simple_cnn" not in color
    assert "001_dense_transport" in discover_reward_plugins("TFP-FoxTransport-Local")
    assert "001_dense_color_sort" in discover_reward_plugins("TFP-FoxColorSort-Local")


def test_color_sort_map_counts_and_views():
    task = get_task("TFP-FoxColorSort-Local")
    train = discover_maps(task.train_map_dir)
    test = discover_maps(task.test_map_dir)
    assert len(train) == 45
    assert len(test) == 15
    for view in (5, 7):
        env = create_task_env(task.env_id, test[:1], view_size=view, reward_module=task.default_reward)
        obs, info = env.reset(seed=103, map_path=test[0])
        assert obs.shape == (26, view, view)
        assert 2 <= info["items_total"] <= 5


def test_color_sort_requires_matching_goal():
    task = get_task("TFP-FoxColorSort-Local")
    maps = discover_maps(task.test_map_dir)
    env = create_task_env(task.env_id, maps[:1], reward_module=task.default_reward)
    env.reset(seed=103, map_path=maps[0])
    color = next(iter(env.objects))
    env.agent_pos = env.objects[color]
    _, _, _, _, _ = env.step(4)
    assert env.carrying_color == color
    wrong_goals = [pos for c, pos in env.goals.items() if c != color]
    if wrong_goals:
        env.agent_pos = wrong_goals[0]
        _, _, terminated, _, info = env.step(5)
        assert not terminated
        assert env.carrying_color == color
        assert info["invalid_actions"] >= 1


def test_result_report_uses_stable_files(tmp_path: Path):
    readme, image = build_task_report(tmp_path, "FOX-TEST", "Test Task")
    assert readme.name == "README.md"
    assert image.name == "summary.png"
    assert readme.exists() and image.exists()


def test_room_transport_task_assets_and_views():
    task = get_task("TFP-FoxRoomTransport-Local")
    train = discover_maps(task.train_map_dir)
    test = discover_maps(task.test_map_dir)
    assert len(train) == 50
    assert len(test) == 20
    sizes = {tuple(map(int, p.stem.rsplit("_", 1)[-1].split("x"))) for p in train}
    assert sizes == {(15, 15), (19, 19), (23, 23), (27, 27), (31, 31)}
    models = discover_model_plugins(task.env_id)
    rewards = discover_reward_plugins(task.env_id)
    assert set(models) == {"001_route_prior_cnn", "002_route_prior_action_memory", "003_route_prior_gru_memory"}
    assert set(rewards) == {"001_actionable_geodesic"}
    for view in (5, 7):
        env = create_task_env(task.env_id, test[:1], view_size=view, reward_module=task.default_reward)
        obs, info = env.reset(seed=107, map_path=test[0])
        assert obs.shape == (14, view, view)
        assert env.action_space_n == 7
        assert info["rooms"] >= 4
        assert info["object_room"] != info["goal_room"]
        assert info["doors_total"] >= 4


def test_room_transport_closed_door_toggle_changes_traversability():
    task = get_task("TFP-FoxRoomTransport-Local")
    maps = discover_maps(task.test_map_dir)
    env = create_task_env(task.env_id, maps[:1], reward_module=task.default_reward)
    env.reset(seed=107, map_path=maps[0])
    closed = next(iter(env.door_positions - env.open_doors))
    adjacent = None
    for dr, dc in env.ACTIONS.values():
        candidate = (closed[0] + dr, closed[1] + dc)
        if candidate in env.room_index:
            adjacent = candidate
            break
    assert adjacent is not None
    env.agent_pos = adjacent
    assert not env._is_free(closed)
    assert env.valid_action_mask("task")[6]
    env.step(6)
    assert closed in env.open_doors
    assert env._is_free(closed)
