import torch

from tfp.models import discover_model_plugins
from tfp.models.model_api import load_model_plugin, rollout_forward
from tfp.rewards import discover_reward_plugins
from tfp.tasks import create_task_env, get_task
from tfp.utils import discover_maps

TASK_ID = "TFP-KeyedHazardLogistics"


def _env(seed=131, view_size=7, wolves_enabled=False):
    task = get_task(TASK_ID)
    maps = discover_maps(task.test_map_dir)
    env = create_task_env(
        task.env_id, maps, view_size=view_size, seed=seed,
        reward_module=task.default_reward, wolves_enabled=wolves_enabled,
    )
    env.reset(seed=seed, map_path=maps[0])
    return env


def test_keyed_assets_scales_models_and_reward():
    task = get_task(TASK_ID)
    train = discover_maps(task.train_map_dir); test = discover_maps(task.test_map_dir)
    assert len(train) == 50 and len(test) == 20
    assert {p.stem.rsplit("_", 1)[-1] for p in train} == {"19x19", "23x23", "27x27", "31x31", "35x35"}
    assert set(discover_model_plugins(TASK_ID)) == {
        "001_dependency_film_shield", "002_event_memory_transformer", "003_dual_timescale_gru_shield"
    }
    assert set(discover_reward_plugins(TASK_ID)) == {"001_dependency_risk_potential"}


def test_keyed_observation_actions_and_structural_dependencies():
    for view in (5, 7):
        env = _env(view_size=view)
        assert env._observation().shape == (37, view, view)
        assert env.action_space_n == 8
        assert env.structurally_solvable()
        assert 1 <= len(env.lock_order_colors) <= 3
        assert 1 <= env.total_cargo <= 3
        assert env.valid_action_mask("task")[7]


def test_locked_frontier_requires_matching_key():
    env = _env()
    assert env.door_colors
    door, color = next(iter(env.door_colors.items()))
    assert color not in env.keys_owned
    # A colored frontier is physically blocked before its key is owned.
    assert not env._is_free(door)
    env.keys_owned.add(color)
    assert door not in env.open_doors
    # It remains closed but becomes an unlockable/toggleable door.
    assert not env._is_free(door)


def test_predator_contact_is_terminal_failure():
    env = _env(wolves_enabled=True)
    ar, ac = env.agent_pos
    step = next((a, (ar+dr, ac+dc)) for a,(dr,dc) in env.ACTIONS.items() if env._is_free((ar+dr,ac+dc)))
    action, pos = step
    env.wolf_positions[0] = pos
    _, reward, terminated, truncated, info = env.step(action)
    assert terminated and not truncated and not info["success"]
    assert info["failure_reason"] == "wolf_collision"
    assert info["hazard_collisions"] == 1 and info["collisions"] >= 1
    assert reward < -20


def test_dependency_reward_cannot_profit_from_wait_or_door_cycle():
    reward = _env().reward_function
    assert reward.compute({"wait_action": True}) < 0
    opened = reward.compute({"door_opened": True})
    closed = reward.compute({"door_closed": True})
    assert opened + closed < 0
    fwd = reward.compute({"progress_delta": 1})
    back = reward.compute({"progress_delta": -1})
    assert fwd + back < 0


def test_keyed_models_forward_and_memory_contracts():
    env = _env(); obs = env._observation()
    x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    for name in discover_model_plugins(TASK_ID):
        _, factory = load_model_plugin(name, task_id=TASK_ID)
        model = factory(tuple(obs.shape), env.action_space_n)
        logits, value, context = rollout_forward(model, x, (0,))
        assert logits.shape == (1, env.action_space_n) and value.shape == (1,)
        if name == "001_dependency_film_shield": assert context is None
        else: assert context is not None
        assert torch.isfinite(logits).all() and torch.isfinite(value).all()
