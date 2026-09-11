from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Callable, Sequence

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from tfp.checkpoints import resolve_checkpoint_path
from tfp.models.model_api import load_model_plugin, optimization_forward, peek_forward, rollout_forward
from tfp.intrinsic import load_intrinsic_plugin
from tfp.policies import load_policy_plugin
from tfp.runtime import resolve_device
from tfp.tasks import create_task_env, get_task
from tfp.utils import set_seed


@dataclass(frozen=True)
class PPOConfig:
    timesteps: int = 120_000
    num_envs: int = 16
    rollout: int = 64
    epochs: int = 4
    minibatch: int = 256
    gamma: float = 0.98
    gae_lambda: float = 0.95
    clip: float = 0.2
    lr: float = 4e-4
    entropy: float = 0.012
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    eval_every: int = 20_000
    seed: int = 7
    task_id: str = "TFP-FoxTransport-Local"
    observation_mode: str = "local"
    view_size: int = 5
    action_mask_mode: str = "task"
    curriculum: bool = True
    reward_module: str = "001_dense_transport"
    policy_module: str = "001_ppo_categorical"
    intrinsic_module: str = "auto"
    intrinsic_coef: float = -1.0
    device: str = "cuda"
    experiment_tag: str = ""


def _distribution(logits: torch.Tensor, mask: torch.Tensor) -> Categorical:
    return Categorical(logits=logits.masked_fill(~mask, -1e9))


def _map_size_key(path: Path) -> tuple[int, int]:
    match = re.search(r"_(\d+)x(\d+)(?:\.|$)", path.name)
    if not match:
        return (10**9, 10**9)
    return int(match.group(1)), int(match.group(2))


def _maps_for_progress(all_maps: Sequence[Path], progress: float, curriculum: bool) -> list[Path]:
    if not curriculum:
        return list(all_maps)
    size_groups = sorted({_map_size_key(p) for p in all_maps})
    if not size_groups or size_groups[0][0] >= 10**9:
        return list(all_maps)
    # Progressive size curriculum. With the legacy three-size tasks this remains
    # equivalent to 1 -> 2 -> 3 groups; tasks with more scale tiers (e.g. FOX-RM-L3)
    # unlock one additional tier at a time across the training horizon.
    if len(size_groups) == 1:
        allowed_count = 1
    else:
        allowed_count = min(len(size_groups), max(1, int(progress * len(size_groups)) + 1))
    allowed = set(size_groups[:allowed_count])
    selected = [p for p in all_maps if _map_size_key(p) in allowed]
    return selected or list(all_maps)


def _quick_eval(
    model: nn.Module,
    policy,
    env,
    test_maps: Sequence[Path],
    device: torch.device,
    mask_mode: str,
) -> tuple[float, float, float]:
    model.eval()
    successes: list[int] = []
    lengths: list[int] = []
    efficiencies: list[float] = []
    seeds = get_task(env.TASK_ID).default_eval_seeds[:2]
    eval_env_id = (1_000_000,)
    with torch.no_grad():
        for map_path in test_maps:
            for seed in seeds:
                policy.reset(model, eval_env_id)
                obs, _ = env.reset(seed=seed, map_path=map_path)
                for _ in range(env.max_steps):
                    x = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    logits, _, _ = rollout_forward(model, x, eval_env_id)
                    mask = torch.as_tensor(env.valid_action_mask(mask_mode), dtype=torch.bool, device=device).unsqueeze(0)
                    action_t = policy.greedy(model, logits, mask, eval_env_id)
                    action = int(action_t.item())
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = bool(terminated or truncated)
                    policy.observe(model, eval_env_id, (action,), (reward,), (done,))
                    if done:
                        successes.append(int(info["success"]))
                        lengths.append(int(info["steps"]))
                        efficiencies.append(float(info["path_efficiency"]))
                        break
    model.train()
    return float(np.mean(successes)), float(np.mean(lengths)), float(np.mean(efficiencies))




def _selection_validation_maps(train_maps: Sequence[Path]) -> list[Path]:
    """Small deterministic validation subset used for checkpoint selection.

    Formal unseen test maps must not choose the exported checkpoint. TFP therefore
    selects a compact set of training-layout maps (two per map size when available)
    and evaluates them with fixed validation seeds. This is intentionally lightweight:
    it avoids test leakage while keeping checkpoint selection fast.
    """
    selected: list[Path] = []
    for size in sorted({_map_size_key(p) for p in train_maps}):
        group = sorted(p for p in train_maps if _map_size_key(p) == size)
        selected.extend(group[:2])
    return selected or list(train_maps[: min(6, len(train_maps))])


def _checkpoint_payload(
    *,
    model: nn.Module,
    model_module: str,
    model_display_name: str,
    config: PPOConfig,
    observation_shape: tuple[int, ...],
    action_count: int,
    role: str,
    global_step: int,
    selection_metrics: dict[str, float] | None = None,
) -> dict[str, object]:
    return {
        "task_id": config.task_id,
        "algorithm": "ppo",
        "model_module": model_module,
        "model_display_name": model_display_name,
        "policy_module": config.policy_module,
        "reward_module": config.reward_module,
        "model_state": model.state_dict(),
        "observation_shape": observation_shape,
        "action_count": action_count,
        "view_size": config.view_size,
        "observation_mode": config.observation_mode,
        "action_mask_mode": config.action_mask_mode,
        "seed": config.seed,
        "device": config.device,
        "experiment_tag": config.experiment_tag,
        "training_config": asdict(config),
        "checkpoint_role": role,
        "checkpoint_step": int(global_step),
        "selection_metrics": selection_metrics or {},
    }


def _validation_rank(success: float, efficiency: float, mean_steps: float) -> tuple[float, float, float]:
    """Higher tuple is better: success, then path efficiency, then fewer steps."""
    return (float(success), float(efficiency), -float(mean_steps))


def train_ppo(
    model_module: str,
    train_maps: Sequence[Path],
    test_maps: Sequence[Path],
    checkpoint_path: str | Path,
    config: PPOConfig | None = None,
    log_callback: Callable[[str], None] | None = None,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
) -> Path:
    config = config or PPOConfig()
    run_mask_mode = str(config.action_mask_mode)
    if run_mask_mode not in {"task", "valid"}:
        raise ValueError(f"Unsupported action mask mode: {run_mask_mode}")
    set_seed(config.seed)
    device = resolve_device(config.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    model_spec, factory = load_model_plugin(model_module, task_id=config.task_id)
    policy_spec, policy_factory = load_policy_plugin(config.policy_module)
    if model_spec.algorithm.lower() != "ppo":
        raise ValueError("The PPO trainer accepts PPO-compatible model plugins only.")
    if policy_spec.algorithm.lower() != "ppo":
        raise ValueError(f"Policy {config.policy_module} is not compatible with PPO.")
    policy = policy_factory()
    if log_callback:
        log_callback(f"Protocol lock: action_mask={run_mask_mode} (immutable for this training run).")

    task = get_task(config.task_id)
    if config.view_size not in task.supported_view_sizes:
        raise ValueError(f"{task.env_id} supports view sizes {task.supported_view_sizes}.")
    initial_maps = _maps_for_progress(train_maps, 0.0, config.curriculum)
    envs = [
        create_task_env(
            config.task_id, initial_maps,
            observation_mode=config.observation_mode,
            view_size=config.view_size,
            seed=config.seed + i,
            reward_module=config.reward_module,
        )
        for i in range(config.num_envs)
    ]
    test_env = create_task_env(
        config.task_id, test_maps,
        observation_mode=config.observation_mode,
        view_size=config.view_size,
        seed=config.seed + 10000,
        reward_module=config.reward_module,
    )
    validation_maps = _selection_validation_maps(train_maps)
    validation_env = create_task_env(
        config.task_id, validation_maps,
        observation_mode=config.observation_mode,
        view_size=config.view_size,
        seed=config.seed + 20000,
        reward_module=config.reward_module,
    )
    obs = np.stack([env.reset()[0] for env in envs])
    model = factory(tuple(obs.shape[1:]), envs[0].action_space_n).to(device)

    bootstrap_from = getattr(model_spec, "bootstrap_from", "") or ""
    if bootstrap_from:
        bootstrap_path = resolve_checkpoint_path(config.task_id, bootstrap_from, role="last")
        init_hook = getattr(model, "initialize_from_feedforward_state", None)
        if not bootstrap_path.exists():
            raise FileNotFoundError(
                f"Model {model_module} requires bootstrap checkpoint {bootstrap_from}, but {bootstrap_path} does not exist."
            )
        if not callable(init_hook):
            raise ValueError(f"Model {model_module} declares bootstrap_from but has no initialization hook.")
        payload = torch.load(bootstrap_path, map_location=device)
        init_hook(payload["model_state"])
        if log_callback:
            log_callback(f"Bootstrapped recurrent model from {bootstrap_path.name}.")

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, eps=1e-5)
    train_env_ids = tuple(range(config.num_envs))
    policy.reset(model, train_env_ids)

    intrinsic_name = config.intrinsic_module
    if intrinsic_name == "auto":
        intrinsic_name = getattr(model_spec, "intrinsic_module", "none") or "none"
    intrinsic = None
    intrinsic_spec = None
    intrinsic_coef = 0.0
    intrinsic_anneal_fraction = 0.0
    if intrinsic_name != "none":
        intrinsic_spec, intrinsic_factory = load_intrinsic_plugin(intrinsic_name)
        intrinsic = intrinsic_factory(tuple(obs.shape[1:]), envs[0].action_space_n, device)
        intrinsic_coef = float(config.intrinsic_coef if config.intrinsic_coef >= 0.0 else intrinsic_spec.default_coef)
        intrinsic_anneal_fraction = float(getattr(intrinsic_spec, "anneal_fraction", 0.0) or 0.0)
        intrinsic.reset(train_env_ids)

    steps_per_update = config.num_envs * config.rollout
    updates = max(1, int(np.ceil(config.timesteps / steps_per_update)))
    global_step = 0
    next_eval = config.eval_every
    recent_success: list[int] = []
    last_loss = 0.0
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = checkpoint_path.with_name(f"{checkpoint_path.stem}.best{checkpoint_path.suffix}")
    best_rank: tuple[float, float, float] | None = None
    best_metrics: dict[str, float] | None = None

    if log_callback:
        gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
        intrinsic_label = intrinsic_spec.display_name if intrinsic_spec is not None else "None"
        log_callback(
            f"Runtime device={device.type} ({gpu}); model={model_spec.display_name}; "
            f"policy={policy_spec.display_name}; reward={config.reward_module}; "
            f"intrinsic={intrinsic_label} coef={intrinsic_coef:.4f}"
        )

    intrinsic_loss = 0.0
    for update in range(1, updates + 1):
        progress = min(1.0, global_step / max(1, config.timesteps))
        if intrinsic is not None and intrinsic_anneal_fraction > 0.0:
            intrinsic_multiplier = max(0.0, 1.0 - progress / intrinsic_anneal_fraction)
        else:
            intrinsic_multiplier = 1.0
        effective_intrinsic_coef = intrinsic_coef * intrinsic_multiplier
        progress_hook = getattr(model, "set_training_progress", None)
        if callable(progress_hook):
            progress_hook(progress)
        active_maps = _maps_for_progress(train_maps, progress, config.curriculum)
        for env in envs:
            env.map_paths = active_maps

        obs_buf = np.zeros((config.rollout, config.num_envs, *obs.shape[1:]), dtype=np.float32)
        actions_buf = np.zeros((config.rollout, config.num_envs), dtype=np.int64)
        masks_buf = np.zeros((config.rollout, config.num_envs, envs[0].action_space_n), dtype=bool)
        logp_buf = np.zeros((config.rollout, config.num_envs), dtype=np.float32)
        rewards_buf = np.zeros((config.rollout, config.num_envs), dtype=np.float32)
        dones_buf = np.zeros((config.rollout, config.num_envs), dtype=np.float32)
        values_buf = np.zeros((config.rollout, config.num_envs), dtype=np.float32)
        next_obs_buf = np.zeros_like(obs_buf)
        intrinsic_buf = np.zeros((config.rollout, config.num_envs), dtype=np.float32)
        context_buf = None

        for t in range(config.rollout):
            obs_buf[t] = obs
            action_masks = np.stack([env.valid_action_mask(run_mask_mode) for env in envs])
            masks_buf[t] = action_masks
            x = torch.as_tensor(obs, dtype=torch.float32, device=device)
            mask_t = torch.as_tensor(action_masks, dtype=torch.bool, device=device)
            with torch.no_grad():
                logits, values, context = rollout_forward(model, x, train_env_ids)
                sample_out = policy.sample(model, logits, mask_t, train_env_ids)
                if len(sample_out) == 4:
                    actions, logp, _, behavior_mask = sample_out
                    masks_buf[t] = behavior_mask.detach().cpu().numpy()
                else:
                    actions, logp, _ = sample_out
            if context is not None:
                context_np = context.detach().cpu().numpy()
                if context_buf is None:
                    # Preserve the model context dtype. Action-memory plugins return
                    # integer action-token histories while feature-memory plugins may
                    # return floating point contexts.
                    context_buf = np.zeros(
                        (config.rollout, config.num_envs, *context_np.shape[1:]),
                        dtype=context_np.dtype,
                    )
                context_buf[t] = context_np

            actions_np = actions.cpu().numpy()
            actions_buf[t] = actions_np
            logp_buf[t] = logp.cpu().numpy()
            values_buf[t] = values.cpu().numpy()

            next_obs = []
            transition_next_obs = []
            step_rewards: list[float] = []
            step_dones: list[bool] = []
            for i, env in enumerate(envs):
                new_obs, reward, terminated, truncated, info = env.step(int(actions_np[i]))
                done = bool(terminated or truncated)
                rewards_buf[t, i] = reward
                dones_buf[t, i] = float(done)
                step_rewards.append(float(reward))
                step_dones.append(done)
                transition_next_obs.append(new_obs)
                if done:
                    recent_success.append(int(info["success"]))
                    recent_success = recent_success[-100:]
                    new_obs, _ = env.reset()
                next_obs.append(new_obs)

            transition_next_obs_np = np.stack(transition_next_obs)
            next_obs_buf[t] = transition_next_obs_np
            if intrinsic is not None:
                bonus = intrinsic.compute(obs, transition_next_obs_np, actions_np, train_env_ids, step_dones)
                intrinsic_buf[t] = bonus
                rewards_buf[t] += effective_intrinsic_coef * bonus

            policy.observe(model, train_env_ids, actions_np.tolist(), step_rewards, step_dones)
            obs = np.stack(next_obs)
            global_step += config.num_envs

        with torch.no_grad():
            _, next_values_t, _ = peek_forward(model, torch.as_tensor(obs, dtype=torch.float32, device=device), train_env_ids)
            next_values = next_values_t.cpu().numpy()

        advantages = np.zeros_like(rewards_buf)
        last_gae = np.zeros(config.num_envs, dtype=np.float32)
        for t in reversed(range(config.rollout)):
            next_nonterminal = 1.0 - dones_buf[t]
            next_value = next_values if t == config.rollout - 1 else values_buf[t + 1]
            delta = rewards_buf[t] + config.gamma * next_value * next_nonterminal - values_buf[t]
            last_gae = delta + config.gamma * config.gae_lambda * next_nonterminal * last_gae
            advantages[t] = last_gae
        returns = advantages + values_buf

        b_obs = torch.as_tensor(obs_buf.reshape(-1, *obs.shape[1:]), dtype=torch.float32, device=device)
        flat_next_obs = next_obs_buf.reshape(-1, *obs.shape[1:])
        b_actions = torch.as_tensor(actions_buf.reshape(-1), dtype=torch.long, device=device)
        b_masks = torch.as_tensor(masks_buf.reshape(-1, envs[0].action_space_n), dtype=torch.bool, device=device)
        b_logp = torch.as_tensor(logp_buf.reshape(-1), dtype=torch.float32, device=device)
        b_adv = torch.as_tensor(advantages.reshape(-1), dtype=torch.float32, device=device)
        if context_buf is not None:
            context_flat = context_buf.reshape(-1, *context_buf.shape[2:])
            context_dtype = torch.long if np.issubdtype(context_flat.dtype, np.integer) else torch.float32
            b_context = torch.as_tensor(context_flat, dtype=context_dtype, device=device)
        else:
            b_context = None
        b_returns = torch.as_tensor(returns.reshape(-1), dtype=torch.float32, device=device)
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        sequence_hook = getattr(model, "training_sequence_forward", None)
        if model_spec.recurrent and callable(sequence_hook) and context_buf is not None:
            # Recurrent PPO uses rollout-length truncated BPTT. We batch complete
            # environment trajectories so hidden-state chronology is preserved; done
            # masks reset memory inside the model hook.
            obs_seq = torch.as_tensor(obs_buf, dtype=torch.float32, device=device)
            actions_seq = torch.as_tensor(actions_buf, dtype=torch.long, device=device)
            masks_seq = torch.as_tensor(masks_buf, dtype=torch.bool, device=device)
            logp_seq = torch.as_tensor(logp_buf, dtype=torch.float32, device=device)
            adv_seq = b_adv.view(config.rollout, config.num_envs)
            returns_seq = b_returns.view(config.rollout, config.num_envs)
            dones_seq = torch.as_tensor(dones_buf, dtype=torch.bool, device=device)
            first_context_np = context_buf[0]
            first_context = torch.as_tensor(first_context_np, dtype=torch.float32, device=device)
            env_indices = np.arange(config.num_envs)
            envs_per_minibatch = max(1, config.minibatch // max(1, config.rollout))
            for _ in range(config.epochs):
                np.random.shuffle(env_indices)
                for start in range(0, config.num_envs, envs_per_minibatch):
                    chosen_np = env_indices[start : start + envs_per_minibatch]
                    chosen = torch.as_tensor(chosen_np, dtype=torch.long, device=device)
                    logits_t, values_t = sequence_hook(
                        obs_seq[:, chosen], first_context[chosen], actions_seq[:, chosen], dones_seq[:, chosen]
                    )
                    flat_logits = logits_t.reshape(-1, envs[0].action_space_n)
                    flat_values = values_t.reshape(-1)
                    flat_masks = masks_seq[:, chosen].reshape(-1, envs[0].action_space_n)
                    flat_actions = actions_seq[:, chosen].reshape(-1)
                    flat_old_logp = logp_seq[:, chosen].reshape(-1)
                    flat_adv = adv_seq[:, chosen].reshape(-1)
                    flat_returns = returns_seq[:, chosen].reshape(-1)
                    dist = _distribution(flat_logits, flat_masks)
                    new_logp = dist.log_prob(flat_actions)
                    ratio = (new_logp - flat_old_logp).exp()
                    policy_loss = -torch.min(
                        ratio * flat_adv,
                        torch.clamp(ratio, 1 - config.clip, 1 + config.clip) * flat_adv,
                    ).mean()
                    value_loss = 0.5 * (flat_values - flat_returns).pow(2).mean()
                    loss = policy_loss + config.value_coef * value_loss - config.entropy * dist.entropy().mean()
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    optimizer.step()
                    last_loss = float(loss.detach().cpu().item())
        else:
            indices = np.arange(b_obs.shape[0])
            for _ in range(config.epochs):
                np.random.shuffle(indices)
                for start in range(0, len(indices), config.minibatch):
                    mb = torch.as_tensor(indices[start : start + config.minibatch], dtype=torch.long, device=device)
                    mb_context = b_context[mb] if b_context is not None else None
                    logits, new_value = optimization_forward(model, b_obs[mb], mb_context)
                    dist = _distribution(logits, b_masks[mb])
                    new_logp = dist.log_prob(b_actions[mb])
                    ratio = (new_logp - b_logp[mb]).exp()
                    policy_loss = -torch.min(
                        ratio * b_adv[mb],
                        torch.clamp(ratio, 1 - config.clip, 1 + config.clip) * b_adv[mb],
                    ).mean()
                    value_loss = 0.5 * (new_value - b_returns[mb]).pow(2).mean()
                    loss = policy_loss + config.value_coef * value_loss - config.entropy * dist.entropy().mean()
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    optimizer.step()
                    last_loss = float(loss.detach().cpu().item())

        if intrinsic is not None:
            intrinsic_loss = intrinsic.update(obs_buf.reshape(-1, *obs.shape[1:]), flat_next_obs, actions_buf.reshape(-1))

        if progress_callback:
            progress_callback(
                {
                    "kind": "rollout",
                    "steps": min(global_step, config.timesteps),
                    "timesteps": config.timesteps,
                    "progress": min(1.0, global_step / max(1, config.timesteps)),
                    "train_success100": float(np.mean(recent_success)) if recent_success else 0.0,
                    "loss": last_loss,
                    "intrinsic_mean": float(intrinsic_buf.mean()) if intrinsic is not None else 0.0,
                    "intrinsic_coef": float(effective_intrinsic_coef),
                    "intrinsic_loss": float(intrinsic_loss),
                    "active_maps": len(active_maps),
                    "device": device.type,
                }
            )

        if global_step >= next_eval or update == updates:
            # Unseen test maps are shown only as a monitoring signal. They never select
            # the exported checkpoint. Best-model selection uses a small deterministic
            # validation suite built from training layouts with fixed held-out seeds.
            success, mean_steps, efficiency = _quick_eval(
                model, policy, test_env, test_maps, device, run_mask_mode
            )
            val_success, val_steps, val_efficiency = _quick_eval(
                model, policy, validation_env, validation_maps, device, run_mask_mode
            )
            recent = float(np.mean(recent_success)) if recent_success else 0.0
            msg = (
                f"steps={global_step} train_success100={recent:.3f} "
                f"quick_test_success={success:.3f} quick_test_steps={mean_steps:.1f} "
                f"efficiency={efficiency:.3f} val_success={val_success:.3f} "
                f"val_steps={val_steps:.1f} val_efficiency={val_efficiency:.3f} "
                f"intrinsic={float(intrinsic_buf.mean()):.3f} coef={effective_intrinsic_coef:.4f} "
                f"maps={len(active_maps)} device={device.type}"
            )
            if log_callback:
                log_callback(msg)
            else:
                print(msg)

            rank = _validation_rank(val_success, val_efficiency, val_steps)
            if best_rank is None or rank > best_rank:
                best_rank = rank
                best_metrics = {
                    "validation_success": float(val_success),
                    "validation_mean_steps": float(val_steps),
                    "validation_path_efficiency": float(val_efficiency),
                    "quick_test_success_monitor_only": float(success),
                }
                torch.save(
                    _checkpoint_payload(
                        model=model,
                        model_module=model_module,
                        model_display_name=model_spec.display_name,
                        config=config,
                        observation_shape=tuple(obs.shape[1:]),
                        action_count=envs[0].action_space_n,
                        role="best",
                        global_step=global_step,
                        selection_metrics=best_metrics,
                    ),
                    best_checkpoint_path,
                )
                if log_callback:
                    log_callback(
                        f"New best validation checkpoint: {best_checkpoint_path} "
                        f"(val_success={val_success:.3f}, step={global_step})"
                    )

            if progress_callback:
                progress_callback(
                    {
                        "kind": "evaluation",
                        "steps": min(global_step, config.timesteps),
                        "timesteps": config.timesteps,
                        "progress": min(1.0, global_step / max(1, config.timesteps)),
                        "train_success100": recent,
                        "quick_test_success": success,
                        "quick_test_steps": mean_steps,
                        "efficiency": efficiency,
                        "validation_success": val_success,
                        "validation_steps": val_steps,
                        "validation_efficiency": val_efficiency,
                        "loss": last_loss,
                        "intrinsic_mean": float(intrinsic_buf.mean()) if intrinsic is not None else 0.0,
                        "intrinsic_coef": float(effective_intrinsic_coef),
                        "intrinsic_loss": float(intrinsic_loss),
                        "active_maps": len(active_maps),
                        "device": device.type,
                    }
                )
            next_eval += config.eval_every

    # The canonical model checkpoint is the exact final optimizer endpoint that
    # produced the last on-screen quick evaluation. Validation-best weights are kept
    # separately so formal evaluation can choose either role explicitly.
    torch.save(
        _checkpoint_payload(
            model=model,
            model_module=model_module,
            model_display_name=model_spec.display_name,
            config=config,
            observation_shape=tuple(obs.shape[1:]),
            action_count=envs[0].action_space_n,
            role="last",
            global_step=global_step,
            selection_metrics=best_metrics,
        ),
        checkpoint_path,
    )

    if not best_checkpoint_path.exists():
        torch.save(
            _checkpoint_payload(
                model=model,
                model_module=model_module,
                model_display_name=model_spec.display_name,
                config=config,
                observation_shape=tuple(obs.shape[1:]),
                action_count=envs[0].action_space_n,
                role="best",
                global_step=global_step,
                selection_metrics=best_metrics,
            ),
            best_checkpoint_path,
        )

    if log_callback:
        log_callback(f"Final checkpoint (default Evaluate): {checkpoint_path}")
        log_callback(f"Best validation checkpoint: {best_checkpoint_path}")
    return checkpoint_path
