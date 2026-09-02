from __future__ import annotations

import csv
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import torch
from torch import nn

from tfp import __version__ as TFP_VERSION
from tfp.envs.transport_env import TransportEnv
from tfp.evaluation.diagnostics import BehaviorDiagnostics
from tfp.models.model_api import load_model_plugin, rollout_forward
from tfp.policies import load_policy_plugin
from tfp.tasks.fox_transport_local import DEFAULT_EVAL_SEEDS
from tfp.visualization.renderer import FoxRenderer, LiveWindow, VideoRecorder

ACTION_NAMES = ("UP", "DOWN", "LEFT", "RIGHT", "PICKUP", "DROP")


@dataclass
class EpisodeRecord:
    task_id: str
    model: str
    policy: str
    reward: str
    map_name: str
    map_family: str
    map_size: str
    seed: int
    success: int
    steps: int
    total_return: float
    oracle_steps: int
    path_efficiency: float
    collisions: int
    invalid_actions: int
    pickup_step: int | None
    cycle_events: int
    interaction_cycle_events: int
    state_revisit_rate: float
    mean_inference_ms: float


@dataclass
class EvaluationSummary:
    episodes: int
    success_rate: float
    pickup_rate: float
    mean_steps: float
    mean_pickup_step: float
    mean_return: float
    mean_path_efficiency: float
    mean_collisions: float
    mean_invalid_actions: float
    mean_cycle_events: float
    mean_interaction_cycle_events: float
    mean_state_revisit_rate: float
    mean_inference_ms: float


def _summarize(records: Sequence[EpisodeRecord]) -> EvaluationSummary:
    success_records = [r for r in records if r.success]
    pickup_records = [r for r in records if r.pickup_step is not None]
    return EvaluationSummary(
        episodes=len(records),
        success_rate=float(np.mean([r.success for r in records])) if records else 0.0,
        pickup_rate=float(len(pickup_records) / len(records)) if records else 0.0,
        mean_steps=float(np.mean([r.steps for r in records])) if records else 0.0,
        mean_pickup_step=float(np.mean([r.pickup_step for r in pickup_records])) if pickup_records else 0.0,
        mean_return=float(np.mean([r.total_return for r in records])) if records else 0.0,
        mean_path_efficiency=float(np.mean([r.path_efficiency for r in success_records])) if success_records else 0.0,
        mean_collisions=float(np.mean([r.collisions for r in records])) if records else 0.0,
        mean_invalid_actions=float(np.mean([r.invalid_actions for r in records])) if records else 0.0,
        mean_cycle_events=float(np.mean([r.cycle_events for r in records])) if records else 0.0,
        mean_interaction_cycle_events=float(np.mean([r.interaction_cycle_events for r in records])) if records else 0.0,
        mean_state_revisit_rate=float(np.mean([r.state_revisit_rate for r in records])) if records else 0.0,
        mean_inference_ms=float(np.mean([r.mean_inference_ms for r in records])) if records else 0.0,
    )


def _save_results(
    records: Sequence[EpisodeRecord],
    summary: EvaluationSummary,
    output_dir: Path,
    prefix: str,
    metadata: dict[str, object] | None = None,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"{prefix}_{stamp}.csv"
    json_path = output_dir / f"{prefix}_{stamp}.json"

    if records:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(records[0]).keys()))
            writer.writeheader()
            writer.writerows(asdict(r) for r in records)
    else:
        csv_path.write_text("", encoding="utf-8")

    by_map: dict[str, dict[str, float]] = {}
    for map_name in sorted({r.map_name for r in records}):
        subset = [r for r in records if r.map_name == map_name]
        by_map[map_name] = asdict(_summarize(subset))
    by_size: dict[str, dict[str, float]] = {}
    for map_size in sorted({r.map_size for r in records}):
        subset = [r for r in records if r.map_size == map_size]
        by_size[map_size] = asdict(_summarize(subset))
    by_family: dict[str, dict[str, float]] = {}
    for family in sorted({r.map_family for r in records}):
        subset = [r for r in records if r.map_family == family]
        by_family[family] = asdict(_summarize(subset))

    payload = {
        "metadata": metadata or {},
        "summary": asdict(summary),
        "benchmark": {
            "maps": len({r.map_name for r in records}),
            "seeds": len({r.seed for r in records}),
            "episodes": len(records),
        },
        "by_map": by_map,
        "by_size": by_size,
        "by_family": by_family,
        "records": [asdict(r) for r in records],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return csv_path, json_path


def evaluate_policy(
    model: nn.Module,
    env: TransportEnv,
    maps: Sequence[Path],
    device: torch.device,
    seeds: Iterable[int] = DEFAULT_EVAL_SEEDS,
    action_mask_mode: str = "task",
    policy_module: str = "ppo_categorical_001",
    model_name: str = "unknown",
    presentation: str = "data",
    output_dir: str | Path = "results",
    video_dir: str | Path = "videos",
    video_fps: int = 8,
    progress_callback: Callable[[str], None] | None = None,
    run_metadata: dict[str, object] | None = None,
) -> tuple[EvaluationSummary, list[EpisodeRecord], tuple[Path, Path]]:
    """Evaluate the complete fixed map x seed matrix."""
    if presentation not in {"data", "display", "video"}:
        raise ValueError("presentation must be 'data', 'display', or 'video'.")

    _, policy_factory = load_policy_plugin(policy_module)
    policy = policy_factory()
    seeds = tuple(int(s) for s in seeds)
    renderer = FoxRenderer() if presentation != "data" else None
    window = LiveWindow() if presentation != "data" else None
    recorder = VideoRecorder(video_dir, fps=video_fps) if presentation == "video" else None
    records: list[EpisodeRecord] = []
    eval_env_id = (2_000_000,)

    model.eval()
    try:
        with torch.no_grad():
            for map_path in maps:
                for seed in seeds:
                    policy.reset(model, eval_env_id)
                    obs, _ = env.reset(seed=seed, map_path=map_path)
                    total_return = 0.0
                    diagnostics = BehaviorDiagnostics()
                    inference_ms: list[float] = []
                    visualize_episode = presentation != "data" and seed == seeds[0]
                    if recorder is not None and visualize_episode:
                        recorder.start_episode(map_path.stem, seed)

                    for _ in range(env.max_steps):
                        x = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                        mask = torch.as_tensor(env.valid_action_mask(action_mask_mode), dtype=torch.bool, device=device).unsqueeze(0)
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                        inference_start = time.perf_counter()
                        logits, _, _ = rollout_forward(model, x, eval_env_id)
                        action_t = policy.greedy(model, logits, mask, eval_env_id)
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                        inference_ms.append((time.perf_counter() - inference_start) * 1000.0)
                        action = int(action_t.item())
                        obs, reward, terminated, truncated, info = env.step(action)
                        total_return += reward
                        diagnostics.observe(
                            action,
                            (tuple(info.get("agent_pos", ())), bool(info.get("carrying", False)), info.get("object_pos")),
                        )
                        done = bool(terminated or truncated)
                        policy.observe(model, eval_env_id, (action,), (reward,), (done,))

                        if renderer is not None and visualize_episode:
                            status_open = bool(window.status_open) if window is not None else False
                            frame = renderer.render(
                                env,
                                action_name=ACTION_NAMES[action],
                                reward=reward,
                                status_open=status_open,
                            )
                            if window is not None and not window.show(frame):
                                presentation = "data"
                                window.close()
                                window = None
                            if recorder is not None:
                                recorder.append(frame)

                        if done:
                            record = EpisodeRecord(
                                task_id=str(info.get("task_id", env.TASK_ID)),
                                model=model_name,
                                policy=policy_module,
                                reward=env.reward_module,
                                map_name=str(info["map"]),
                                map_family=str(info["map"]).split("_")[1] if "_" in str(info["map"]) else "unknown",
                                map_size=f"{info['map_shape'][0]}x{info['map_shape'][1]}",
                                seed=seed,
                                success=int(info["success"]),
                                steps=int(info["steps"]),
                                total_return=float(total_return),
                                oracle_steps=int(info["oracle_steps"]),
                                path_efficiency=float(info["path_efficiency"]),
                                collisions=int(info["collisions"]),
                                invalid_actions=int(info["invalid_actions"]),
                                pickup_step=info["pickup_step"],
                                cycle_events=int(diagnostics.cycle_events),
                                interaction_cycle_events=int(diagnostics.interaction_cycle_events),
                                state_revisit_rate=float(diagnostics.state_revisit_rate),
                                mean_inference_ms=float(np.mean(inference_ms)) if inference_ms else 0.0,
                            )
                            records.append(record)
                            if progress_callback:
                                progress_callback(
                                    f"{record.map_name} seed={seed}: success={record.success} "
                                    f"steps={record.steps} efficiency={record.path_efficiency:.3f} "
                                    f"cycles={record.cycle_events} revisit={record.state_revisit_rate:.3f}"
                                )
                            break
                    if recorder is not None and visualize_episode:
                        recorder.finish_episode()
    finally:
        if window is not None:
            window.close()
        if recorder is not None:
            recorder.close()

    summary = _summarize(records)
    safe_task = env.TASK_CODE.lower().replace("-", "_")
    safe_model = model_name.lower().replace("-", "_")
    safe_policy = policy_module.lower().replace("-", "_")
    safe_reward = env.reward_module.lower().replace("-", "_")
    metadata: dict[str, object] = {
        "tfp_version": TFP_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "observation_mode": env.observation_mode,
        "view_size": env.view_size,
        "action_mask_mode": action_mask_mode,
    }
    if run_metadata:
        metadata.update(run_metadata)
    paths = _save_results(
        records,
        summary,
        Path(output_dir),
        prefix=f"{safe_task}_{safe_model}_{safe_policy}_{safe_reward}",
        metadata=metadata,
    )
    return summary, records, paths


def load_checkpoint_model(checkpoint_path: str | Path, device: torch.device) -> tuple[nn.Module, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    module_name = checkpoint.get("model_module", "simple_cnn_001")
    _, factory = load_model_plugin(module_name)
    model = factory(tuple(checkpoint["observation_shape"]), int(checkpoint["action_count"])).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model, checkpoint
