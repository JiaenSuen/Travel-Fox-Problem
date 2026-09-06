from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from tfp.envs.transport_env import TransportEnv
from tfp.evaluation.protocol import evaluate_policy, load_checkpoint_model
from tfp.tasks import get_task
from tfp.training.ppo import PPOConfig, train_ppo
from tfp.utils import discover_maps

MODELS = (
    "001_simple_cnn",
    "006_ppo_gru_bootstrap",
    "007_ppo_gru_action_memory",
    "008_ppo_gru_episodic_count",
    "009_ppo_gru_action_memory_tabux",
    "010_ppo_gru_gobi",
)


def balanced_subset(paths, per_size: int):
    out = []
    for size in ("10x10", "15x15", "20x20"):
        out.extend([p for p in paths if size in p.name][:per_size])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the TFP exploration and anti-deadlock suite.")
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--reward", default="002_cycle_safe_transport")
    parser.add_argument(
        "--policy",
        default="001_ppo_categorical",
        choices=("001_ppo_categorical", "002_ppo_cycle_guard"),
    )
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--models", nargs="*", default=list(MODELS))
    args = parser.parse_args()

    task = get_task("TFP-FoxTransport-Local")
    all_train = discover_maps(task.train_map_dir)
    all_test = discover_maps(task.test_map_dir)
    if args.mode == "smoke":
        train_maps = balanced_subset(all_train, 2)
        test_maps = balanced_subset(all_test, 1)
        timesteps, num_envs, rollout, epochs, minibatch = 2048, 4, 32, 2, 128
        eval_seeds = (11, 29, 47)
        output_dir = Path("results/smoke")
    else:
        train_maps = all_train
        test_maps = all_test
        timesteps, num_envs, rollout, epochs, minibatch = 120_000, 16, 64, 4, 256
        eval_seeds = tuple(task.default_eval_seeds)
        output_dir = Path("results/exploration_full")

    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for model_name in args.models:
        print(f"\n===== {model_name} / {args.policy} / {args.reward} =====", flush=True)
        ckpt = output_dir / "checkpoints" / f"{model_name}.pt"
        cfg = PPOConfig(
            timesteps=timesteps,
            num_envs=num_envs,
            rollout=rollout,
            epochs=epochs,
            minibatch=minibatch,
            eval_every=timesteps + 1,
            seed=args.seed,
            curriculum=False if args.mode == "smoke" else True,
            reward_module=args.reward,
            policy_module=args.policy,
            device=args.device,
            experiment_tag=args.mode,
        )
        train_ppo(model_name, train_maps, test_maps, ckpt, cfg, log_callback=print)
        device = torch.device(
            "cuda" if args.device == "auto" and torch.cuda.is_available()
            else ("cpu" if args.device == "auto" else args.device)
        )
        model, payload = load_checkpoint_model(ckpt, device)
        env = TransportEnv(
            test_maps,
            observation_mode="local",
            view_size=5,
            seed=args.seed + 9000,
            reward_module=args.reward,
        )
        summary, _, paths = evaluate_policy(
            model,
            env,
            test_maps,
            device,
            seeds=eval_seeds,
            action_mask_mode=str(payload.get("action_mask_mode", "task")),
            policy_module=str(payload.get("policy_module", args.policy)),
            model_name=model_name,
            presentation="data",
            output_dir=output_dir / "episodes",
            run_metadata={
                "suite_mode": args.mode,
                "training_seed": args.seed,
                "training_timesteps": timesteps,
                "protocol_source": "checkpoint",
                "checkpoint_reward_module": str(payload.get("reward_module", args.reward)),
                "checkpoint_policy_module": str(payload.get("policy_module", args.policy)),
                "checkpoint_action_mask_mode": str(payload.get("action_mask_mode", "task")),
            },
        )
        summaries[model_name] = asdict(summary)
        print(json.dumps(summaries[model_name], indent=2), flush=True)
        print("saved:", *(str(p) for p in paths), sep="\n  ", flush=True)

    summary_path = output_dir / "suite_summary.json"
    summary_path.write_text(
        json.dumps({
            "mode": args.mode,
            "seed": args.seed,
            "reward": args.reward,
            "policy": args.policy,
            "timesteps": timesteps,
            "train_maps": [p.name for p in train_maps],
            "test_maps": [p.name for p in test_maps],
            "eval_seeds": list(eval_seeds),
            "results": summaries,
        }, indent=2),
        encoding="utf-8",
    )
    print(f"\nSuite summary: {summary_path}")


if __name__ == "__main__":
    main()
