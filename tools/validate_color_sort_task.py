from __future__ import annotations

import json
from collections import deque
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tfp.tasks import create_task_env, get_task
from tfp.utils import discover_maps

TASK_ID = "TFP-FoxColorSort-Local"


def shortest_move(env, target):
    if env.agent_pos == target:
        return None
    q = deque([env.agent_pos])
    parent = {env.agent_pos: (None, None)}
    while q:
        pos = q.popleft()
        for action, (dr, dc) in env.ACTIONS.items():
            nxt = (pos[0] + dr, pos[1] + dc)
            if env._is_free(nxt) and nxt not in parent:
                parent[nxt] = (pos, action)
                if nxt == target:
                    cur = nxt
                    while parent[cur][0] != env.agent_pos:
                        cur = parent[cur][0]
                    return parent[cur][1]
                q.append(nxt)
    raise RuntimeError(f"No path from {env.agent_pos} to {target}")


def expert_action(env):
    if env.carrying_color is None:
        color = env._object_color_at(env.agent_pos)
        if color is not None:
            return 4
    elif env.goals[env.carrying_color] == env.agent_pos:
        return 5
    _, target = env._current_target()
    if target is None:
        return 0
    action = shortest_move(env, target)
    if action is None:
        return 4 if env.carrying_color is None else 5
    return int(action)


def main() -> None:
    task = get_task(TASK_ID)
    maps = discover_maps(task.test_map_dir)
    env = create_task_env(TASK_ID, maps, view_size=5, reward_module=task.default_reward, seed=0)
    records = []
    for map_path in maps:
        for seed in task.default_eval_seeds:
            env.reset(seed=seed, map_path=map_path)
            total_return = 0.0
            for _ in range(env.max_steps):
                action = expert_action(env)
                _, reward, terminated, truncated, info = env.step(action)
                total_return += reward
                if terminated or truncated:
                    records.append({
                        "map": map_path.name,
                        "seed": seed,
                        "success": int(info["success"]),
                        "steps": int(info["steps"]),
                        "oracle_steps": int(info["oracle_steps"]),
                        "path_efficiency": float(info["path_efficiency"]),
                        "items_total": int(info["items_total"]),
                        "total_return": float(total_return),
                    })
                    break
    summary = {
        "task_id": TASK_ID,
        "episodes": len(records),
        "success_rate": sum(r["success"] for r in records) / max(1, len(records)),
        "mean_efficiency": sum(r["path_efficiency"] for r in records) / max(1, len(records)),
        "min_items": min(r["items_total"] for r in records),
        "max_items": max(r["items_total"] for r in records),
    }
    out_dir = ROOT / "reference_results" / task.code
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "oracle_solvability.json"
    out.write_text(json.dumps({"summary": summary, "records": records}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
