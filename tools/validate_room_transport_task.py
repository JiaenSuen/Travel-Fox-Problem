from __future__ import annotations

import json
import sys
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tfp.tasks import create_task_env, get_task
from tfp.utils import discover_maps

TASK_ID = "TFP-RoomTransport"
MOVE_FOR_DELTA = {(-1, 0): 0, (1, 0): 1, (0, -1): 2, (0, 1): 3}


def shortest_path(env, start, goal):
    queue = deque([start])
    parent = {start: None}
    while queue:
        pos = queue.popleft()
        if pos == goal:
            break
        for dr, dc in env.ACTIONS.values():
            nxt = (pos[0] + dr, pos[1] + dc)
            if nxt not in parent and env._is_floor_or_door(nxt):
                parent[nxt] = pos
                queue.append(nxt)
    if goal not in parent:
        raise RuntimeError(f"No structural path from {start} to {goal}")
    out = []
    cur = goal
    while cur != start:
        out.append(cur)
        cur = parent[cur]
    out.reverse()
    return out


def solve_episode(env):
    while env.steps < env.max_steps:
        target = env.goal_pos if env.carrying else env.object_pos
        if target is None:
            raise RuntimeError("missing target")
        if env.agent_pos == target:
            action = 5 if env.carrying else 4
        else:
            path = shortest_path(env, env.agent_pos, target)
            nxt = path[0]
            if nxt in env.door_positions and nxt not in env.open_doors:
                action = 6
            else:
                delta = (nxt[0] - env.agent_pos[0], nxt[1] - env.agent_pos[1])
                action = MOVE_FOR_DELTA[delta]
        _, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            return bool(info["success"]), info
    return False, env._info(False)


def main():
    task = get_task(TASK_ID)
    maps = discover_maps(task.test_map_dir)
    env = create_task_env(TASK_ID, maps, view_size=7, reward_module=task.default_reward)
    records = []
    for map_path in maps:
        for seed in task.default_eval_seeds:
            env.reset(seed=seed, map_path=map_path)
            success, info = solve_episode(env)
            records.append({
                "map": map_path.name,
                "seed": seed,
                "success": success,
                "steps": info["steps"],
                "rooms": info["rooms"],
                "doors": info["doors_total"],
                "tier": info["size_tier"],
            })
    summary = {
        "task_id": TASK_ID,
        "episodes": len(records),
        "successes": sum(int(r["success"]) for r in records),
        "tiers": sorted({r["tier"] for r in records}),
        "room_counts": sorted({r["rooms"] for r in records}),
    }
    print(json.dumps(summary, indent=2))
    if summary["successes"] != summary["episodes"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
