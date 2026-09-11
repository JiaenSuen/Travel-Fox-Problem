from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont

from tfp.models import discover_model_plugins, load_model_plugin
from tfp.tasks import create_task_env, discover_tasks, get_task
from tfp.utils import discover_maps


BENCHMARK_POLICY = "001_ppo_categorical"
BENCHMARK_MASK = "task"


@dataclass(frozen=True)
class EvaluationRun:
    path: Path
    payload: dict[str, Any]
    task_id: str
    task_code: str
    model: str
    source: str
    controlled_benchmark: bool


def _font(size: int, bold: bool = False):
    names = ["DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", "Arial.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _read_payload(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("summary"), dict):
        return None
    return payload


def _first_record(payload: dict[str, Any]) -> dict[str, Any]:
    records = payload.get("records", [])
    return records[0] if isinstance(records, list) and records and isinstance(records[0], dict) else {}


def _task_identity(payload: dict[str, Any]) -> tuple[str, str]:
    metadata = payload.get("metadata", {}) if isinstance(payload.get("metadata"), dict) else {}
    first = _first_record(payload)
    task_id = str(metadata.get("task_id") or first.get("task_id") or "")
    task_code = str(metadata.get("task_code") or first.get("task_code") or "")
    if task_id and not task_code:
        try:
            task_code = get_task(task_id).code
        except KeyError:
            pass
    if task_code and not task_id:
        for known_id, spec in discover_tasks().items():
            if spec.code == task_code:
                task_id = known_id
                break
    return task_id, task_code


def _model_identity(payload: dict[str, Any]) -> str:
    first = _first_record(payload)
    return str(first.get("model") or "")


def _source_label(path: Path, payload: dict[str, Any], repository_root: Path) -> str:
    metadata = payload.get("metadata", {}) if isinstance(payload.get("metadata"), dict) else {}
    note = " ".join(
        str(metadata.get(k, "")) for k in ("note", "reference_note", "experiment_tag")
    ).lower()
    if "smoke" in path.stem.lower() or "smoke" in note:
        return "smoke"
    try:
        path.relative_to(repository_root / "reference_results")
        return "reference"
    except ValueError:
        return "local"


def _controlled_protocol(payload: dict[str, Any], task_id: str) -> bool:
    """Return True only for the stable cross-model benchmark protocol.

    README benchmark tables intentionally exclude ad-hoc, smoke and ablation runs. A run
    qualifies only when it evaluates the entire packaged test map set using exactly the
    task's default seed matrix, default local view, default reward, task action mask and
    the standard PPO categorical action policy.
    """
    if not task_id:
        return False
    try:
        task = get_task(task_id)
    except KeyError:
        return False

    records = payload.get("records", [])
    if not isinstance(records, list) or not records:
        return False
    metadata = payload.get("metadata", {}) if isinstance(payload.get("metadata"), dict) else {}
    first = _first_record(payload)

    expected_maps = {p.name for p in discover_maps(task.test_map_dir)}
    observed_maps = {str(r.get("map_name")) for r in records if isinstance(r, dict) and r.get("map_name")}
    observed_seeds = {int(r.get("seed")) for r in records if isinstance(r, dict) and r.get("seed") is not None}
    if observed_maps != expected_maps or observed_seeds != set(task.default_eval_seeds):
        return False

    if str(metadata.get("observation_mode", task.default_observation_mode)) != task.default_observation_mode:
        return False
    if int(metadata.get("view_size", task.default_view_size) or task.default_view_size) != task.default_view_size:
        return False
    if str(metadata.get("action_mask_mode", BENCHMARK_MASK)) != BENCHMARK_MASK:
        return False
    if str(first.get("policy", BENCHMARK_POLICY)) != BENCHMARK_POLICY:
        return False
    if str(first.get("reward", task.default_reward)) != task.default_reward:
        return False
    return True


def collect_evaluation_runs(repository_root: str | Path) -> list[EvaluationRun]:
    """Collect valid evaluation JSON records from local results and packaged references."""
    root = Path(repository_root)
    candidates: set[Path] = set()
    for base in (root / "results", root / "reference_results"):
        if base.exists():
            candidates.update(base.rglob("*.json"))

    runs: list[EvaluationRun] = []
    for path in sorted(candidates):
        payload = _read_payload(path)
        if payload is None:
            continue
        task_id, task_code = _task_identity(payload)
        model = _model_identity(payload)
        # Oracle diagnostics and other non-policy JSON files deliberately do not enter
        # comparison/reporting tables.
        if not task_id or not task_code or not model:
            continue
        runs.append(
            EvaluationRun(
                path=path,
                payload=payload,
                task_id=task_id,
                task_code=task_code,
                model=model,
                source=_source_label(path, payload, root),
                controlled_benchmark=_controlled_protocol(payload, task_id),
            )
        )
    return runs


def _mean(values: Iterable[Any]) -> float | None:
    parsed: list[float] = []
    for value in values:
        try:
            parsed.append(float(value))
        except (TypeError, ValueError):
            pass
    return sum(parsed) / len(parsed) if parsed else None


def _estimate_model_parameters(task_id: str, model_name: str) -> int | None:
    try:
        task = get_task(task_id)
        maps = discover_maps(task.test_map_dir)
        if not maps:
            return None
        env = create_task_env(
            task_id,
            maps[:1],
            observation_mode=task.default_observation_mode,
            view_size=task.default_view_size,
            reward_module=task.default_reward,
        )
        obs, _ = env.reset(seed=task.default_eval_seeds[0], map_path=maps[0])
        _, factory = load_model_plugin(model_name, task_id=task_id)
        model = factory(tuple(obs.shape), int(env.action_space_n))
        return int(sum(p.numel() for p in model.parameters()))
    except Exception:
        return None


def benchmark_rows(repository_root: str | Path, task_id: str) -> list[dict[str, Any]]:
    """Return one controlled-benchmark row for every model registered to a task.

    Repeated full-protocol evaluations are aggregated across runs instead of replacing the
    table with the most recently executed test. Models without a qualifying benchmark are
    retained with metric values set to ``None``.
    """
    root = Path(repository_root)
    task = get_task(task_id)
    plugins = discover_model_plugins(task_id)
    all_runs = collect_evaluation_runs(root)
    controlled = [r for r in all_runs if r.task_id == task_id and r.controlled_benchmark]

    rows: list[dict[str, Any]] = []
    for model_name, spec in plugins.items():
        model_runs = [r for r in controlled if r.model == model_name]
        summaries = [r.payload.get("summary", {}) for r in model_runs]
        metas = [r.payload.get("metadata", {}) for r in model_runs]
        episode_count = sum(int(s.get("episodes", 0) or 0) for s in summaries)
        params = _mean(m.get("model_parameters") for m in metas)
        if params is None:
            params = _estimate_model_parameters(task_id, model_name)
        rows.append(
            {
                "task_id": task_id,
                "task_code": task.code,
                "model": model_name,
                "display_name": spec.display_name,
                "memory_type": spec.memory_type,
                "run_count": len(model_runs),
                "episodes": episode_count,
                "params": int(round(params)) if params is not None else None,
                "success": _mean(s.get("success_rate") for s in summaries),
                "pickup": _mean(s.get("pickup_rate") for s in summaries),
                "completion": _mean(s.get("mean_completion_rate", s.get("success_rate")) for s in summaries),
                "steps": _mean(s.get("mean_steps") for s in summaries),
                "return": _mean(s.get("mean_return") for s in summaries),
                "efficiency": _mean(s.get("mean_path_efficiency") for s in summaries),
                "collisions": _mean(s.get("mean_collisions") for s in summaries),
                "invalid": _mean(s.get("mean_invalid_actions") for s in summaries),
                "cycles": _mean(s.get("mean_cycle_events") for s in summaries),
                "interaction_cycles": _mean(s.get("mean_interaction_cycle_events") for s in summaries),
                "revisit": _mean(s.get("mean_state_revisit_rate") for s in summaries),
                "inference_ms": _mean(s.get("mean_inference_ms") for s in summaries),
                "status": "benchmark" if model_runs else "pending",
            }
        )
    return rows


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def _pct(value: Any) -> str:
    if value is None:
        return "—"
    try:
        return f"{100.0 * float(value):.1f}%"
    except (TypeError, ValueError):
        return "—"


def _params(value: Any) -> str:
    if value is None:
        return "—"
    value = int(value)
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}k"
    return str(value)


def benchmark_markdown(repository_root: str | Path, task_id: str) -> str:
    task = get_task(task_id)
    rows = benchmark_rows(repository_root, task_id)
    header = [
        "| Model | Params | Runs | N | Success ↑ | Completion ↑ | Pickup ↑ | Steps ↓ | Return ↑ | Path Eff. ↑ | Collision ↓ | Invalid ↓ | Cycles ↓ | Interact cycles ↓ | Revisit ↓ | Infer ms ↓ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    body: list[str] = []
    for row in rows:
        body.append(
            f"| `{row['model']}` | {_params(row['params'])} | {row['run_count'] or '—'} | {row['episodes'] or '—'} | "
            f"{_pct(row['success'])} | {_pct(row['completion'])} | {_pct(row['pickup'])} | {_fmt(row['steps'], 1)} | "
            f"{_fmt(row['return'], 2)} | {_fmt(row['efficiency'])} | {_fmt(row['collisions'], 2)} | {_fmt(row['invalid'], 2)} | "
            f"{_fmt(row['cycles'], 2)} | {_fmt(row['interaction_cycles'], 2)} | {_pct(row['revisit'])} | {_fmt(row['inference_ms'])} |"
        )
    expected_episodes = len(discover_maps(task.test_map_dir)) * len(task.default_eval_seeds)
    note = (
        f"Controlled protocol: {len(discover_maps(task.test_map_dir))} test maps × {len(task.default_eval_seeds)} seeds "
        f"= {expected_episodes} episodes/run, `{task.default_observation_mode}` {task.default_view_size}×{task.default_view_size}, "
        f"`{task.default_reward}`, `{BENCHMARK_MASK}` mask, `{BENCHMARK_POLICY}`. "
        "Only complete protocol-matched runs enter this table; repeated runs are averaged. `—` means no qualifying benchmark yet."
    )
    return "\n".join([*header, *body, "", f"*{note}*"])


def _replace_marked_section(text: str, marker: str, content: str) -> str:
    start = f"<!-- {marker}:START -->"
    end = f"<!-- {marker}:END -->"
    if start not in text or end not in text:
        return text
    before, tail = text.split(start, 1)
    _, after = tail.split(end, 1)
    return f"{before}{start}\n{content.rstrip()}\n{end}{after}"


def update_root_benchmark(repository_root: str | Path, task_id: str) -> None:
    root = Path(repository_root)
    readme = root / "README.md"
    if not readme.exists():
        return
    task = get_task(task_id)
    text = readme.read_text(encoding="utf-8")
    marker = f"TFP-BENCHMARK:{task.code}"
    updated = _replace_marked_section(text, marker, benchmark_markdown(root, task_id))
    if updated != text:
        readme.write_text(updated, encoding="utf-8")


def _find_task_id(task_code: str) -> str | None:
    for task_id, spec in discover_tasks().items():
        if spec.code == task_code:
            return task_id
    return None


def _draw_benchmark_image(path: Path, task_id: str, rows: list[dict[str, Any]]) -> None:
    task = get_task(task_id)
    row_h = 50
    height = max(430, 250 + row_h * len(rows))
    width = 1420
    image = Image.new("RGB", (width, height), (248, 249, 252))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((30, 24, width - 30, height - 24), radius=20, fill=(255, 255, 255), outline=(223, 226, 234), width=2)
    draw.text((65, 50), f"{task.code} · Controlled Benchmark", fill=(35, 40, 49), font=_font(29, True))
    draw.text((65, 91), task.display_name, fill=(90, 97, 109), font=_font(17))
    draw.text((65, 120), "Success rate by registered model · complete protocol runs only", fill=(116, 122, 133), font=_font(15))

    y0 = 170
    label_x = 65
    bar_x = 485
    bar_w = 680
    for index, row in enumerate(rows):
        y = y0 + index * row_h
        display = str(row["display_name"]).replace(" · ", " ")
        draw.text((label_x, y + 4), display, fill=(61, 67, 77), font=_font(15, True))
        draw.rounded_rectangle((bar_x, y + 5, bar_x + bar_w, y + 27), radius=10, fill=(232, 235, 241))
        success = row.get("success")
        if success is not None:
            fill = int(bar_w * max(0.0, min(1.0, float(success))))
            if fill > 0:
                draw.rounded_rectangle((bar_x, y + 5, bar_x + fill, y + 27), radius=10, fill=(112, 143, 190))
            draw.text((bar_x + bar_w + 24, y + 3), f"{100 * float(success):.1f}%", fill=(44, 50, 60), font=_font(16, True))
            draw.text((bar_x + bar_w + 105, y + 4), f"n={row['episodes']}", fill=(110, 116, 127), font=_font(14))
        else:
            draw.text((bar_x + 14, y + 4), "benchmark pending", fill=(137, 143, 153), font=_font(14))

    footer_y = height - 58
    maps = len(discover_maps(task.test_map_dir))
    seeds = len(task.default_eval_seeds)
    draw.text(
        (65, footer_y),
        f"Protocol: {maps} maps × {seeds} seeds · {task.default_view_size}×{task.default_view_size} local · {task.default_reward} · task mask",
        fill=(112, 118, 129),
        font=_font(14),
    )
    image.save(path)


def build_task_report(
    task_dir: str | Path,
    task_code: str,
    task_name: str,
    repository_root: str | Path | None = None,
) -> tuple[Path, Path]:
    """Regenerate stable per-task README + PNG from the controlled benchmark matrix.

    Unlike the previous latest-run report, this table enumerates every model registered to
    the task. Only full, protocol-matched evaluations contribute metrics; repeated formal
    runs are aggregated. Ad-hoc/smoke/ablation records stay available to TFP Studio Compare
    without silently changing the README benchmark.
    """
    task_dir = Path(task_dir)
    task_dir.mkdir(parents=True, exist_ok=True)
    readme_path = task_dir / "README.md"
    png_path = task_dir / "summary.png"

    task_id = _find_task_id(task_code)
    if repository_root is None:
        candidate = task_dir.parent.parent if task_dir.parent.name == "results" else None
        repository_root = candidate if candidate and (candidate / "tfp").exists() else None

    if task_id is None or repository_root is None:
        # Generic fallback retained for external/plugin tests.
        readme_path.write_text(
            f"# {task_code} — Results\n\nNo registered benchmark context is available yet.\n",
            encoding="utf-8",
        )
        image = Image.new("RGB", (900, 320), (248, 249, 252))
        draw = ImageDraw.Draw(image)
        draw.text((48, 48), f"{task_code} · {task_name}", fill=(35, 40, 49), font=_font(28, True))
        draw.text((48, 110), "No registered benchmark context is available yet.", fill=(100, 106, 118), font=_font(17))
        image.save(png_path)
        return readme_path, png_path

    root = Path(repository_root)
    task = get_task(task_id)
    rows = benchmark_rows(root, task_id)
    markdown = benchmark_markdown(root, task_id)
    lines = [
        f"# {task.code} — {task.display_name}",
        "",
        "Controlled cross-model benchmark generated from complete evaluation records.",
        "",
        markdown,
        "",
        "![Controlled benchmark summary](summary.png)",
        "",
        "Ad-hoc, smoke, and ablation evaluations remain visible in **TFP Studio → Compare** but do not alter this table.",
        "",
    ]
    readme_path.write_text("\n".join(lines), encoding="utf-8")
    _draw_benchmark_image(png_path, task_id, rows)
    update_root_benchmark(root, task_id)
    return readme_path, png_path


def rebuild_all_reports(repository_root: str | Path) -> list[tuple[Path, Path]]:
    root = Path(repository_root)
    outputs: list[tuple[Path, Path]] = []
    for task_id, task in discover_tasks().items():
        outputs.append(
            build_task_report(root / "results" / task.code, task.code, task.display_name, repository_root=root)
        )
    return outputs
