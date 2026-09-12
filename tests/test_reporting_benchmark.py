from pathlib import Path

from tfp.reporting import benchmark_rows, collect_evaluation_runs

ROOT = Path(__file__).resolve().parents[1]


def test_transport_readme_benchmark_lists_every_registered_model():
    rows = benchmark_rows(ROOT, "TFP-LocalTransport")
    assert len(rows) == 10
    by_model = {row["model"]: row for row in rows}
    assert by_model["001_simple_cnn"]["episodes"] == 180
    assert by_model["005_action_memory_tabux_cnn"]["episodes"] == 180
    assert by_model["006_ppo_gru_bootstrap"]["episodes"] == 0


def test_color_sort_smoke_record_is_not_promoted_to_controlled_benchmark():
    rows = benchmark_rows(ROOT, "TFP-ColorSort")
    assert len(rows) == 3
    assert all(row["run_count"] == 0 for row in rows)
    smoke = [r for r in collect_evaluation_runs(ROOT) if r.task_code == "COLOR-SORT"]
    assert smoke
    assert not any(r.controlled_benchmark for r in smoke)


def test_gitignore_blocks_weights_videos_and_common_secret_files():
    text = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for pattern in ("*.pt", "*.pth", "*.ckpt", "*.mp4", ".env", "*.pem", "credentials*.json"):
        assert pattern in text
    assert "!checkpoints/LOCAL-TRANSPORT/" not in text


def test_room_transport_benchmark_has_all_registered_models():
    rows = benchmark_rows(ROOT, "TFP-RoomTransport")
    assert [row["model"] for row in rows] == [
        "001_route_prior_cnn",
        "002_route_prior_action_memory",
        "003_route_prior_gru_memory",
    ]
    assert all(row["status"] == "pending" for row in rows)
