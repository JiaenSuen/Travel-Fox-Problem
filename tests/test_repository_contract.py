from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_readme_has_demo_asset():
    text = (ROOT / "README.md").read_text(encoding="utf-8").lower()
    assert "assets/demo.png" in text
    assert (ROOT / "assets" / "demo.png").exists()


def test_compare_records_checkpoint_and_eval_masks():
    text = (ROOT / "tfp_studio.py").read_text(encoding="utf-8")
    assert '"pt_mask":"PT Mask"' in text
    assert '"mask":"Eval Mask"' in text
    assert 'metadata.get("checkpoint_action_mask_mode"' in text
    assert 'metadata.get("action_mask_mode"' in text


def test_reference_results_keep_protocol_metadata():
    paths = sorted((ROOT / "reference_results" / "initial_study").glob("*.json"))
    assert len(paths) == 5
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        meta = data["metadata"]
        assert meta.get("action_mask_mode") in {"task", "valid"}
        assert meta.get("checkpoint_action_mask_mode") in {"task", "valid"}
