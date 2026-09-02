from __future__ import annotations

import json
import unittest
from pathlib import Path

import torch

from tfp import __version__

ROOT = Path(__file__).resolve().parents[1]


class TFPReleaseContractTests(unittest.TestCase):
    def test_release_version(self):
        self.assertEqual(__version__, "1.0.0")

    def _assert_same_neural_weights(self, source_name: str, wrapper_name: str):
        source = torch.load(ROOT / "checkpoints" / "FOX-TR-L1" / source_name, map_location="cpu")
        wrapper = torch.load(ROOT / "checkpoints" / "FOX-TR-L1" / wrapper_name, map_location="cpu")
        self.assertEqual(set(source["model_state"]), set(wrapper["model_state"]))
        for key in source["model_state"]:
            self.assertTrue(torch.equal(source["model_state"][key], wrapper["model_state"][key]), key)
        self.assertEqual(wrapper.get("inference_controller"), "tabux_local_basin")

    def test_tabux_reference_uses_identical_baseline_neural_weights(self):
        self._assert_same_neural_weights("simple_cnn_001.pt", "simple_cnn_tabux_004.pt")

    def test_action_memory_tabux_uses_identical_action_memory_neural_weights(self):
        self._assert_same_neural_weights("action_memory_cnn_003.pt", "action_memory_tabux_cnn_005.pt")

    def test_initial_study_contains_five_complete_runs(self):
        paths = sorted((ROOT / "reference_results" / "initial_study").glob("*.json"))
        self.assertEqual(len(paths), 5)
        models = set()
        for path in paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["metadata"].get("tfp_version"), "1.0.0")
            self.assertEqual(payload["benchmark"].get("episodes"), 180)
            self.assertEqual(payload["benchmark"].get("maps"), 18)
            self.assertEqual(payload["benchmark"].get("seeds"), 10)
            models.add(payload["records"][0]["model"])
        self.assertEqual(models, {
            "simple_cnn_001",
            "simple_cnn_tabu_002",
            "action_memory_cnn_003",
            "simple_cnn_tabux_004",
            "action_memory_tabux_cnn_005",
        })


if __name__ == "__main__":
    unittest.main()
