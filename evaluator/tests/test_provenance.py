"""Protect resumed runs and GUI score breakdowns."""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run


class ProvenanceTest(unittest.TestCase):
    def test_rex_bridge_resolves_repository_eval_tree(self):
        rex_path = Path(__file__).resolve().parents[1] / "scoring" / "rex.py"
        self.assertEqual(
            rex_path.resolve().parents[2] / "evals_localization",
            Path(__file__).resolve().parents[2] / "evals_localization",
        )

    def test_categorized_adapter_aliases_remain_loadable(self):
        self.assertNotIn("discrete_ntp", run.ADAPTER_MODULES)
        adapter = run.load_adapter("oracle", {})
        self.assertEqual(adapter["name"], "oracle")

    def test_changed_batch_size_cannot_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.jsonl"
            manifest.write_text("{}\n")
            args = SimpleNamespace(
                manifest=manifest, prompt_family="elliot-loc", batch_size=8,
                max_new_tokens=512, retry_max_new_tokens=4096,
                shard_index=0, num_shards=1, limit=0,
            )
            adapter = {"name": "empty", "config_sha256": "test"}
            output = Path(directory) / "answers.jsonl"
            run.validate_metadata(output, run.run_metadata(args, adapter))
            args.batch_size = 1
            with self.assertRaisesRegex(ValueError, "resume metadata mismatch"):
                run.validate_metadata(output, run.run_metadata(args, adapter))

    def test_gui_groups_survive_success_and_failure(self):
        record = {
            "sample_id": "gui-1", "dataset_name": "sspro",
            "image_path": "image.png", "gt": {"button": [[0, 0, 10, 10]]},
            "data_type": "text", "data_source": "desktop",
        }
        adapter = {"name": "test", "config_sha256": "test"}
        rows = [
            run.prediction_record(record, "click", adapter, {}, 512),
            run.error_record(record, "click", adapter, RuntimeError("test"), 512),
        ]
        for row in rows:
            self.assertEqual(row["data_type"], "text")
            self.assertEqual(row["data_source"], "desktop")


if __name__ == "__main__":
    unittest.main()
