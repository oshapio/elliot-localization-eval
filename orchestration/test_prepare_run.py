import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("prepare_run.py")
SPEC = importlib.util.spec_from_file_location("prepare_run", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class PrepareRunTests(unittest.TestCase):
    def test_source_must_be_clean(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
            tracked = root / "tracked.txt"
            tracked.write_text("clean\n")
            subprocess.run(["git", "add", "tracked.txt"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-qm", "test"], cwd=root, check=True)
            self.assertTrue(MODULE.validate_source(root))
            (root / "hidden-launcher.sh").write_text("#!/bin/bash\n")
            with self.assertRaises(SystemExit):
                MODULE.validate_source(root)

    def test_manifest_contract_counts_rows_and_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.jsonl"
            path.write_text(
                json.dumps({"sample_id": "a", "group": "one"}) + "\n" +
                json.dumps({"sample_id": "b", "group": "two"}) + "\n"
            )
            self.assertEqual(MODULE.validate_manifest(path, 2, 2), (2, 2))

    def test_manifest_contract_rejects_wrong_count(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.jsonl"
            path.write_text(json.dumps({"sample_id": "a", "group": "one"}) + "\n")
            with self.assertRaises(SystemExit):
                MODULE.validate_manifest(path, 2, 1)

    def test_full_groups_are_dataset_protocol_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in [
                {"sample_id": "a", "benchmark_dataset": "COCO", "benchmark_protocol": "box"},
                {"sample_id": "b", "benchmark_dataset": "COCO", "benchmark_protocol": "point"},
                {"sample_id": "c", "benchmark_dataset": "LVIS", "benchmark_protocol": "box"},
            ]) + "\n")
            self.assertEqual(MODULE.validate_manifest(path, 3, 3), (3, 3))

    def test_config_always_binds_prepared_hf_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "config.json"
            source.write_text(json.dumps({
                "name": "old",
                "model_path": "/old/model",
                "max_model_len": 16384,
                "max_num_seqs": 16,
                "tensor_parallel_size": 1,
                "max_pixels": 4194304,
                "min_pixels": 65536,
                "coordinate_max": 1000,
                "output_format": "mixed_special_xyxy_grouped_all_instances_v2",
            }))
            target = root / "run" / "hf_export"
            config = MODULE.load_config(source, "new", target)
            self.assertEqual(config["name"], "new")
            self.assertEqual(config["model_path"], str(target))


if __name__ == "__main__":
    unittest.main()
