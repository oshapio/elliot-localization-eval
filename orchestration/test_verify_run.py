import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("verify_run.py")
SPEC = importlib.util.spec_from_file_location("verify_run", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class VerifyRunTests(unittest.TestCase):
    def test_manifest_shape_matches_full_group_definition(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.jsonl"
            rows = [
                {"benchmark_dataset": "COCO", "benchmark_protocol": "box"},
                {"benchmark_dataset": "COCO", "benchmark_protocol": "point"},
                {"benchmark_dataset": "COCO", "benchmark_protocol": "box"},
            ]
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            self.assertEqual(MODULE.manifest_shape(path), (3, 2))


if __name__ == "__main__":
    unittest.main()
