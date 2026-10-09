#!/usr/bin/env python3
"""Check that capped generations are selectively retried."""

import sys
from pathlib import Path


EVALUATOR_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVALUATOR_ROOT))

import run


def test_retry_capped():
    """Replace only capped outputs with a larger-budget generation."""
    calls = []

    def predict_batch(records, prompts, max_new_tokens):
        calls.append(([record["sample_id"] for record in records], max_new_tokens))
        return [{"predictions": {"cat": [[1, 2, 3, 4]]}} for record in records]

    adapter = {"name": "test", "config_sha256": "test", "predict_batch": predict_batch}
    records = [
        {"sample_id": "a", "dataset_name": "d", "image_path": "a.jpg", "gt": {}},
        {"sample_id": "b", "dataset_name": "d", "image_path": "b.jpg", "gt": {}},
    ]
    prompts = ["a", "b"]
    outputs = [
        run.prediction_record(records[0], prompts[0], adapter, {"hit_token_cap": False}, 512),
        run.prediction_record(records[1], prompts[1], adapter, {"hit_token_cap": True}, 512),
    ]

    retried = run.retry_capped(adapter, records, prompts, outputs, 4096, True)

    assert calls == [(["b"], 4096)]
    assert retried[0]["max_new_tokens"] == 512
    assert not retried[0]["retried_after_token_cap"]
    assert retried[1]["max_new_tokens"] == 4096
    assert retried[1]["retried_after_token_cap"]
    assert retried[1]["predictions"] == {"cat": [[1.0, 2.0, 3.0, 4.0]]}


if __name__ == "__main__":
    test_retry_capped()
    print("token-cap retry test passed")
