"""Prove that prompt construction is centralized and independent of annotations."""

import json
import sys
from pathlib import Path

import pytest


EVALUATOR = Path(__file__).resolve().parents[1]
EVAL_ROOT = Path(__file__).resolve().parents[2] / "evals_localization"
for path in (EVALUATOR, EVAL_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run
import tasks


MIXED = "mixed_special_xyxy_grouped_all_instances_v2"


def record(dataset, category, task_name=None, protocol="box", family="detection"):
    """Create one prompt-safe benchmark record."""
    return {
        "dataset_name": dataset,
        "categories": [category],
        "task_name": task_name,
        "benchmark_protocol": protocol,
        "benchmark_family": family,
    }


def test_v022_is_rejected_before_inference():
    with pytest.raises(ValueError, match="retired.*GT cardinality"):
        run.prompt_for(record("COCO", "car"), "elliot-v022", MIXED)
    with pytest.raises(ValueError, match="retired.*GT cardinality"):
        run.prompt_protocol_for("elliot-v022")


@pytest.mark.parametrize("gt", [
    {},
    {"car": [[1, 2, 3, 4]]},
    {"car": [[1, 2, 3, 4], [5, 6, 7, 8]]},
    {"car": [[900, 800, 700, 600]]},
])
def test_detection_prompt_is_gt_invariant(gt):
    sample = record("Dense200", "car")
    sample["gt"] = gt
    prompt = run.prompt_for(sample, "elliot-loc", MIXED)
    expected = run.prompt_for(record("Dense200", "car"), "elliot-loc", MIXED)
    assert prompt == expected


@pytest.mark.parametrize("sample", [
    record("COCO", "car", "common_object_detection", "box", "common_detection"),
    record("Dense200", "chicken", "common_object_detection", "box", "dense_detection"),
    record("HumanRef", "people in the bottom row", "pointing_referring", "point", "pointing"),
    record("HumanRef", "people in the bottom row", "referring_object_detection", "box", "referring"),
    record("RefCOCOg_val", "the red cup", "referring_object_detection", "box", "referring"),
    record("sspro", "submit button", "gui_pointing", "gui", "gui_grounding"),
    record("HierText", "OPEN", "common_object_detection", "box", "ocr_localization"),
])
def test_every_task_family_is_gt_invariant(sample):
    first = dict(sample, gt={})
    second = dict(sample, gt={sample["categories"][0]: [[1, 2, 3, 4]] * 20})
    assert run.prompt_for(first, "elliot-loc", MIXED) == run.prompt_for(
        second, "elliot-loc", MIXED
    )


def test_detection_uses_existing_task_registry_then_one_output_contract():
    sample = record("Dense200", "chicken", family="dense_detection")
    prompt = run.prompt_for(sample, "elliot-loc", MIXED)
    assert prompt.startswith(tasks.build_prompt_for_entry(sample, "elliot-loc"))
    assert prompt.count("Respond only with the requested JSON-like structure") == 1
    assert "ALL matching instances" in prompt
    assert "Group ALL boxes for the same label" in prompt


def test_referring_datasets_do_not_reveal_a_cardinality_regime():
    human = run.prompt_for(
        record("HumanRef", "the red cup", family="referring"),
        "elliot-loc", MIXED,
    )
    referring = run.prompt_for(
        record("RefCOCOg_val", "the red cup", family="referring"),
        "elliot-loc", MIXED,
    )
    assert human == referring
    assert "HumanRef" not in human
    assert "matching person" not in human
    assert "requested region" not in human
    assert "ALL matching instances" in human


def test_ocr_queries_preserve_all_supplied_transcriptions_and_quotes():
    sample = {
        **record("HierText", "unused", family="ocr_localization"),
        "categories": ["OPEN", 'How to Pay & Download "Products".'],
    }
    prompt = run.prompt_for(sample, "elliot-loc", MIXED)
    assert '1. "OPEN"' in prompt
    assert '2. "How to Pay & Download \\"Products\\"."' in prompt
    assert prompt.count("Return each box as [x_min,y_min,x_max,y_max].") == 1


def test_point_and_gui_output_contracts_have_correct_arity():
    point = run.prompt_for(
        record("COCO", "donut", "pointing", "point", "pointing"),
        "elliot-loc", MIXED,
    )
    gui = run.prompt_for(
        record("screenspot_v2", "search icon", "gui_pointing", "gui", "gui_grounding"),
        "elliot-loc", MIXED,
    )
    assert "Return each point as [x,y]." in point
    assert "[x_min,y_min,x_max,y_max]" not in point
    assert "Locate every region matching the description" in gui
    assert "Return each box as [x_min,y_min,x_max,y_max]." in gui
    assert "requested region" not in gui


def test_refspatial_official_question_is_compiled_by_tasks_registry():
    sample = {
        **record("RefSpatialBench_Location", "target", "pointing", "point", "pointing"),
        "evaluation": {
            "metric": "refspatial_accuracy",
            "question": "Point to the empty area.",
        },
    }
    assert tasks.build_prompt_for_entry(sample, "elliot-loc") == "Point to the empty area."
    assert "point_2d" in tasks.build_prompt_for_entry(sample, "qwen")
    assert "Return each point as [x,y]." in run.prompt_for(sample, "elliot-loc", MIXED)


def test_output_contract_rejects_unknown_combinations():
    sample = record("COCO", "car")
    with pytest.raises(ValueError, match="Unknown output format"):
        tasks.build_prompt_for_entry(sample, "elliot-loc", "mystery")
    with pytest.raises(ValueError, match="only defined for elliot-loc"):
        tasks.build_prompt_for_entry(sample, "qwen", MIXED)


def test_all_850_records_compile_without_annotations():
    manifest = Path(__file__).resolve().parents[2] / "manifests/benchmark_850_v4.jsonl"
    rows = [json.loads(line) for line in manifest.open() if line.strip()]
    assert len(rows) == 850
    prompts = []
    for row in rows:
        stripped = {key: value for key, value in row.items() if key != "gt"}
        prompt = run.prompt_for(stripped, "elliot-loc", MIXED)
        assert prompt
        assert prompt.count("Respond only with the requested JSON-like structure") == 1
        prompts.append(prompt)
    assert len(prompts) == 850
    assert len(set(prompts)) > 100


def test_prompt_protocol_is_new_registry_contract():
    assert run.PROMPT_PROTOCOL == "localization-v5-cardinality-neutral-trained-contract"
    assert run.prompt_protocol_for("elliot-loc") == run.PROMPT_PROTOCOL
    assert run.PROMPT_PROTOCOL == tasks.LOCALIZATION_PROTOCOL_ID


def test_manifest_exact_preserves_the_prepared_training_prompt_verbatim():
    prompt = "Locate every car.\nReturn JSON only."
    assert run.prompt_for({"prompt": prompt}, "manifest-exact", MIXED) == prompt
    assert run.prompt_protocol_for("manifest-exact") == "manifest-exact-v1"
    with pytest.raises(ValueError, match="non-empty prompt"):
        run.prompt_for({}, "manifest-exact", MIXED)


def test_humanref_baseline_prompts_preserve_all_people():
    sample = record("HumanRef", "person wearing red", "referring", "box", "referring")
    assert run.prompt_for(sample, "rex-omni") == (
        'Detect all people matching "person wearing red". Output a separate '
        'bounding box in [x0, y0, x1, y1] format for each person.')
    assert run.prompt_for(sample, "qwen") == (
        'Locate all people matching "person wearing red". Return a JSON list '
        'with bbox_2d and label for every matching person. Use coordinates '
        'from 0 to 1000 and the description as label.')
    sample["dataset_name"] = "RefCOCOg"
    assert "all people" not in run.prompt_for(sample, "qwen")
    sample.update(dataset_name="HumanRef", benchmark_protocol="point")
    assert "Point to" in run.prompt_for(sample, "qwen")
