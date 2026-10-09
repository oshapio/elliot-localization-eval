"""Multi-model task/prompt registry. Structure from rex_omni/tasks.py; "rex-omni"
templates are verbatim (typos included). qwen = 0-1000 ints, elliot = 0-1 decimals."""
import json
from enum import Enum


class TaskType(Enum):
    DETECTION = "detection"
    REFERRING = "referring"
    POINTING = "pointing"
    VISUAL_PROMPTING = "visual_prompting"
    KEYPOINT = "keypoint"
    OCR_BOX = "ocr_box"
    OCR_POLYGON = "ocr_polygon"
    GUI_DETECTION = "gui_grounding"
    GUI_POINTING = "gui_pointing"


COORD = {
    "rex-omni": {"style": "special_tokens_0_999", "scale": None, "range_text": None},
    "qwen":     {"style": "int_0_1000", "scale": 1000.0, "range_text": "integers from 0 to 1000"},
    "elliot":   {"style": "decimal_0_1", "scale": 1.0, "range_text": "decimal values from 0 to 1"},
    "elliot-loc": {"style": "int_0_1000", "scale": 1000.0,
                   "range_text": "integers from 0 to 1000"},
}
MODEL_FAMILIES = tuple(COORD)

# validated referring recipe: grounding-comparison "short" (100% parse; used in full runs)
_REFER = 'Locate "{expression}". Return only [x1, y1, x2, y2] with coordinates in {range_text}.'
_QE_DETECT = ("Detect every instance of the following categories: {categories}. Output one line "
              "per instance in the format label: [x1, y1, x2, y2], with coordinates as {range_text}.")
# qwen's TRAINED grounding format. Measured on 25 COCO samples (2026-07-31): 164/165 boxes
# carry a label vs 0/160 with _QE_DETECT, same box count and same localization quality
# (mean best-IoU 0.632 vs 0.644); coords stay 0-1000 (IoU 0.63 vs 0.03 for resized-pixel).
_QWEN_DETECT = ('Locate every instance of the following categories: {categories}. Output the '
                'coordinates in JSON format, with "bbox_2d" and "label" for each instance.')
_QE_POINT = "Point to each {categories}. Output one [x, y] point per instance, with coordinates as {range_text}."

# per task: templates per family + output format; eval_override = wording the model's
# own eval pipeline sends where it differs (that wording made the published numbers)
TASKS = {
    TaskType.DETECTION: {
        "output": "boxes",
        "templates": {
            "rex-omni": "Detect {categories}. Output the bounding box coordinates in [x0, y0, x1, y1] format.",
            # elliot was never SFT'd on a grounding format -> keep the explicit instruction
            "qwen": _QWEN_DETECT, "elliot": _QE_DETECT,
            "elliot-loc": ("Locate every {categories} in the image. Output a separate tight box "
                           "for each matching instance; do not merge instances."),
        },
    },
    TaskType.REFERRING: {
        "output": "boxes",
        "templates": {
            # rex referring = their detection template with the expression as category
            "rex-omni": "Detect {expression}. Output the bounding box coordinates in [x0, y0, x1, y1] format.",
            "qwen": _REFER, "elliot": _REFER,
            "elliot-loc": "Locate every region matching the description: {expression}.",
        },
    },
    TaskType.POINTING: {
        # protocol: COCO/LVIS/VisDrone pointing queries ONE category at a time
        "output": "points",
        "templates": {"rex-omni": "Point to {categories}.", "qwen": _QE_POINT,
                      "elliot": _QE_POINT,
                      "elliot-loc": "Point to all targets matching the description: {categories}."},
    },
    TaskType.VISUAL_PROMPTING: {
        "output": "boxes",
        "templates": {
            "rex-omni": ("Given reference boxes {visual_prompt} indicating one or more objects, "
                         "find all similar objects in the image and output their bounding boxes."),
            "qwen": ("Reference boxes {visual_prompt} mark example objects. Find all similar objects, "
                     "one [x1, y1, x2, y2] box per instance, coordinates as {range_text}."),
            "elliot": ("Reference boxes {visual_prompt} mark example objects. Find all similar objects, "
                       "one [x1, y1, x2, y2] box per instance, coordinates as {range_text}."),
        },
    },
    TaskType.KEYPOINT: {
        "output": "keypoints",
        "templates": {
            "rex-omni": ("Can you detect each {categories} in the image using a [x0, y0, x1, y1] box format, "
                         "and then provide the coordinates of its {keypoints} as [x0, y0]? "
                         "Output the answer in JSON format."),
            "qwen": ("Detect each {categories} with a [x0, y0, x1, y1] box and the coordinates of its "
                     "{keypoints} as [x0, y0], all as {range_text}. Output JSON."),
            "elliot": ("Detect each {categories} with a [x0, y0, x1, y1] box and the coordinates of its "
                       "{keypoints} as [x0, y0], all as {range_text}. Output JSON."),
        },
    },
    TaskType.OCR_BOX: {
        # scoring matches predicted TRANSCRIPTIONS as labels (joint spotting+recognition)
        "output": "boxes_with_text",
        "templates": {
            "rex-omni": "Detect all {categories} and recognize them.",
            "qwen": ("Detect all {categories}. Output one line per instance as text: [x1, y1, x2, y2], "
                     "text = transcription, coordinates as {range_text}."),
            "elliot": ("Detect all {categories}. Output one line per instance as text: [x1, y1, x2, y2], "
                       "text = transcription, coordinates as {range_text}."),
        },
        # rex tasks.py differs from their eval script; the eval wording made the published numbers
        "eval_override": {"rex-omni": ("Can you detect all the {categories} in this image in box format "
                                       "like [x0, y0, x1, y1] and then recognize them?")},
    },
    TaskType.OCR_POLYGON: {
        "output": "polygons_with_text",
        "templates": {
            "rex-omni": ("Can you detect all {categories} in this image in polygon format like "
                         "[x0, y0, x1, y1, x2, y2 ...] and then recognize them?"),
            "qwen": ("Detect all {categories} in polygon format [x0, y0, x1, y1, x2, y2, ...] and "
                     "recognize them. Coordinates as {range_text}."),
            "elliot": ("Detect all {categories} in polygon format [x0, y0, x1, y1, x2, y2, ...] and "
                       "recognize them. Coordinates as {range_text}."),
        },
    },
    TaskType.GUI_DETECTION: {
        "output": "boxes",
        "templates": {
            "rex-omni": 'Detect element "{expression}"" in the image.',   # doubled quote = THEIR typo
            "qwen": _REFER, "elliot": _REFER,   # box is legal for sspro metric (center-in-GT-box)
            "elliot-loc": "Locate every region matching the description: {expression}.",
        },
    },
    TaskType.GUI_POINTING: {
        "output": "points",
        "templates": {
            "rex-omni": 'Point to element "{expression}".',   # published sspro anchor: Avg 36.8
            "qwen": 'Point to the element described by: "{expression}". Output one [x, y] point as {range_text}.',
            "elliot": 'Point to the element described by: "{expression}". Output one [x, y] point as {range_text}.',
        },
    },
}

# dataset -> task, listed EXPLICITLY: unknown names raise, no silent detection fallback
DATASET_TASKS = {
    "COCO": TaskType.DETECTION, "LVIS": TaskType.DETECTION, "LVIS_fullval": TaskType.DETECTION,
    "Dense200": TaskType.DETECTION, "VisDrone": TaskType.DETECTION, "DocLayNet": TaskType.DETECTION,
    # M6Doc: jsonl task_name says referring (scored label-agnostic) but the prompt is plain detection
    "M6Doc": TaskType.DETECTION,
    "TotalText": TaskType.OCR_BOX, "SROIE": TaskType.OCR_BOX,
    "IC15": TaskType.OCR_BOX, "HierText": TaskType.OCR_BOX,
    "HumanRef": TaskType.REFERRING, "RefCOCOg": TaskType.REFERRING,
    "RefCOCOg_val": TaskType.REFERRING, "RefCOCOg_test": TaskType.REFERRING,
    "sspro": TaskType.GUI_POINTING, "screenspot_v2": TaskType.GUI_POINTING,
    "FSCD_test": TaskType.VISUAL_PROMPTING,
    "ap10k": TaskType.KEYPOINT,
}
OCR_DATASETS = {d for d, t in DATASET_TASKS.items() if t is TaskType.OCR_BOX}
REFERRING_DATASETS = {d for d, t in DATASET_TASKS.items() if t is TaskType.REFERRING}
PER_CATEGORY_POINTING = {"COCO", "LVIS", "LVIS_fullval", "VisDrone"}  # one query PER category


def task_for_entry(entry):
    if entry.get('task_name') == 'referring':
        return TaskType.REFERRING
    if entry.get('task_name') == 'detection':
        return TaskType.DETECTION
    # point/keypoint/visual-prompt jsonls reuse the same dataset names -> sniff the entry
    if entry.get("task_name") in ("pointing", "pointing_referring"):
        return TaskType.POINTING
    if "keypoint_names" in entry:
        return TaskType.KEYPOINT
    if "visual_prompt" in entry:
        return TaskType.VISUAL_PROMPTING
    return DATASET_TASKS[entry["dataset_name"]]


def build_prompt(task, model, categories=None, expression=None, keypoints=None,
                 visual_prompt=None, eval_protocol=False):
    cfg = TASKS[task]
    template = cfg["templates"][model]
    if eval_protocol:
        template = cfg.get("eval_override", {}).get(model, template)
    cats = ", ".join(categories) if isinstance(categories, (list, tuple)) else categories
    values = {
        "categories": cats,
        "expression": expression if expression is not None else cats,
        "range_text": COORD[model]["range_text"],
        "keypoints": ", ".join(keypoints) if keypoints else None,
        "visual_prompt": visual_prompt,
    }
    needed = {k: v for k, v in values.items() if "{" + k + "}" in template}
    missing = [k for k, v in needed.items() if v is None]
    if missing:
        raise ValueError(f"{task.value}/{model}: missing template values {missing}")
    return template.format(**needed)


MIXED_SPECIAL_XYXY_GROUPED = "mixed_special_xyxy_grouped_all_instances_v2"
LOCALIZATION_PROTOCOL_ID = "localization-v5-cardinality-neutral-trained-contract"


def task_for_model_entry(entry, model):
    """Resolve task semantics for one model without consulting annotations."""
    protocol = entry.get("benchmark_protocol")
    task = (TaskType.POINTING if protocol == "point" else
            TaskType.GUI_POINTING if protocol == "gui" else task_for_entry(entry))
    # Rex was trained to point at GUI elements. The other reference families use
    # a box, whose centre is legal under the GUI scorer.
    if task is TaskType.GUI_POINTING and model != "rex-omni":
        task = TaskType.GUI_DETECTION
    return task


def _base_prompt_for_entry(entry, model):
    """Build task/model wording from prompt-safe record fields only."""
    spec = entry.get("evaluation") or {}
    categories = entry["categories"]
    categories = [categories] if isinstance(categories, str) else list(categories)

    # RefSpatial supplies the benchmark question itself. Model-specific response
    # wording still belongs here rather than in a metric module.
    if spec.get("metric") == "refspatial_accuracy":
        question = spec["question"].strip()
        if model == "qwen":
            return (question + '\nOutput the point coordinates in JSON format, with '
                    '"point_2d" and "label". Use integers from 0 to 1000.')
        return question

    family = entry.get("benchmark_family")

    # These are supplied-transcription localization queries, not generic OCR
    # spotting prompts. Keep their exact benchmark contract in the task registry.
    if family == "ocr_localization":
        if model == "qwen":
            items = json.dumps(categories, ensure_ascii=False)
            return (f"Locate every instance of these text items: {items}. "
                    'Return a JSON list with bbox_2d and label for each box. '
                    'Use coordinates from 0 to 1000 and the exact text as label.')
        if model == "rex-omni":
            items = json.dumps(categories, ensure_ascii=False)
            return (f"Detect these text items: {items}. "
                    "Output the bounding box coordinates in [x0, y0, x1, y1] format.")
        if len(categories) == 1:
            return f"Locate the text item {json.dumps(categories[0], ensure_ascii=False)}."
        items = "\n".join(
            f"{index}. {json.dumps(text, ensure_ascii=False)}"
            for index, text in enumerate(categories, 1)
        )
        return f"Locate each text item below:\n{items}"

    task = task_for_model_entry(entry, model)
    expression_tasks = (TaskType.REFERRING, TaskType.GUI_DETECTION, TaskType.GUI_POINTING)
    kwargs = ({"expression": categories[0]} if task in expression_tasks
              else {"categories": categories})
    if task is TaskType.KEYPOINT:
        kwargs["keypoints"] = entry.get("keypoint_names")
    if task is TaskType.VISUAL_PROMPTING:
        kwargs["visual_prompt"] = entry.get("visual_prompt")
    return build_prompt(task, model, eval_protocol=(model == "rex-omni"), **kwargs)


def _mixed_special_prompt(entry, prompt):
    """Append the trained grouped contract without revealing answer cardinality."""
    point = entry.get("benchmark_protocol") == "point"
    if point:
        instructions = (
            "Return a separate point for ALL matching instances. "
            "Group all points under their matching label; do not merge instances.\n"
            "Return each point as [x,y]."
        )
    else:
        instructions = (
            "Return a separate tight box for ALL matching instances of every requested label. "
            "Group ALL boxes for the same label in its coordinate list; "
            "do not merge instances or limit the answer to one box per label.\n"
            "Return each box as [x_min,y_min,x_max,y_max]."
        )
    return (prompt + "\n" + instructions
            + "\nRespond only with the requested JSON-like structure grouped by label.")


def build_prompt_for_entry(entry, model, output_format=None):
    """Compile one complete prompt; annotations are intentionally unnecessary."""
    if model not in MODEL_FAMILIES:
        raise ValueError(f"Unknown prompt model family: {model}")
    prompt = _base_prompt_for_entry(entry, model)
    if output_format in (None, "ref_box"):
        return prompt
    if output_format == MIXED_SPECIAL_XYXY_GROUPED:
        if model != "elliot-loc":
            raise ValueError(f"{output_format} is only defined for elliot-loc")
        return _mixed_special_prompt(entry, prompt)
    raise ValueError(f"Unknown output format: {output_format}")
