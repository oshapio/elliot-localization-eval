# evals_localization

Localization/grounding evals (Rex-Omni protocol) for Elliot and baseline models.

## Layout

- `vendor/rex_omni/` — verbatim copies from https://github.com/IDEA-Research/Rex-Omni
  (`evaluation/`, fetched 2026-07-30, master). DO NOT EDIT — this is the reference
  scorer that produced the published Rex-Omni / LocateAnything tables.
  - `metrics/other_metric.py` — scorer for everything except COCO/LVIS box
    (per-task matching keyed off `task_name` in each answer.jsonl line).
  - `metrics/coco_lvis_metric.py` + `utils/convert_coco_lvis_to_standard_format.py`
    — COCO/LVIS box F1 via fastevaluate against the official GT jsons.
  - `inference_text_prompt.py` / `inference_visual_prompt.py` — Rex-Omni's own
    inference (vLLM + Qwen2.5-VL recipe). Reference only; our runner replaces it.
- `vendor/fastevaluate/` — C++ scorer package (copied from eagle/Embodied, which
  svn-exported it from Rex-Omni). Install once: `pip install -e vendor/fastevaluate`.
- `EvalData` — symlink to the downloaded data (../../elliot-locate-eval/EvalData,
  25 GB: Rex-Omni-EvalData + ScreenSpot-Pro + generated converted_box.jsonl).
- `scripts/` — ours: data download (`download_evaldata.sh`,
  `make_sspro_converted_box.py`) and (to come) the model-agnostic inference runner.
- `notebooks/`, `outputs/` — working area (outputs gitignored).

## Provenance check (2026-07-30)

The Embodied fork's scorers (`eagle/Embodied/evaluation/metrics/`) were diffed
against these Rex-Omni originals at the AST level. Differences are cosmetic or
defensive only; identical numbers on well-formed data:
- comments translated zh -> en, default paths changed;
- Embodied `calculate_pointing_metrics` additionally accepts bbox-format GT
  (point-in-box) besides RLE masks — mask path unchanged;
- Embodied adds isinstance guards + warnings in `calculate_metrics_for_sample`;
- makedirs guard in `main`.

## Protocol notes (established from both repos)

- Interface: inference writes answer.jsonl lines
  `{image_path, extracted_predictions: {label: [[x1,y1,x2,y2]|[x,y],...]}, gt, dataset_name, task_name}`
  in ABSOLUTE pixels; scorers read only that file. For point_eval, `gt` is replaced
  by the RLE `gt_mask`.
- COCO/LVIS box positive_only protocol: model is queried only with GT-positive categories.
- Pointing on COCO/LVIS/VisDrone: one query per category; parsed points force-assigned
  to the queried category.
- Rex-Omni decodes greedy (temperature=0, top_k=1); the Embodied fork samples at
  temperature=0.7 — decoding regime differs between the two papers' tables.
- Per-model prompt/coordinate recipes differ by design (Rex/Qwen2.5: [x0,y0,x1,y1]
  in resized-input pixel space; Qwen3.5: 0-1000 integers; Elliot: 0-1 decimals;
  LocateAnything: <ref>/<box> tokens 0-1000). Fairness = same data + same scorer +
  same task contract; recipes are per-model best-effort, with parse rate reported.
