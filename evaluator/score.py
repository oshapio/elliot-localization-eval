#!/usr/bin/env python3
"""Score the cross-domain benchmark without mixing incompatible protocols."""

import argparse
import contextlib
import io
import json
from collections import defaultdict
from pathlib import Path

from scoring import rex as common


def parse_args():
    """Parse predictions, EvalData, and output paths."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--eval-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def write_jsonl(path, rows):
    """Write scorer-compatible JSONL records."""
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def gui_point(target):
    """Convert a predicted point or box to one click point."""
    if len(target) == 2:
        return target
    if len(target) == 4:
        return [(target[0] + target[2]) / 2, (target[1] + target[3]) / 2]
    raise ValueError(f"Unsupported GUI target with {len(target)} coordinates")


def gui_accuracy(rows):
    """Measure whether any predicted center falls inside the target UI box."""
    correct = 0
    by_type = defaultdict(lambda: [0, 0])
    by_source = defaultdict(lambda: [0, 0])
    for row in rows:
        gt_box = next(iter(row["gt"].values()))[0]
        x1, y1, x2, y2 = gt_box
        targets = [target for values in row["predictions"].values() for target in values]
        hit = any(x1 <= gui_point(target)[0] <= x2 and y1 <= gui_point(target)[1] <= y2
                  for target in targets)
        correct += hit
        data_type = row.get("data_type", "unknown") or "unknown"
        data_source = row.get("data_source", "unknown") or "unknown"
        by_type[data_type][0] += hit
        by_type[data_type][1] += 1
        by_source[data_source][0] += hit
        by_source[data_source][1] += 1
    return {
        "accuracy": correct / len(rows),
        "n": len(rows),
        "by_type": {key: value[0] / value[1] for key, value in sorted(by_type.items())},
        "by_source": {key: value[0] / value[1] for key, value in sorted(by_source.items())},
    }


def official_gt(eval_data, dataset):
    """Resolve official COCO/LVIS ground truth for FastEvaluate."""
    if dataset == "COCO":
        return eval_data / "coco" / "instances_val2017.json"
    if dataset == "LVIS":
        return eval_data / "missing_annotaitons" / "lvis_v1_val_with_filename2.json"
    return None


def primary_metric(protocol, metrics):
    """Choose the headline metric appropriate to one protocol."""
    if 'mAP' in metrics:
        return metrics['mAP']
    if 'accuracy' in metrics:
        return metrics['accuracy']
    if "Acc@IoU0.5" in metrics:
        return metrics["Acc@IoU0.5"]
    if protocol == "gui":
        return metrics["accuracy"]
    if protocol == "point":
        return metrics["F1@0.5"]
    return metrics["F1@mean"]


def referring_accuracy(rows):
    """First predicted box accuracy at IoU >= 0.5 for single-target referring."""
    correct = 0
    for row in rows:
        targets = [box for boxes in row['gt'].values() for box in boxes]
        if len(targets) != 1:
            raise ValueError('Referring accuracy requires exactly one ground-truth box')
        predictions = [box for boxes in row['predictions'].values() for box in boxes]
        if not predictions or row['status'] == 'error':
            continue
        a, b = predictions[0], targets[0]
        intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
        union = max(0, a[2]-a[0])*max(0, a[3]-a[1]) + max(0, b[2]-b[0])*max(0, b[3]-b[1]) - intersection
        correct += union > 0 and intersection / union >= 0.5
    return correct / len(rows)


def diagnostics(rows):
    """Summarize output coverage and spatial target counts."""
    return {
        "samples": len(rows),
        "errors": sum(row["status"] == "error" for row in rows),
        "nonempty_predictions": sum(bool(row["predictions"]) for row in rows),
        "predicted_targets": sum(len(values) for row in rows for values in row["predictions"].values()),
        "ground_truth_targets": sum(len(values) for row in rows for values in row["gt"].values()),
    }


def score_group(rows, eval_data, output_dir, scorer):
    """Score one dataset under one output protocol."""
    protocol = rows[0]["benchmark_protocol"]
    dataset = rows[0]["benchmark_dataset"]
    spec = rows[0].get('evaluation') or {}
    if spec.get('metric') == 'referring_accuracy':
        return {'Acc@IoU0.5': referring_accuracy(rows)}
    if spec.get('metric') in ('refspatial_accuracy', 'coco_ap'):
        from scoring.protocol_metrics import odinw_metrics, refspatial_accuracy
        if any((r.get('evaluation') or {}).get('metric') != spec['metric'] for r in rows):
            raise ValueError('Mixed metric definitions within one dataset')
        return (refspatial_accuracy(rows) if spec['metric'] == 'refspatial_accuracy'
                else odinw_metrics(rows))
    if protocol == "gui":
        return gui_accuracy(rows)
    path = output_dir / f"{protocol}__{dataset}.answer.jsonl"
    write_jsonl(path, rows)
    gt_path = official_gt(eval_data, dataset) if protocol == "box" else None
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        metrics = scorer.score(path, coco_gt=gt_path)
    if protocol == 'box' and dataset.startswith('RefCOCO'):
        metrics['Acc@IoU0.5'] = referring_accuracy(rows)
    return metrics


def main():
    """Score every benchmark and write dataset and family summaries."""
    args = parse_args()
    rows = common.read_jsonl(args.predictions)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["benchmark_protocol"], row["benchmark_dataset"])].append(row)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scorer = common.load_scorer()
    datasets = {}
    family_values = defaultdict(list)
    for protocol, dataset in sorted(grouped):
        dataset_rows = grouped[(protocol, dataset)]
        metrics = score_group(dataset_rows, args.eval_data, args.output_dir, scorer)
        key = f"{protocol}/{dataset}"
        value = primary_metric(protocol, metrics)
        family = dataset_rows[0]["benchmark_family"]
        family_values[family].append(value)
        datasets[key] = {
            "family": family,
            "primary_metric": value,
            "metrics": metrics,
            "diagnostics": diagnostics(dataset_rows),
        }
        partial = {
            "samples_scored": sum(item["diagnostics"]["samples"] for item in datasets.values()),
            "datasets": datasets,
        }
        (args.output_dir / "scores.partial.json").write_text(
            json.dumps(partial, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"{key}: {value:.4f}", flush=True)
    summary = {
        "samples": len(rows),
        "family_macro": {
            family: sum(values) / len(values) for family, values in sorted(family_values.items())
        },
        "datasets": datasets,
    }
    output = args.output_dir / "scores.json"
    from scoring.protocol_metrics import suite_summary
    extra = suite_summary(datasets)
    if extra:
        summary['benchmark_summary'] = extra
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
