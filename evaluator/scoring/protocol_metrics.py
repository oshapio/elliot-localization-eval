"""Metrics for benchmark protocols outside the shared Rex/FastEvaluate path."""

import contextlib
import io


def suite_summary(datasets):
    """Only report complete suites; RefSpatial averages queries, ODinW datasets."""
    out = {}
    spatial = [datasets.get("point/RefSpatialBench_" + split)
               for split in ("Location", "Placement", "Unseen")]
    if all(spatial):
        total = sum(dataset["diagnostics"]["samples"] for dataset in spatial)
        out["RefSpatialBench"] = {
            "metric": "accuracy",
            "samples": total,
            "score": sum(
                dataset["metrics"]["accuracy"] * dataset["diagnostics"]["samples"]
                for dataset in spatial
            ) / total,
        }
    odinw = [dataset for name, dataset in datasets.items()
             if name.startswith("box/ODinW13_")]
    if len(odinw) == 13:
        out["ODinW13"] = {
            "metric": "mean_dataset_mAP",
            "datasets": 13,
            "score": sum(dataset["metrics"]["mAP"] for dataset in odinw) / 13,
        }
    return out


def refspatial_accuracy(rows):
    """Official mask-hit fraction per answer, then mean over answers."""
    from pycocotools import mask as mask_utils

    values = []
    for row in rows:
        masks = [mask for items in row["gt"].values() for mask in items]
        if len(masks) != 1:
            raise ValueError("RefSpatial requires exactly one valid-region mask")
        mask = mask_utils.decode(masks[0])
        points = [point for items in row["predictions"].values() for point in items]
        hits = 0
        if row["status"] != "error":
            for point in points:
                if len(point) != 2:
                    raise ValueError("RefSpatial predictions must be points")
                x, y = point
                if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]:
                    hits += int(mask[int(y), int(x)] != 0)
        values.append(hits / len(points) if points else 0.0)
    return {"accuracy": sum(values) / len(values), "n": len(values)}


def odinw_metrics(rows):
    """COCO AP with every detection confidence fixed at 1, as in Qwen."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    paths = {row["evaluation"]["coco_gt"] for row in rows}
    if len(paths) != 1:
        raise ValueError("Mixed ODinW ground truths")
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(next(iter(paths)))
        names = {value["name"]: key for key, value in gt.cats.items()}
        predictions = []
        unknown = 0
        for row in rows:
            if row["status"] == "error":
                continue
            for label, boxes in row["predictions"].items():
                if label not in names:
                    unknown += len(boxes)
                    continue
                for box in boxes:
                    if len(box) != 4:
                        raise ValueError("ODinW predictions must be boxes")
                    x1, y1, x2, y2 = box
                    predictions.append({
                        "image_id": row["evaluation"]["image_id"],
                        "category_id": names[label],
                        "score": 1.0,
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                    })
        if predictions:
            dt = gt.loadRes(predictions)
        else:
            dt = COCO()
            dt.dataset = {
                "images": gt.dataset["images"],
                "categories": gt.dataset["categories"],
                "annotations": [],
            }
            dt.createIndex()
        evaluator = COCOeval(gt, dt, "bbox")
        evaluator.params.imgIds = [row["evaluation"]["image_id"] for row in rows]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return {
        "mAP": float(evaluator.stats[0]),
        "AP50": float(evaluator.stats[1]),
        "AP75": float(evaluator.stats[2]),
        "unknown_label_targets": unknown,
    }
