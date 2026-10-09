#!/usr/bin/env python3
"""Generate ScreenSpot-Pro converted_box.jsonl (the Embodied README's dead 'TODO' link).

Reads ScreenSpot-Pro annotations/*.json and emits one JSONL line per sample with
the fields expected by eagle/Embodied/evaluation/inference_sspro_ddp.py
(file_name, instruction, answer) and metrics/sspro_metric.py
(img_filename + data_type/data_source, with ui_type/group fallback).

Source bbox format is absolute-pixel xyxy (verified empirically 2026-07-29:
median element size 69x23 px under xyxy; xywh reading gives implausible sizes).
"""
import argparse
import glob
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sspro_dir", required=True, help="ScreenSpot-Pro root (contains annotations/ and images/)")
    ap.add_argument("--out", required=True, help="Output converted_box.jsonl path")
    args = ap.parse_args()

    ann_files = sorted(glob.glob(os.path.join(args.sspro_dir, "annotations", "*.json")))
    if not ann_files:
        raise SystemExit(f"ERROR: no annotation jsons under {args.sspro_dir}/annotations")

    n = 0
    missing_img = 0
    bad_box = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for path in ann_files:
            with open(path, encoding="utf-8") as f:
                records = json.load(f)
            for r in records:
                img = r["img_filename"]
                if not os.path.exists(os.path.join(args.sspro_dir, "images", img)):
                    missing_img += 1
                x1, y1, x2, y2 = map(float, r["bbox"])
                if x2 <= x1 or y2 <= y1:
                    bad_box += 1
                entry = {
                    "file_name": img,
                    "img_filename": img,
                    "instruction": r["instruction"],
                    "answer": [x1, y1, x2, y2],
                    "data_type": r.get("ui_type", ""),
                    "data_source": r.get("group", ""),
                    "ui_type": r.get("ui_type", ""),
                    "group": r.get("group", ""),
                    "application": r.get("application", ""),
                    "platform": r.get("platform", ""),
                    "img_size": r.get("img_size"),
                    "id": r.get("id"),
                }
                out.write(json.dumps(entry, ensure_ascii=False) + "\n")
                n += 1

    print(f"wrote {n} samples from {len(ann_files)} annotation files -> {args.out}")
    if missing_img:
        print(f"WARNING: {missing_img} samples reference images missing under images/")
    if bad_box:
        print(f"WARNING: {bad_box} degenerate boxes (x2<=x1 or y2<=y1)")


if __name__ == "__main__":
    main()
