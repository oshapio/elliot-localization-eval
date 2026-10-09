"""Single-file eval flow: inference -> answer.jsonl -> (convert) -> score.
Prompts come from tasks.py; scoring runs the VENDORED Rex-Omni code in-process
(imported, unmodified). `python eval.py selftest` proves the in-process calls
give byte-identical results to their untouched CLIs. `python eval.py score ...`
scores an existing answer.jsonl (no GPU needed)."""
import json
import os
from pathlib import Path

from paths import ROOT, DATA, VENDOR   # all cluster-specific paths live in paths.py

import fastevaluate as fe
import other_metric as om                                  # vendored scorer (== CLI, proven by selftest)
import coco_lvis_metric as clm                             # vendored: safe_mean/f1 = their summary math
import convert_coco_lvis_to_standard_format as convert_mod # vendored converter
import sspro_metric as sm                                  # vendored from eagle/Embodied (GUI grounding)
import tasks as T

IOU_SWEEP = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
FASTEVAL_DATASETS = {"COCO", "LVIS", "LVIS_fullval"}       # box eval goes through fastevaluate


def load_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def load_dataset(ds):
    """Entries for a dataset key, normalized to the standard schema.
    Keys: box datasets by name ("COCO"), point datasets as "point/COCO", "sspro"."""
    if ds == "sspro":
        return [{"image_path": f"ScreenSpot-Pro/images/{e['file_name']}",
                 "categories": [e["instruction"]],
                 "gt": e["answer"],                        # flat box, as their sspro scorer expects
                 "dataset_name": "sspro", "task_name": None,
                 "data_type": e.get("ui_type", "")}
                for e in load_jsonl(DATA / "ScreenSpot-Pro/converted_box.jsonl")]
    if ds.startswith("point/"):
        entries = load_jsonl(DATA / "_annotations/point_eval" / (ds[6:] + ".jsonl"))
        for e in entries:
            e["gt"] = e["gt_mask"]                         # pointing is scored against RLE masks
        return entries
    return load_jsonl(DATA / "_annotations/box_eval" / (ds + ".jsonl"))


def convert_to_fasteval_tsv(answers_path, coco_json, out_tsv):
    """answer.jsonl -> fastevaluate TSV. Same logic as the vendored converter's main()
    (positive_only mode), written out here for readability and using their helpers for
    the id maps. selftest proves the output is byte-identical to their CLI."""
    info = convert_mod.load_coco_categories(str(coco_json))
    name_to_id, id_to_name = info["name_to_id"], info["id_to_name"]
    image_path_to_id = convert_mod.build_image_path_to_id(info["data"])
    gt_cats = {}                                           # image_id -> category ids with GT
    for ann in info["data"]["annotations"]:
        gt_cats.setdefault(ann["image_id"], []).append(ann["category_id"])

    with open(out_tsv, "w", encoding="utf-8") as out:
        for a in load_jsonl(answers_path):
            image_id = image_path_to_id[os.path.basename(a["image_path"])]
            entries = []
            for cat_name, boxes in a["extracted_predictions"].items():
                cat_key = str(cat_name).lower().replace("_", " ")
                if cat_key not in name_to_id:              # unknown label -> dropped, like theirs
                    continue
                cat_id = name_to_id[cat_key]
                # positive_only, exactly like theirs: only filters when the image HAS gt
                if image_id in gt_cats and cat_id not in gt_cats[image_id]:
                    continue
                for box in boxes:
                    if len(box) != 4:
                        continue
                    x1, y1, x2, y2 = box
                    entries.append({"class": id_to_name[cat_id], "conf": 0.2,
                                    "rect": [x1, y1, x2 - x1, y2 - y1]})
            out.write(f"{image_id}\t{json.dumps(entries, ensure_ascii=False)}\n")
    return out_tsv


# ---------------------------------------------------------------- inference
BATCH = 16          # per-step cost is batch-independent (weight streaming dominates)
FIRST_CAP = 512     # most answers fit; the rest are retried with the full budget below
RETRY_CAP = 4096    # == rex's --max_new_tokens, so nothing is scored as truncated


def run_inference(adapter, data, out_path, indices=None, data_root=DATA, batch=BATCH):
    """adapter = {"name", "question": fn(entry), "predict_batch": fn(images, qs, entries, cap)
    -> [(preds, raw, hit_cap)]}. Writes their answer.jsonl schema, appending so a killed
    job resumes mid-dataset. Two passes: FIRST_CAP for everyone, then RETRY_CAP for answers
    that were still emitting boxes when the budget ran out (prose that produced nothing is
    not retried -- more tokens cannot rescue it, and it would drag the whole batch)."""
    from PIL import Image
    from tqdm.auto import tqdm
    if not isinstance(data, list):
        data = load_jsonl(data)
    idxs = list(indices) if indices is not None else list(range(len(data)))
    done = len(load_jsonl(out_path)) if Path(out_path).exists() else 0
    if done:
        print(f"resuming {out_path}: {done} answers already written")
    idxs = idxs[done:]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    answers, n_empty, n_retry = [], 0, 0
    with open(out_path, "a") as f:
        for s in tqdm(range(0, len(idxs), batch), desc=adapter["name"]):
            chunk = [data[i] for i in idxs[s:s + batch]]
            imgs = [Image.open(data_root / e["image_path"]).convert("RGB") for e in chunk]
            qs = [adapter["question"](e) for e in chunk]
            out = adapter["predict_batch"](imgs, qs, chunk, FIRST_CAP)
            again = [i for i, (preds, _, hit) in enumerate(out) if hit and preds]
            if again:
                n_retry += len(again)
                for i, r in zip(again, adapter["predict_batch"](
                        [imgs[i] for i in again], [qs[i] for i in again],
                        [chunk[i] for i in again], RETRY_CAP)):
                    out[i] = r
            for e, q, (preds, raw, _) in zip(chunk, qs, out):
                n_empty += not preds
                a = {"image_path": e["image_path"], "extracted_predictions": preds, "gt": e["gt"],
                     "question": q, "dataset_name": e["dataset_name"], "task_name": e.get("task_name"),
                     "raw_response": raw, "model": adapter["name"]}
                if "data_type" in e:                        # sspro: ui_type for the grouped accuracies
                    a["data_type"] = e["data_type"]
                answers.append(a)
                f.write(json.dumps(a, ensure_ascii=False) + "\n")
            f.flush()
    print(f"{len(answers)} run, {n_empty} empty parses, {n_retry} retried at {RETRY_CAP} -> {out_path}")
    return answers, n_empty


# ---------------------------------------------------------------- scoring
def _subset_gt(entries, full_gt_json, out_path):
    # restrict official GT to the answered images + populated categories
    # (fastevaluate fills unpopulated categories with -1 sentinels -> poisoned means)
    gt = json.load(open(full_gt_json))
    files = {os.path.basename(a["image_path"]) for a in entries}
    # basename both sides, like their converter (LVIS file_names carry train2017/ prefixes)
    imgs = [i for i in gt["images"] if os.path.basename(i["file_name"]) in files]
    if len(imgs) == len(gt["images"]):
        return full_gt_json                                # full run: use the pristine official file
    ids = {i["id"] for i in imgs}
    anns = [a for a in gt["annotations"] if a["image_id"] in ids]
    cats = {a["category_id"] for a in anns}
    json.dump({"images": imgs, "annotations": anns,
               "categories": [c for c in gt["categories"] if c["id"] in cats]}, open(out_path, "w"))
    return out_path


def _fasteval_summary(res):
    p, r = clm.safe_mean(res["precision"]), clm.safe_mean(res["recall"])          # THEIR summary math
    p50, r50 = clm.safe_mean(res["precision50"]), clm.safe_mean(res["recall50"])
    p95, r95 = clm.safe_mean(res["precision95"]), clm.safe_mean(res["recall95"])
    return {"F1@0.5": clm.f1(p50, r50), "F1@0.95": clm.f1(p95, r95), "F1@mean": clm.f1(p, r)}


def _om_summary(res):
    out = {}
    for iou, blk in sorted(res.items(), key=lambda kv: float(kv[0])):
        m = blk["basic_metrics"][next(iter(blk["basic_metrics"]))]
        p = sum(m["precisions"]) / len(m["precisions"])
        r = sum(m["recalls"]) / len(m["recalls"])
        out[f"F1@{iou}"] = 2 * p * r / (p + r) if p + r > 0 else 0.0
    out["F1@mean"] = sum(out.values()) / len(out)
    return out


def score(answers_path, coco_gt=None):
    """Score one answer.jsonl. sspro -> their sspro scorer; COCO/LVIS box -> their
    converter + fastevaluate vs official GT; everything else -> their other_metric."""
    entries = load_jsonl(answers_path)
    ds, task = entries[0]["dataset_name"], entries[0].get("task_name")
    if ds == "sspro":
        # basename-keyed group/ui_type mapping from converted_box.jsonl
        mapping = {os.path.basename(e["file_name"]): {"source": e.get("group", ""), "type": e.get("ui_type", "")}
                   for e in load_jsonl(DATA / "ScreenSpot-Pro/converted_box.jsonl")}
        r = sm.calculate_metrics(str(answers_path), mapping)
        out = {"accuracy": r["overall_accuracy"], "n": r["total_count"]}
        out.update({c: v["accuracy"] for c, v in sorted(r["category_accuracies"].items())})
        return out
    if ds in FASTEVAL_DATASETS and task not in ("pointing", "pointing_referring"):
        assert coco_gt, f"{ds} box scoring needs the official GT json (coco_gt=...)"
        gt = _subset_gt(entries, coco_gt, str(answers_path) + ".subset_gt.json")
        tsv = convert_to_fasteval_tsv(answers_path, gt, str(answers_path) + ".fast_eval.tsv")
        res = fe.evaluate(str(gt), tsv, 0, 0, "lvis" if ds.startswith("LVIS") else "coco")
        return _fasteval_summary(res)
    ious = [0.5] if task in ("pointing", "pointing_referring") else IOU_SWEEP
    return _om_summary(om.UniversalMetricsCalculator().calculate_all_metrics(entries, ious))


# ---------------------------------------------------------------- selftest (the proof)
# other_metric is imported: their CLI main() is ten lines that call the same
# calculate_all_metrics we call — read it, nothing to execute.
# convert_to_fasteval_tsv was byte-compared against their converter CLI on
# 2026-07-30 (identical output); the invariants below re-verify its behavior.
def selftest():
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="loceval_selftest_"))

    def gt_as_answers(jsonl, out, n=None):
        entries = load_jsonl(jsonl)[:n]
        with open(out, "w") as f:
            for e in entries:
                e["extracted_predictions"] = e["gt"]
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        return entries

    # 1) GT-as-pred through other_metric (TotalText, 300 samples): exactly 1.0
    tt = gt_as_answers(DATA / "_annotations/box_eval/TotalText.jsonl", tmp / "tt.jsonl")
    s = score(tmp / "tt.jsonl")
    assert abs(s["F1@mean"] - 1.0) < 1e-9, s
    print("PASS other_metric path: GT-as-pred F1=1.0 (300 samples, 10 IoUs)")

    def dump(entries, path):
        with open(path, "w") as f:
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        return path

    # 1b) FEWER: first 30 samples predict nothing -> mean P = mean R = F1 = (n-30)/n exactly
    few = [dict(e) for e in tt]
    for e in few[:30]:
        e["extracted_predictions"] = {}
    s = score(dump(few, tmp / "tt_few.jsonl"))
    exp = (len(tt) - 30) / len(tt)
    assert abs(s["F1@0.5"] - exp) < 1e-9 and abs(s["F1@0.95"] - exp) < 1e-9, (s, exp)
    print(f"PASS fewer preds: F1 = (n-30)/n = {exp:.4f} at every IoU")

    # 1c) MORE: one junk box per sample (never matches) -> recall 1, precision g/(g+1).
    # g mirrors their exact preprocessing for common_object_detection: category keys
    # become lowercase-letters-only (normalize_category_name), colliding keys merge
    # last-wins — e.g. numeric transcriptions all collapse into '' — so raw counts lie.
    def om_merge(d):
        return {"".join(c for c in k.lower() if c.isalpha()): v for k, v in d.items()}

    more, precs = [], []
    for e in tt:
        e = dict(e)
        preds = {k: list(v) for k, v in e["gt"].items()}
        preds["zzzjunkzzz"] = [[-10.0, -10.0, -1.0, -1.0]]   # non-gt key: counted, never matched
        e["extracted_predictions"] = preds
        more.append(e)
        g = sum(len(v) for v in om_merge(e["gt"]).values())
        precs.append(g / (g + 1))
    s = score(dump(more, tmp / "tt_more.jsonl"))
    p = sum(precs) / len(precs)
    exp = 2 * p * 1.0 / (p + 1.0)
    assert abs(s["F1@0.5"] - exp) < 1e-9, (s["F1@0.5"], exp)
    print(f"PASS extra preds: recall stays 1, F1 = {exp:.4f} from precision g/(g+1)")

    # 1d) SHIFTED: +4px x-shift -> F1 non-increasing over the IoU sweep, high-IoU collapse
    sh = [dict(e) for e in tt]
    for e in sh:
        e["extracted_predictions"] = {k: [[b[0] + 4, b[1], b[2] + 4, b[3]] for b in v]
                                      for k, v in e["gt"].items()}
    s = score(dump(sh, tmp / "tt_shift.jsonl"))
    curve = [s[f"F1@{t}"] for t in IOU_SWEEP]
    assert all(a >= b - 1e-12 for a, b in zip(curve, curve[1:])), curve
    assert s["F1@0.5"] > 0.8 and s["F1@0.95"] < s["F1@0.5"], s
    print(f"PASS shifted preds: monotone IoU curve, F1@0.5={s['F1@0.5']:.3f} -> F1@0.95={s['F1@0.95']:.3f}")

    # 2) GT-as-pred through converter + fastevaluate (COCO slice): exactly 1.0,
    #    and every GT box survives conversion (nothing silently dropped)
    coco = gt_as_answers(DATA / "_annotations/box_eval/COCO.jsonl", tmp / "coco.jsonl", n=5)
    s = score(tmp / "coco.jsonl", coco_gt=DATA / "coco/instances_val2017.json")
    assert abs(s["F1@0.5"] - 1.0) < 1e-9 and abs(s["F1@mean"] - 1.0) < 1e-9, s
    n_in = sum(len(b) for e in coco for b in e["gt"].values())
    n_out = sum(len(json.loads(l.split("\t")[1]))
                for l in open(str(tmp / "coco.jsonl") + ".fast_eval.tsv"))
    assert n_in == n_out, f"converter dropped boxes: {n_in} in, {n_out} out"
    print(f"PASS coco pipeline: GT-as-pred F1=1.0; converter kept all {n_in} boxes")

    # 3) sspro: GT-box center as prediction -> their scorer says accuracy exactly 1.0
    with open(tmp / "ss.jsonl", "w") as f:
        for e in load_dataset("sspro")[:20]:
            b = e["gt"]
            e["extracted_predictions"] = {"t": [[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]]}
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    s = score(tmp / "ss.jsonl")
    assert s["accuracy"] == 1.0 and s["n"] == 20, s
    print("PASS sspro: GT-center-as-pred accuracy=1.0 (their sspro_metric)")

    # 3b) sspro MISS: first 2 samples point outside their gt box -> accuracy exactly 18/20
    ss = load_jsonl(tmp / "ss.jsonl")
    for e in ss[:2]:
        e["extracted_predictions"] = {"t": [[-5.0, -5.0]]}
    s = score(dump(ss, tmp / "ss_miss.jsonl"))
    assert abs(s["accuracy"] - 18 / 20) < 1e-9, s
    print("PASS sspro misses: 2/20 off-target points -> accuracy exactly 0.9")

    # 4) pointing: a pixel taken from inside each GT mask -> F1 exactly 1.0
    from pycocotools import mask as cocomask
    with open(tmp / "pt.jsonl", "w") as f:
        for e in load_dataset("point/COCO")[:10]:
            preds = {}
            for cat, masks in e["gt"].items():
                pts = []
                for mi in masks:
                    m = cocomask.decode({"counts": mi["counts"].encode(), "size": mi["size"]})
                    ys, xs = m.nonzero()
                    pts.append([float(xs[0]), float(ys[0])])
                preds[cat] = pts
            e["extracted_predictions"] = preds
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    s = score(tmp / "pt.jsonl")
    assert abs(s["F1@0.5"] - 1.0) < 1e-9, s
    print("PASS pointing: in-mask points F1=1.0 (other_metric point path)")

    # 4b) FEWER points: first 3 of 10 samples predict nothing -> F1 exactly 7/10
    pt = load_jsonl(tmp / "pt.jsonl")
    few = [dict(e) for e in pt]
    for e in few[:3]:
        e["extracted_predictions"] = {}
    s = score(dump(few, tmp / "pt_few.jsonl"))
    assert abs(s["F1@0.5"] - 7 / 10) < 1e-9, s
    print("PASS fewer points: F1 = 7/10 exactly")

    # 4c) MORE points: one out-of-mask junk point per sample -> recall 1, precision m/(m+1)
    more, precs = [], []
    for e in pt:
        e = dict(e)
        preds = {k: [list(p) for p in v] for k, v in e["extracted_predictions"].items()}
        first = next(iter(preds))
        preds[first] = preds[first] + [[-5.0, -5.0]]   # bounds-checked miss in is_point_in_mask
        e["extracted_predictions"] = preds
        more.append(e)
        m = sum(len(v) for v in e["gt"].values())
        precs.append(m / (m + 1))
    s = score(dump(more, tmp / "pt_more.jsonl"))
    p = sum(precs) / len(precs)
    exp = 2 * p * 1.0 / (p + 1.0)
    assert abs(s["F1@0.5"] - exp) < 1e-9, (s["F1@0.5"], exp)
    print(f"PASS extra points: recall stays 1, F1 = {exp:.4f} from precision m/(m+1)")

    # 5) prompt regressions: rex strings verbatim, validated recipes exact
    assert T.build_prompt(T.TaskType.DETECTION, "rex-omni", categories=["a", "b"]) == \
        "Detect a, b. Output the bounding box coordinates in [x0, y0, x1, y1] format."
    assert T.build_prompt(T.TaskType.OCR_BOX, "rex-omni", categories=["word"], eval_protocol=True) == \
        "Can you detect all the word in this image in box format like [x0, y0, x1, y1] and then recognize them?"
    assert T.build_prompt(T.TaskType.REFERRING, "elliot", expression="x") == \
        'Locate "x". Return only [x1, y1, x2, y2] with coordinates in decimal values from 0 to 1.'
    assert T.build_prompt(T.TaskType.DETECTION, "qwen", categories=["a", "b"]) == \
        ('Locate every instance of the following categories: a, b. Output the coordinates in '
         'JSON format, with "bbox_2d" and "label" for each instance.')
    print("PASS prompts: rex templates verbatim, validated recipes exact")
    print("SELFTEST OK")


# ---------------------------------------------------------------- full runs
COCO_GT = {"COCO": DATA / "coco/instances_val2017.json",
           "LVIS": DATA / "coco/lvis_v1_val_with_filename2.json"}
ALL_DATASETS = (["COCO", "LVIS", "Dense200", "VisDrone", "DocLayNet", "M6Doc",
                 "TotalText", "SROIE", "IC15", "HierText",
                 "HumanRef", "RefCOCOg_val", "RefCOCOg_test"]                     # box_eval
                + ["point/COCO", "point/LVIS", "point/Dense200", "point/VisDrone",
                   "point/HumanRef", "point/RefCOCOg_val", "point/RefCOCOg_test"]  # point_eval
                + ["sspro"])                                                       # GUI grounding


def run(model, datasets, limit=0, out_dir=None, model_path=None, name=None, batch=BATCH):
    """One model over N box_eval datasets: inference (resumed if answers complete),
    then scoring. Writes <ds>.answer.jsonl, <ds>.scores.json and summary.json."""
    import time
    import models as M
    import report as R
    if model == "rex":
        adapter = M.make_rex(model_id=model_path or M.REX_ID)
    elif model == "qwen":
        adapter = M.make_qwen(path=model_path or M.QWEN_PATH, name=name or "qwen3.5-2b")
    elif model == "elliot":
        adapter = M.make_elliot(path=model_path or M.ELLIOT_PATH, name=name or "elliot-2b")
    elif model == "elliot-mcore":
        if not model_path:
            raise ValueError("elliot-mcore requires --model-path")
        adapter = M.make_elliot(path=M.ELLIOT_PATH, name=name or "elliot-loc",
                                mcore_path=model_path)
    else:
        raise ValueError(f"unknown model {model!r}")
    out_dir = Path(out_dir or ROOT / "outputs" / "runs" / adapter["name"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # offline on compute nodes; sync later from a login node:
    #   wandb sync outputs/wandb/wandb/offline-run-*
    import wandb
    os.environ.setdefault("WANDB_MODE", "offline")
    wb = wandb.init(project="loceval", name=adapter["name"], dir=str(ROOT / "outputs" / "wandb"),
                    config={"model": adapter["name"], "model_path": model_path,
                            "datasets": list(datasets), "limit": limit})

    summary = {"model": adapter["name"], "limit": limit, "datasets": {}}
    for ds in datasets:
        entries = load_dataset(ds)
        n = min(len(entries), limit) if limit else len(entries)
        safe = ds.replace("/", "_")
        ans = out_dir / f"{safe}.answer.jsonl"
        if ans.exists() and len(load_jsonl(ans)) == n:
            print(f"SKIP inference {ds}: {n} answers already there")
        else:
            t0 = time.time()
            _, n_empty = run_inference(adapter, entries, ans, indices=range(n), batch=batch)
            print(f"{ds}: {n} samples in {time.time() - t0:.0f}s, {n_empty} empty parses")
        try:
            s = score(ans, coco_gt=COCO_GT.get(ds))
            json.dump(s, open(out_dir / f"{safe}.scores.json", "w"), indent=1)
            wb.log({f"{ds}/{k}": v for k, v in s.items()})
            print(f"SCORE {ds}: " + "  ".join(f"{k}={v:.3f}" for k, v in s.items()))
        except Exception as e:                       # one broken dataset must not kill the job
            print(f"ERROR scoring {ds}: {e!r}")
            s = {"error": repr(e)}
        summary["datasets"][ds] = s
        rep = out_dir / "reports" / f"{safe}.{time.strftime('%Y%m%d-%H%M%S')}.html"
        try:
            R.report(ans, s, rep)
            print(f"REPORT -> {rep}")
        except Exception as e:
            print(f"ERROR report {ds}: {e!r}")
    json.dump(summary, open(out_dir / "summary.json", "w"), indent=1)
    wb.finish()
    print(f"DONE -> {out_dir}/summary.json")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    p = sub.add_parser("score")
    p.add_argument("answers")
    p.add_argument("--coco-gt", default=None)
    p = sub.add_parser("report")
    p.add_argument("answers")
    p.add_argument("--coco-gt", default=None)
    p.add_argument("--out", default=None)
    p = sub.add_parser("run")
    p.add_argument("--model", required=True, choices=["rex", "qwen", "elliot", "elliot-mcore"])
    p.add_argument("--datasets", nargs="*", default=["all"], help="dataset names, or 'all' (default)")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--model-path", default=None)
    p.add_argument("--name", default=None)
    p.add_argument("--batch", type=int, default=BATCH, help="samples per generate() call")
    a = ap.parse_args()
    if a.cmd == "selftest":
        selftest()
    elif a.cmd == "score":
        print(json.dumps(score(a.answers, coco_gt=a.coco_gt), indent=1))
    elif a.cmd == "report":
        import time
        import report as R
        s = score(a.answers, coco_gt=a.coco_gt)
        out = Path(a.out or str(a.answers).replace(".answer.jsonl", "")
                   + f".{time.strftime('%Y%m%d-%H%M%S')}.html")
        print(f"REPORT -> {R.report(a.answers, s, out)}")
    else:
        datasets = ALL_DATASETS if a.datasets in ([], ["all"]) else a.datasets
        run(a.model, datasets, limit=a.limit, out_dir=a.out_dir,
            model_path=a.model_path, name=a.name, batch=a.batch)
