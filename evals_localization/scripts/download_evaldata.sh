#!/usr/bin/env bash
# Download + extract all eval data for LocateAnything-style grounding evals:
#   1. Mountchicken/Rex-Omni-EvalData (annotations + ~11.6 GB image tarballs)
#   2. likaixin/ScreenSpot-Pro (GUI grounding)
#   3. Generate ScreenSpot-Pro/converted_box.jsonl (upstream README link is a dead TODO)
#   4. Fallback-fetch coco/instances_val2017.json if the coco tarball lacks it
# Idempotent: re-running skips completed downloads (hf resume) and extracted tarballs.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVALDATA="$ROOT/EvalData"
VENV_BIN=/mnt/lustre/work/oh/owl661/ocl-llava/qwenfast-venv310/bin
export HF_HOME=/mnt/lustre/work/oh/owl661/elliot-mllm-inspect/cache/huggingface

mkdir -p "$EVALDATA"

echo "[1/5] Downloading Rex-Omni-EvalData (~12 GB, resumable)..."
"$VENV_BIN/hf" download Mountchicken/Rex-Omni-EvalData \
    --repo-type dataset --local-dir "$EVALDATA" || exit 1

echo "[2/5] Extracting image tarballs..."
cd "$EVALDATA"
for f in *.tar.gz; do
    stamp=".extracted_${f%.tar.gz}"
    if [[ -f "$stamp" ]]; then
        echo "  skip $f (already extracted)"
        continue
    fi
    echo "  extracting $f ..."
    if tar -xzf "$f"; then
        touch "$stamp"
    else
        echo "  ERROR extracting $f" >&2
        exit 1
    fi
done

# LVIS gt json ships in a typo'd dir on HF; put it where eval_lvis.sh expects it
if [[ -f missing_annotaitons/lvis_v1_val_with_filename2.json && ! -f coco/lvis_v1_val_with_filename2.json ]]; then
    mkdir -p coco
    cp missing_annotaitons/lvis_v1_val_with_filename2.json coco/
    echo "  placed coco/lvis_v1_val_with_filename2.json"
fi

echo "[3/5] Checking for coco/instances_val2017.json..."
if [[ ! -f coco/instances_val2017.json ]]; then
    echo "  not in tarball -> fetching official COCO annotations zip (241 MB)..."
    curl -fL --retry 3 -o /tmp/coco_ann2017.zip \
        http://images.cocodataset.org/annotations/annotations_trainval2017.zip || exit 1
    "$VENV_BIN/python" - <<'EOF' || exit 1
import zipfile, shutil, os
os.makedirs("coco", exist_ok=True)
with zipfile.ZipFile("/tmp/coco_ann2017.zip") as z:
    with z.open("annotations/instances_val2017.json") as src, \
         open("coco/instances_val2017.json", "wb") as dst:
        shutil.copyfileobj(src, dst)
print("  extracted coco/instances_val2017.json")
EOF
    rm -f /tmp/coco_ann2017.zip
else
    echo "  already present"
fi

echo "[4/5] Downloading ScreenSpot-Pro..."
"$VENV_BIN/hf" download likaixin/ScreenSpot-Pro \
    --repo-type dataset --local-dir "$EVALDATA/ScreenSpot-Pro" || exit 1

echo "[5/5] Generating ScreenSpot-Pro converted_box.jsonl..."
"$VENV_BIN/python" "$ROOT/scripts/make_sspro_converted_box.py" \
    --sspro_dir "$EVALDATA/ScreenSpot-Pro" \
    --out "$EVALDATA/ScreenSpot-Pro/converted_box.jsonl" || exit 1

echo ""
echo "================ VERIFICATION ================"
for d in box_eval point_eval keypoint_eval visual_prompt_eval; do
    n=$(ls "$EVALDATA/_annotations/$d" 2>/dev/null | wc -l)
    echo "  _annotations/$d: $n jsonl files"
done
for f in coco/instances_val2017.json coco/lvis_v1_val_with_filename2.json ScreenSpot-Pro/converted_box.jsonl; do
    if [[ -f "$EVALDATA/$f" ]]; then echo "  OK      $f"; else echo "  MISSING $f"; fi
done
# spot-check one extracted image dir per tarball
for d in coco dense200 visdrone humanref hiertext totaltext sroie ic15 doclaynet m6doc ap10k; do
    if [[ -d "$EVALDATA/$d" ]]; then
        echo "  OK      $d/ ($(find "$EVALDATA/$d" -type f | head -100000 | wc -l)+ files)"
    else
        echo "  MISSING $d/ (check tarball name vs extracted dir)"
    fi
done
du -sh "$EVALDATA"
echo "Done."
