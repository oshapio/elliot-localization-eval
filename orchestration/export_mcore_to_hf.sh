#!/bin/bash
# Export one Elliot MCore sidecar into a validated HF/vLLM checkpoint.
set -euo pipefail

output_dir=${1:?output directory}
mcore_checkpoint=${2:?MCore model_optim_rng.pt sidecar}
hf_template=${HF_TEMPLATE_PATH:?Set HF_TEMPLATE_PATH to the reviewed HF template}
train_repo=${TRAIN_REPO:?Set TRAIN_REPO to the elliot-vl-train checkout}
project=${TRAIN_PROJECT:-$train_repo/examples/elliovl_qwen35_siglip2_localization}
minimal=${MINIMAL_INFER_DIR:?Set MINIMAL_INFER_DIR to the validated minimal inference code}
export_dir="$output_dir/hf_export"

[[ -s "$mcore_checkpoint" ]] || { echo "Missing MCore sidecar: $mcore_checkpoint" >&2; exit 2; }
[[ -s "$hf_template/config.json" ]] || { echo "Invalid HF template: $hf_template" >&2; exit 2; }
[[ -s "$train_repo/tools_local/export_elliot_9k_to_hf.py" ]] || {
  echo "Missing exporter in TRAIN_REPO=$train_repo" >&2; exit 2;
}
[[ -d "$minimal" ]] || { echo "Missing MINIMAL_INFER_DIR: $minimal" >&2; exit 2; }

validate_export() {
  python3 - "$export_dir" "$mcore_checkpoint" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
checkpoint = sys.argv[2]
report = json.loads((root / "overlay_report.json").read_text())
assert report["checkpoint"] == checkpoint
assert report["loaded_count"] == 761
assert not report.get("missing_hf")
assert not report.get("missing_mcore")
assert not report.get("shape_mismatch")
assert (root / "hf_smoke_test.json").is_file()
index = json.loads((root / "model.safetensors.index.json").read_text())
for name in set(index["weight_map"].values()):
    assert (root / name).is_file(), name
print("EXPORT_IDENTITY_VERIFIED", checkpoint)
PY
}

if [[ -s "$export_dir/overlay_report.json" && -s "$export_dir/hf_smoke_test.json" ]]; then
  if validate_export; then
    exit 0
  fi
fi

scratch_root=${CINECA_SCRATCH:?CINECA_SCRATCH must point at durable scratch}
storage="$scratch_root/elliot-localization/cache/eval-hf-exports/${SLURM_JOB_ID:-manual-$$}"
template_hash=$(sha256sum "$hf_template/config.json" | cut -c1-16)
compat="$scratch_root/elliot-localization/cache/export-compat/${template_hash}-flat-imports-${SLURM_JOB_ID:-$$}"
mkdir -p "$output_dir" "$storage"
if [[ -e "$export_dir" || -L "$export_dir" ]]; then
  mv "$export_dir" "$export_dir.incomplete.${SLURM_JOB_ID:-$$}"
fi
ln -s "$storage" "$export_dir"

(
  # The exporter needs the training environment.  Keep it inside this
  # subshell so AiAK/VeOmni variables cannot leak into the vLLM process.
  source "$project/train/env_leonardo.sh"
  export HF_HOME="$scratch_root/.caches/huggingface"
  export TRITON_CACHE_DIR="$scratch_root/.caches/triton/export-${SLURM_JOB_ID:-$$}"
  mkdir -p "$TRITON_CACHE_DIR"

  building="$compat.building.${SLURM_JOB_ID:-$$}"
  rm -rf "$building"
  mkdir -p "$building"
  cp -as "$hf_template/." "$building/"
  for source in "$hf_template"/*.py; do
    name=$(basename "$source")
    rm -f "$building/$name"
    cp "$source" "$building/$name"
    sed -i -E 's/^from \.([A-Za-z0-9_]+)/from \1/' "$building/$name"
  done
  touch "$building/.complete"
  rm -rf "$compat"
  mv "$building" "$compat"

  python3 "$train_repo/tools_local/export_elliot_9k_to_hf.py" \
    --minimal-infer-dir "$minimal" \
    --base-model-path "$compat" \
    --code-model-path "$compat" \
    --export-code-path "$hf_template" \
    --mcore-checkpoint "$mcore_checkpoint" \
    --output-dir "$export_dir" \
    --max-shard-size 2GB \
    --smoke-prompt 'Locate the region matching the description: car.' \
    --max-new-tokens 16
)

validate_export
