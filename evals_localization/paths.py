"""All cluster-specific paths in ONE place. Each resource lists candidates per
cluster; the first existing one wins, otherwise a loud error naming everything
tried. Porting to a new cluster = add its paths to these lists."""
import os
import sys
import types
from pathlib import Path

def first_existing(name, candidates):
    for c in candidates:
        if Path(c).exists():
            return Path(c)
    raise FileNotFoundError(f"{name}: none of these exist:\n  " + "\n  ".join(str(c) for c in candidates))


def optional_existing(candidates):
    """Resolve an optional model without blocking unrelated adapters."""
    for candidate in candidates:
        if Path(candidate).exists():
            return Path(candidate)
    return Path(candidates[0])


ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / "vendor" / "rex_omni"


DATA = first_existing("EvalData", [
    *([os.environ["EVAL_DATA"]] if os.environ.get("EVAL_DATA") else []),
    ROOT / "EvalData",   # symlink/dir next to the code (preferred; re-point per cluster)
    "/mnt/lustre/work/oh/owl661/elliot-mllm-inspect/elliot-locate-eval/EvalData",  # galvani
    "/e/scratch/token-compression/uselis1/nas/elliot-locate-eval/EvalData",        # jupiter
    "/leonardo_scratch/large/userexternal/auselis0/elliot-locate-eval/EvalData",   # leonardo
    # "/leonardo_scratch/large/userexternal/.../EvalData",                         # leonardo: add when staged
])

ELLIOT_PATH = optional_existing([
    *([os.environ["ELLIOT_MODEL_PATH"]] if os.environ.get("ELLIOT_MODEL_PATH") else []),
    "/mnt/lustre/work/oh/owl661/elliot-mllm-inspect/elliot-ov2-sft-9k",            # galvani
    "/e/scratch/token-compression/uselis1/nas/checkpoints/elliot-ov2-sft-9k",      # jupiter
    "/leonardo_scratch/large/userexternal/auselis0/checkpoints/elliot-ov2-sft-9k-coord1000",
    # "/leonardo_work/EUHPC_E07_125/ElliotVL/.../elliot-ov2-sft-9k",               # leonardo
])

QWEN_PATH = optional_existing([
    "/mnt/lustre/work/oh/owl661/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B"
    "/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc",                          # galvani
    # "/leonardo_scratch/.../Qwen3.5-2B",                                          # leonardo
])

REX_ID = "IDEA-Research/Rex-Omni"   # HF id -> resolves via HF_HOME cache; pre-stage on offline clusters

# make our modules + the vendored rex code importable regardless of cwd,
# and satisfy the vendored file's top-level `from vllm import ...` (unused here)
sys.path[:0] = [str(ROOT), str(VENDOR), str(VENDOR / "metrics"), str(VENDOR / "utils"),
                str(ROOT / "vendor" / "embodied")]
sys.modules.setdefault("vllm", types.SimpleNamespace(LLM=None, SamplingParams=None))
