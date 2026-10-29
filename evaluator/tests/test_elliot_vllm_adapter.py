"""Exercise coordinate decoding and request/result routing without a GPU."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "integrations/elliot_vllm"))
from adapters.elliot_vllm import create_adapter


class AdapterTest(unittest.TestCase):
    def test_launcher_uses_selected_checkout_over_stale_plugin(self):
        launcher = Path(__file__).resolve().parents[1] / "slurm/vllm_benchmark.sbatch"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plugin = root / "integrations/elliot_vllm/elliot_vllm"
            plugin.mkdir(parents=True)
            shutil.copyfile(launcher.parents[2] / "integrations/elliot_vllm/elliot_vllm/__init__.py",
                            plugin / "__init__.py")
            scorer = root / "evaluator/scoring"
            scorer.mkdir(parents=True)
            (scorer / "__init__.py").touch()
            (scorer / "rex.py").write_text("def load_scorer(): pass\n")
            stale = root / "stale/elliot_vllm"
            stale.mkdir(parents=True)
            (stale / "__init__.py").write_text("raise RuntimeError('stale plugin loaded')\n")
            binaries = root / "bin"
            binaries.mkdir()
            srun = binaries / "srun"
            srun.write_text("#!/bin/sh\nexit 77\n")
            srun.chmod(0o755)
            result = subprocess.run(["bash", str(launcher)], text=True, capture_output=True,
                                    env={**os.environ, "EVAL_REPO": str(root),
                                         "EVAL_PYTHON": sys.executable, "CONFIG": "unused.json",
                                         "OUTPUT_DIR": str(root / "output"),
                                         "SLURM_CPUS_PER_TASK": "1", "SLURM_NTASKS": "1",
                                         "PYTHONPATH": str(stale.parent),
                                         "PATH": str(binaries) + os.pathsep + os.environ["PATH"]})
            self.assertEqual(result.returncode, 77, result.stdout + result.stderr)
            self.assertIn(f"Elliot plugin: {plugin / '__init__.py'}", result.stdout)

    def test_current_configs_select_the_verified_release(self):
        configs = Path(__file__).resolve().parents[1] / "configs/current"
        paths = sorted(configs.glob("*.json"))
        self.assertEqual([path.name for path in paths],
                         ["elliot_vllm.json", "elliot_vllm_tp2.json"])
        values = [json.loads(path.read_text()) for path in paths]
        self.assertEqual({value["model_path"] for value in values}, {
            "/leonardo_scratch/large/userexternal/auselis0/elliot-localization/"
            "cache/eval-hf-exports/58552791"
        })
        self.assertEqual([value["tensor_parallel_size"] for value in values], [1, 2])
        self.assertTrue(all(value["max_pixels"] == 4194304 for value in values))
        self.assertTrue(all("cardinality-repair-70-30-local500" in value["name"]
                            for value in values))

    def test_launcher_preflights_scorer_before_gpu_inference(self):
        launcher = Path(__file__).resolve().parents[1] / "slurm/vllm_benchmark.sbatch"
        script = launcher.read_text()
        self.assertIn("from scoring.rex import load_scorer", script)
        self.assertLess(script.index("load_scorer()"), script.index('srun --ntasks='))

    def test_slurm_shards_do_not_share_device_zero(self):
        script = Path(__file__).resolve().parents[1] / "slurm/run_shard.sh"
        self.assertNotIn("RETRY_MAX_NEW_TOKENS", script.read_text())
        self.assertNotIn("--retry-max-new-tokens", script.read_text())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evaluator = root / "eval"
            evaluator.mkdir()
            (evaluator / "run.py").touch()
            manifest = root / "manifest.jsonl"
            manifest.touch()
            config = root / "config.json"
            config.write_text("{}\n")
            output = root / "output"
            python = root / "python3"
            python.write_text(
                '#!/bin/bash\n'
                f'if [[ "$1" == "-" ]]; then exec {sys.executable} "$@"; fi\n'
                'printf "%s\\n" "$CUDA_VISIBLE_DEVICES" "$@"\n'
            )
            python.chmod(0o755)
            for visible, rank, count, expected in [
                ("0,1,2,3", "2", "1", "2"), ("GPU-uuid", "3", "1", "GPU-uuid"),
                ("0,1,2,3", "0", "2", "0,1"), ("0,1,2,3", "1", "2", "2,3"),
                ("GPU-a,GPU-b", "1", "2", "GPU-a,GPU-b"),
            ]:
                with self.subTest(visible=visible):
                    env = dict(os.environ, PATH=tmp + ":" + os.defpath,
                               ADAPTER="baseline_vllm", BATCH_SIZE="8", GPUS_PER_MODEL=count,
                               CUDA_VISIBLE_DEVICES=visible, SLURM_LOCALID=rank,
                               SLURM_PROCID=rank, SLURM_NTASKS="4")
                    result = subprocess.run(
                        ["bash", str(script), str(evaluator), str(manifest), str(config),
                         str(output), "0"],
                        env=env, capture_output=True, text=True, check=True,
                    ).stdout.splitlines()
                    self.assertEqual(result[0], expected)
                    self.assertEqual(result[result.index("--batch-size") + 1], "8")

    def test_run_shard_fails_fast_with_named_input_errors(self):
        script = Path(__file__).resolve().parents[1] / "slurm/run_shard.sh"
        missing_args = subprocess.run(
            ["bash", str(script), "only-one-argument"], capture_output=True, text=True,
        )
        self.assertEqual(missing_args.returncode, 64)
        self.assertIn("expected 5 positional arguments", missing_args.stderr)
        self.assertIn("CONFIG.json", missing_args.stderr)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evaluator = root / "eval"
            evaluator.mkdir()
            (evaluator / "run.py").touch()
            manifest = root / "manifest.jsonl"
            manifest.touch()
            config = root / "config.json"
            config.write_text("{}\n")
            args = ["bash", str(script), str(evaluator), str(manifest), str(config),
                    str(root / "output"), "0"]
            env = dict(os.environ, SLURM_PROCID="0", SLURM_NTASKS="1", SLURM_LOCALID="0")

            missing_adapter = subprocess.run(args, env=env, capture_output=True, text=True)
            self.assertEqual(missing_adapter.returncode, 64)
            self.assertIn("ADAPTER is missing", missing_adapter.stderr)

            env["ADAPTER"] = "elliot_vllm"
            args[4] = str(root / "missing-config.json")
            missing_config = subprocess.run(args, env=env, capture_output=True, text=True)
            self.assertEqual(missing_config.returncode, 64)
            self.assertIn("adapter config is not readable", missing_config.stderr)

            malformed = root / "malformed.json"
            malformed.write_text("not JSON\n")
            args[4] = str(malformed)
            malformed_config = subprocess.run(args, env=env, capture_output=True, text=True)
            self.assertEqual(malformed_config.returncode, 64)
            self.assertIn("adapter config must be a readable JSON object",
                          malformed_config.stderr)

    def test_navit_rejects_fixed_resolution_prompt_tokens(self):
        class Engine:
            count = 512
            def __init__(self, **kwargs):
                pass
            def get_tokenizer(self):
                return SimpleNamespace(convert_tokens_to_ids=lambda token: 999)
            def generate(self, requests, params, use_tqdm):
                return [SimpleNamespace(prompt_token_ids=[999] * self.count,
                    outputs=[SimpleNamespace(text="<ref>person</ref><box><0><0><1000><1000></box>",
                                             token_ids=[1], finish_reason="stop")])]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "image.png"
            Image.new("RGB", (1024, 512)).save(path)
            with patch.dict(sys.modules, {
                "vllm": SimpleNamespace(LLM=Engine, SamplingParams=lambda **kwargs: kwargs),
                "qwen_vl_utils.vision_process": SimpleNamespace(smart_resize=lambda *a, **kw: (512, 1024)),
            }):
                adapter = create_adapter(dict(model_path=tmp, max_model_len=16384, max_pixels=4194304))
                record = {"image_file": str(path), "benchmark_protocol": "box"}
                result = adapter["predict_batch"]([record], ["person"], 4096)[0]
                self.assertEqual(result["image_tokens"], 512)
                self.assertEqual(result["processed_image_size"], [1024, 512])
                Engine.count = 256
                with self.assertRaisesRegex(ValueError, "Image token count"):
                    adapter["predict_batch"]([record], ["person"], 4096)

    def test_batch_preserves_image_sizes_special_tokens_and_point_protocol(self):
        calls = []

        class Engine:
            def __init__(self, **kwargs):
                pass

            def generate(self, requests, params, use_tqdm):
                calls.append((requests, params))
                return [SimpleNamespace(prompt_token_ids=[1], outputs=[SimpleNamespace(
                    text=text, token_ids=[2, 3], finish_reason=finish,
                )]) for text, finish in [
                    ("<ref>person</ref><box><100><200><300><400></box>", "stop"),
                    ("<ref>person</ref><box><500><500></box>", "length"),
                ]]

        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp) / 'a.png', Path(tmp) / 'b.png']
            for path, size in zip(paths, [(100, 200), (400, 100)]):
                Image.new('RGB', size).save(path)
            with patch.dict(sys.modules, {'vllm': SimpleNamespace(
                LLM=Engine, SamplingParams=lambda **kwargs: kwargs,
            )}):
                adapter = create_adapter({'model_path': tmp, 'max_model_len': 4096})
                self.assertEqual(adapter['output_format'], 'ref_box')
                self.assertNotIn('prepare_prompt', adapter)
                results = adapter['predict_batch']([
                    {'image_file': str(paths[0]), 'benchmark_protocol': 'box'},
                    {'image_file': str(paths[1]), 'benchmark_protocol': 'point'},
                ], ['first query', 'second query'], 128)
        self.assertEqual(results[0]['predictions'], {'person': [[10., 40., 30., 80.]]})
        self.assertEqual(results[1]['predictions'], {'person': [[200., 50.]]})
        self.assertEqual([r['hit_token_cap'] for r in results], [False, True])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1]['max_tokens'], 128)
        self.assertFalse(calls[0][1]['skip_special_tokens'])
        self.assertIn('first query', calls[0][0][0]['prompt'])
        self.assertIn('second query', calls[0][0][1]['prompt'])

    def test_opt_in_logprobs_save_the_chosen_token_at_every_position(self):
        calls = []

        class Engine:
            def __init__(self, **kwargs):
                pass

            def generate(self, requests, params, use_tqdm):
                calls.append(params)
                return [SimpleNamespace(prompt_token_ids=[1], outputs=[SimpleNamespace(
                    text='<ref>car</ref><box><10><20><30><40></box>',
                    token_ids=[7, 8], finish_reason='stop',
                    logprobs=[
                        {7: SimpleNamespace(logprob=-0.25)},
                        {'8': SimpleNamespace(logprob=-0.75)},
                    ],
                )])]

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'image.png'
            Image.new('RGB', (100, 100)).save(path)
            with patch.dict(sys.modules, {'vllm': SimpleNamespace(
                LLM=Engine, SamplingParams=lambda **kwargs: kwargs,
            )}):
                adapter = create_adapter({
                    'model_path': tmp, 'max_model_len': 4096, 'return_logprobs': True,
                })
                result = adapter['predict_batch']([{
                    'image_file': str(path), 'benchmark_protocol': 'box',
                }], ['car'], 64)[0]

        self.assertEqual(calls[0]['logprobs'], 1)
        self.assertEqual(result['token_logprobs'], [-0.25, -0.75])
        self.assertEqual(result['cumulative_logprob'], -1.0)
        self.assertEqual(result['mean_logprob'], -0.5)

    def test_tar_image_reference_is_read_without_materializing_an_image_copy(self):
        calls = []

        class Engine:
            def __init__(self, **kwargs):
                pass

            def generate(self, requests, params, use_tqdm):
                calls.append(requests[0]['multi_modal_data']['image'].size)
                return [SimpleNamespace(prompt_token_ids=[1], outputs=[SimpleNamespace(
                    text='<ref>car</ref><box><10><20><30><40></box>',
                    token_ids=[7], finish_reason='stop',
                )])]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / 'source.png'
            Image.new('RGB', (37, 23)).save(image)
            payload = image.read_bytes()
            archive = root / 'images.tar'
            prefix = b'not-a-real-tar-header'
            archive.write_bytes(prefix + payload + b'trailer')
            with patch.dict(sys.modules, {'vllm': SimpleNamespace(
                LLM=Engine, SamplingParams=lambda **kwargs: kwargs,
            )}):
                adapter = create_adapter({'model_path': tmp, 'max_model_len': 4096})
                adapter['predict_batch']([{
                    'image_reference': {'k': 'tar', 'p': 'images.tar',
                                        'o': len(prefix), 'n': len(payload)},
                    'media_root': tmp, 'benchmark_protocol': 'box',
                }], ['car'], 64)

        self.assertEqual(calls, [(37, 23)])


if __name__ == '__main__':
    unittest.main()
