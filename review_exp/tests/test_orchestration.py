"""Exercise experiment manifests and initialization choices without GPU training."""
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

removal = importlib.import_module("review_exp.01_sample_removal.run_experiment")
overlap = importlib.import_module("review_exp.02_data_overlap.run_experiment")
reuse = importlib.import_module("review_exp.02_data_overlap.reuse_full")
ROOT = Path(__file__).resolve().parents[1]


class Rows:  # 3 用法
    _fingerprint = "fixed_D"

    def __init__(self, labels):
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, key):
        return self.labels if key == "labels" else {"labels": self.labels[key]}

    def select(self, indices):  # 24 个用法 (24 个动态)
        return Rows([self.labels[i] for i in indices])

class FakeAccelerator:  # 2 用法
    is_main_process = True

    def __init__(self, **unused):
        pass

    def wait_for_everyone(self):  # 16 个用法 (16 个动态)
        pass


class ExperimentTests(unittest.TestCase):
    def bundle(self):  # 2 用法
        return {"dataset": {"train": Rows([0, 1] * 10), "validation": Rows([0, 1])},
                "labels": ["a", "b"]}

    def test_removal_all_independent_groups_start_from_F(self):
        calls = []

        def train(args, **options):
            calls.append((args.seed, args.checkpoint, len(options["dataset_override"]["train"])))
            return str(Path(args.output_dir) / "checkpoint.pt")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "loss.json").write_text(json.dumps({"num_train": 20, "losses": list(range(20))}))
            fake_accelerate = types.SimpleNamespace(Accelerator=FakeAccelerator)
            with patch.dict(sys.modules, {"accelerate": fake_accelerate}), \
                 patch.object(removal.runtime, "load_data", return_value=self.bundle()), \
                 patch.object(removal, "train_model", side_effect=train):
                removal.main(["--output_dir", str(root / "out"), "--loss_file", str(root / "loss.json"),
                              "--seeds", "43", "44"])
            models = json.loads((root / "out/models.json").read_text())
            self.assertEqual(len(models), 6)
            self.assertTrue(all(initial is None for _, initial, _ in calls))
            self.assertEqual(sum(row["expected"] == "independent" for row in models), 4)
            self.assertEqual(sum(row["expected"] == "reference" for row in models), 2)
            self.assertEqual([count for _, _, count in calls], [20, 20, 10, 10, 10, 10])

    def test_overlap_one_uses_same_raw_D_for_every_training_seed(self):
        bundle, seen = self.bundle(), []

        def train(args, **options):
            self.assertIs(options["dataset_override"]["train"], bundle["dataset"]["train"])
            self.assertIsNone(args.checkpoint)
            seen.append(args.seed)
            return str(Path(args.output_dir) / "checkpoint.pt")

        with tempfile.TemporaryDirectory() as directory:
            fake_accelerate = types.SimpleNamespace(Accelerator=FakeAccelerator)
            with patch.dict(sys.modules, {"accelerate": fake_accelerate}), \
                 patch.object(overlap.runtime, "load_data", return_value=bundle), \
                 patch.object(overlap, "train_model", side_effect=train):
                overlap.main(["--output_dir", directory, "--seeds", "43", "44"])
            models = json.loads((Path(directory) / "models.json").read_text())
            self.assertEqual(seen, [43, 44])
            self.assertTrue(all(row["actual_overlap"] == 1 for row in models))
            self.assertTrue(all(row["expected"] == "independent" for row in models))

    def test_reuse_full_certifies_origin_and_adds_no_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ckpt_dir = root / "full43"
            ckpt_dir.mkdir()
            (ckpt_dir / "checkpoint.pt").write_bytes(b"checkpoint")
            metadata = {"complete_model": True, "classifier_saved": True, "initial_checkpoint": None,
                        "initialization": "foundation_model_F", "dataset_train_fingerprint": "fixed_D",
                        "labels": ["a", "b"], "train_count": 2, "model_name_or_path": "/F/clip-model",
                        "seed": 43, "selection": {"original_indices": [0, 1]}}
            (ckpt_dir / "metadata.json").write_text(json.dumps(metadata))
            (root / "V.json").write_text(json.dumps(metadata))
            (root / "models.json").write_text(json.dumps([
                {"model_id": "full_seed43", "group": "full", "checkpoint": str(ckpt_dir / "checkpoint.pt"), "expected": "independent"}]))
            reuse.main(["--models", str(root / "models.json"), "--victim-metadata", str(root / "V.json"),
                        "--output-dir", str(root / "reuse")])
            note = json.loads((root / "reuse/reuse_note.json").read_text())
            self.assertEqual(note["additional_training_model_count"], 0)
            self.assertEqual(note["model_count"], 1)

class ShellHandoverTests(unittest.TestCase):
    def test_training_shell_preserves_space_paths_and_device_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stub = root / "accelerate"
            # Generated test fixture only; no project files are modified here.
            stub.write_text(f"#!{sys.executable}\nimport json,os,sys\n"
                            "with open(os.environ['CALL_LOG'],'a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')\n")
            stub.chmod(0o755)
            log = root / "calls.jsonl"
            env = dict(os.environ, PATH=str(root) + os.pathsep + os.environ["PATH"], CALL_LOG=str(log),
                       MODEL_NAME_OR_PATH="/server/model with spaces/clip-vit-base-patch32",
                       NUM_PROCESSES="2", CUDA_VISIBLE_DEVICES="2,3", BASELINE_DIR=str(root / "baseline"))
            for script in ("00_prepare_baseline", "01_sample_removal", "02_data_overlap"):
                command = ["bash", str(ROOT / script / "run.sh")]
                if script.startswith("00"):
                    command.append("all")
                subprocess.run(command, cwd=ROOT.parent, env=env, check=True, capture_output=True, text=True)
            calls = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertEqual(len(calls), 7)
            for call in calls:
                self.assertIn("--multi_gpu", call)
                self.assertEqual(call[call.index("--num_processes") + 1], "2")
                self.assertEqual(call[call.index("--model_name_or_path") + 1], env["MODEL_NAME_OR_PATH"])


if __name__ == "__main__":
    unittest.main()