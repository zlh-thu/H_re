"""Fast regression checks for scientific validity, independent of GPU weights."""
import importlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from review_exp.common import collect_outputs as collect
from review_exp.common import evaluate_checkpoints as batch
from review_exp.common import verifier

try:
    import numpy as np
except ImportError:
    np = None

removal = importlib.import_module("review_exp.01_sample_removal.selection")
overlap = importlib.import_module("review_exp.02_data_overlap.selection")
baseline = importlib.import_module("review_exp.00_prepare_baseline.prepare_baseline")


class SelectionTests(unittest.TestCase):
    def test_low_loss_drops_low_not_high(self):
        labels = [0, 0, 1, 1, 2, 2]
        keep = removal.low_loss_keep(labels, [0.2, 0.1, 0.3, 0.6, 0.5, 0.4], 0.5)
        self.assertEqual(keep, [3, 4, 5])
        random = removal.random_keep_matching_classes(labels, keep, 2026)
        self.assertEqual(removal.class_counts(labels, keep), removal.class_counts(labels, random))
        self.assertEqual(random, removal.random_keep_matching_classes(labels, keep, 2026))

    def test_incomplete_or_nonfinite_loss_rejected(self):
        with self.assertRaises(ValueError):
            removal.validate_loss_mapping({"0": 0.2}, 2)
        with self.assertRaises(ValueError):
            removal.low_loss_keep([0, 1], [0.2, float("nan")], 0.5)

    def test_original_poison_selects_then_excludes_target(self):
        selected = baseline.protocol_indices([0, 1, 2, 1, 0, 2], [0, 2, 1, 3, 4, 5], 0.5, 0.5, 0)
        self.assertEqual(selected["candidate_indices"], [0, 2, 1])
        self.assertEqual(selected["query_indices"], [1, 2])
        self.assertEqual(selected["benign_retained_indices"], [3, 4, 5])
        self.assertEqual(len(selected["poisoned_training_order"]), 6)

    def test_identical_D_is_identical_for_all_training_seeds(self):
        labels = [0, 1] * 10
        plan = overlap.select_overlap(labels, [], 1.0)
        self.assertEqual(plan["base_indices"], list(range(20)))
        self.assertEqual(plan["extra_indices"], [])
        self.assertEqual(plan["actual_overlap"], 1.0)

    def test_partial_overlap_needs_real_sufficient_pool(self):
        with self.assertRaises(ValueError):
            overlap.select_overlap([0, 1] * 10, [], 0.5)
        plan = overlap.select_overlap([0, 1] * 10, [0, 1] * 10, 0.5)
        self.assertEqual(len(plan["base_indices"]), 10)
        self.assertEqual(len(plan["extra_indices"]), 10)
        self.assertEqual(plan["total_count"], 20)
        self.assertEqual(plan["actual_overlap"], 0.5)

    def test_pseudo_label_pool_is_not_independent(self):
        value = {"source": "example", "label_source": "victim_pseudo_labels", "in_distribution": True,
                 "victim_disjoint": True, "not_test_data": True, "class_names": ["a", "b"]}
        with self.assertRaises(ValueError):
            overlap.validate_pool_provenance(value, ["a", "b"])

class QueryTests(unittest.TestCase):
    def test_invalid_indices(self):
        for value in ([], [0, 0], [True], [-1], [4], [1.0]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                collect.validate_indices(value, size=4)
        self.assertEqual(collect.validate_indices(values=[3, 0, 1], size=4), [3, 0, 1])

    def test_checkpoint_identity_survives_rename_and_directory_form(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            (source / "checkpoint.pt").write_bytes(b"same checkpoint bytes")
            (source / "renamed.pt").write_bytes(b"same checkpoint bytes")
            self.assertEqual(collect.checkpoint_digest(source), collect.checkpoint_digest(source / "renamed.pt"))

    def test_model_manifest_rejects_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "models.json"
            row = {"model_id": "seed43", "checkpoint": "seed43/checkpoint.pt", "expected": "independent"}
            path.write_text(json.dumps([row, row]))
            with self.assertRaises(ValueError):
                batch.load_models(path)

    def test_batch_extra_args_cannot_override_checkpoint_alias(self):
        args = SimpleNamespace(num_processes=1, mixed_precision="bf16", port=29620,
                               model_name_or_path="/F/clip-model", dataset_name="./dataset/cifar10.py",
                               query_manifest="queries.json", query_batch_size=32, verifier="frozenC",
                               poison="P.npz", device="cpu", collector_extra_args="--init-checkpoint other.pt")
        entry = {"model_id": "seed43", "checkpoint": "correct.pt", "expected": "independent"}
        with self.assertRaisesRegex(ValueError, "override"):
            batch.build_commands(args, entry, Path("results"))

@unittest.skipIf(np is None, "NumPy is required for output-contract tests")
class OutputContractTests(unittest.TestCase):
    def make_output(self, path, *, ids=None, classes=None, dtype=None):
        np.savez(path, logits=np.zeros((20, 10), dtype=dtype or np.float32), labels=np.arange(20) % 10,
                 query_ids=np.asarray(ids or [f"query{i}" for i in range(20)]),
                 class_names=np.asarray(classes or [f"class{i}" for i in range(10)]))
        return verifier.load_outputs(path)

    def test_query_permutation_and_class_permutation_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            base = self.make_output(Path(directory) / "base.npz")
            shuffled = self.make_output(Path(directory) / "shuffled.npz", ids=[f"query{i}" for i in reversed(range(20))])
            renamed = self.make_output(Path(directory) / "renamed.npz", classes=[f"class{i}" for i in reversed(range(10))])
            for value in (shuffled, renamed):
                with self.assertRaises(ValueError):
                    verifier.assert_aligned(base, value)

    def test_float64_and_duplicate_ids_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                self.make_output(Path(directory) / "double.npz", dtype=np.float64)
            with self.assertRaises(ValueError):
                self.make_output(Path(directory) / "duplicate.npz", ids=["same"] * 20)

    def test_legacy_C_overlap_rejected_before_loading_torch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_output(root / "V.npz")
            self.make_output(root / "P.npz")
            (root / "used.json").write_text('["query0"]')
            (root / "holdout.json").write_text('["query0", "query1"]')
            args = verifier.build_parser().parse_args([
                "import-legacy", "--checkpoint", "unused.pt", "--preprocessing-json", "unused.json",
                "--training-query-ids", str(root / "used.json"), "--holdout-query-ids", str(root / "holdout.json"),
                "--reference-query", str(root / "V.npz"), "--poison", str(root / "P.npz"),
                "--output", str(root / "C"), "--sample-num", "2"])
            with self.assertRaisesRegex(ValueError, "overlap"):
                verifier.import_legacy(args)

    def test_legacy_mixed_identity_namespaces_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ids = [f"sha256:{i:064x}" for i in range(20)]
            self.make_output(root / "V.npz", ids=ids)
            self.make_output(root / "P.npz", ids=ids)
            (root / "used.json").write_text('["cifar10:train:0"]')
            (root / "holdout.json").write_text(json.dumps(ids[:2]))
            args = verifier.build_parser().parse_args([
                "import-legacy", "--checkpoint", "unused.pt", "--preprocessing-json", "unused.json",
                "--training-query-ids", str(root / "used.json"), "--holdout-query-ids", str(root / "holdout.json"),
                "--reference-query", str(root / "V.npz"), "--poison", str(root / "P.npz"),
                "--output", str(root / "C"), "--sample-num", "2"])
            with self.assertRaisesRegex(ValueError, "sha256"):
                verifier.import_legacy(args)

    def test_frozen_scaler_transforms_without_refitting(self):
        values = np.asarray([[3., 4.], [0., 0.]], dtype=np.float32)
        transformed = verifier.transform(values, config={"norm": True, "scale": True},
                                         scaler={"center": np.asarray([0.1, 0.2]), "scale": np.asarray([0.5, 0.2])})
        np.testing.assert_allclose(transformed, [[1., 3.], [-0.2, -1.]], atol=1e-6)

class ModelCountTests(unittest.TestCase):
    def row(self, model_id, expected, stolen, digest):  # 5 用法
        return {"verifier_sha256": "frozenC", "protocol": {"alpha": 0.01},
                "suspect_metadata": {"checkpoint_sha256": digest}, "suspect_path": model_id + ".npz",
                "model_level": {"model_id": model_id, "expected": expected,
                                "decision": "stolen" if stolen else "independent", "model_count": 1,
                                "false_positive": expected == "independent" and stolen}}

    def test_reference_and_query_repeats_do_not_count_as_models(self):
        rows = [self.row("seed43", "independent", True, "ckpt43"),
                self.row("seed44", "independent", False, "ckpt44"),
                self.row("B", "reference", False, "B")]
        with patch.object(verifier, "read_json", side_effect=rows):
            report = verifier.summarize_reports(["a", "b", "c"])
        self.assertEqual(report["independent_model_count"], 2)
        self.assertEqual(report["false_positive_model_count"], 1)
        self.assertEqual(report["model_level_false_positive_rate"], 0.5)
        self.assertEqual(report["excluded_reference_or_unknown_count"], 1)

    def test_same_checkpoint_under_two_model_names_is_rejected(self):
        rows = [self.row("seed43","independent",True,"same"),
                self.row("renamed43","independent",True,"same")]
        with patch.object(verifier, "read_json", side_effect=rows), self.assertRaises(ValueError):
            verifier.summarize_reports(["a", "b"])

if __name__ == "__main__":
    unittest.main()