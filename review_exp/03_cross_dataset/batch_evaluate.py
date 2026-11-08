#!/usr/bin/env python3
"""Run the two supplementary cross-dataset B-as-S checks with frozen verifiers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from review_exp.common import verifier


def resolve(base, value):  # 43 个用法 (40 个动态)
    path = Path(value).expanduser()
    return str(path.resolve() if path.is_absolute() else (base / path).resolve())


def validate_manifest(document : (get) , base):  # 1 个用法
    directions = document.get("directions", [])
    if len(directions) != 2:
        raise ValueError("exactly two reciprocal directions are required")
    names, pairs = set(), []
    checked = []
    for entry in directions:
        name = entry.get("name", "")
        if not name or Path(name).name != name or name in {".", ".."} or name in names:
            raise ValueError("direction names must be unique safe directory names")
        names.add(name)
        for key in ("verifier", "poison", "suspect", "model_id", "verification_domain",
                    "suspect_training_domain", "class_names", "independence"):
            if key not in entry:
                raise ValueError(f"{name}: missing field {key}")
        evidence = entry["independence"]
        if evidence.get("uses_victim_outputs") is not False:
            raise ValueError(f"{name}: victim-generated labels/distillation are not independent negatives")
        if evidence.get("trained_independently") is not True or not evidence.get("label_source"):
            raise ValueError(f"{name}: document independent training and the actual label source")
        if evidence.get("label_source_role") not in {"ground_truth", "external_independent"}:
            raise ValueError(f"{name}: label_source_role must be ground_truth or external_independent; "
                             "a protected V teacher does not qualify")
        if not evidence.get("training_checkpoint") or any("REPLACE" in str(evidence.get(key, ""))
                                                          for key in ("label_source", "training_checkpoint")):
            raise ValueError(f"{name}: replace placeholder label/training provenance with actual evidence")
        verifier_path = resolve(base, entry["verifier"])
        manifest = verifier.read_json(Path(verifier_path) / "verifier.json")
        suspect = verifier.load_outputs(resolve(base, entry["suspect"]))
        poison = verifier.load_outputs(resolve(base, entry["poison"]))
        verifier.assert_aligned(suspect, poison)
        if not entry["class_names"] or entry["class_names"] != manifest["class_names"]:
            raise ValueError(f"{name}: manifest class_names must exactly equal the verifier's class semantics/order")
        if suspect["class_names"].tolist() != entry["class_names"]:
            raise ValueError(f"{name}: suspect class space differs; native CIFAR-10/TinyImageNet cannot be subtracted")
        if suspect["query_ids"].tolist() != manifest["query_ids"] or poison["query_ids"].tolist() != manifest["query_ids"]:
            raise ValueError(f"{name}: query IDs/order differ from this direction's verifier domain")
        if suspect["labels"].tolist() != manifest["labels"]:
            raise ValueError(f"{name}: query labels differ from the fixed verifier")
        if verifier.array_digest(poison["logits"]) != manifest["poison_logits_sha256"]:
            raise ValueError(f"{name}: P logits differ from the fixed verifier")
        if entry["verification_domain"] == entry["suspect_training_domain"]:
            raise ValueError(f"{name}: this is not a cross-dataset direction")
        # When collection supplied a domain, contradicting manifest declarations fail closed.
        for outputs in (suspect, poison):
            actual_domain = outputs["metadata"].get("query_domain")
            if actual_domain is not None and actual_domain != entry["verification_domain"]:
                raise ValueError(f"{name}: collected query_domain contradicts the declared verifier domain")
        suspect_metadata = suspect["metadata"]
        if suspect_metadata.get("uses_victim_outputs") is True:
            raise ValueError(f"{name}: suspect metadata declares use of victim outputs")
        pairs.append((entry["verification_domain"], entry["suspect_training_domain"]))
        checked.append((entry, verifier_path, suspect["path"], poison["path"]))

    if pairs[0] != tuple(reversed(pairs[1])):
        raise ValueError("the two domain pairs must be reciprocal")
    if directions[0]["class_names"] != directions[1]["class_names"]:
        raise ValueError("both directions must use the same explicitly documented output class space")
    return checked

def main(argv=None):  # 6 个用法 (5 个动态)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="two directions, paths relative to this JSON")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto", help="cuda:0 after CUDA_VISIBLE_DEVICES remapping, or cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--validate-only", action="store_true", help="check inputs and protocol without importing torch")
    args = parser.parse_args(argv)
    try:
        manifest_path = Path(args.manifest).resolve()
        document = verifier.read_json(manifest_path)
        checked = validate_manifest(document, manifest_path.parent)
        if args.validate_only:
            print("Both directions have compatible classes, ordered queries and declared independent provenance.")
            return
        rows = []
        for entry, frozen, suspect, poison in checked:
            report = verifier.evaluate(argparse.Namespace(
                verifier=frozen, suspect=suspect, poison=poison,
                output=str(Path(args.output_dir) / entry["name"] / "result.json"),
                device=args.device, batch_size=args.batch_size,
                model_id=entry["model_id"], expected="independent"))
            rows.append({"direction": entry["name"],
                         "verification_domain": entry["verification_domain"],
                         "suspect_training_domain": entry["suspect_training_domain"],
                         **report["model_level"]})
        verifier.write_json(Path(args.output_dir) / "cross_dataset_results.json", {
            "directions": rows, "manifest": document,
            "note": "Two different frozen verifiers are reported separately. This supplement alone does not exclude the low-loss deletion shortcut."})
    except (ValueError, KeyError, FileNotFoundError, ImportError, RuntimeError) as exc:
        parser.exit(2, f"error: {exc}\n")

if __name__ == "__main__":
    main()