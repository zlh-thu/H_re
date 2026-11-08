"""Freeze raw query identities, then export aligned deterministic model outputs.

Run as `python -m review_exp.common.collect_outputs --help` from the repo root.
Heavy dependencies are imported only by the requested operation.
"""

import argparse
import hashlib
import json
from pathlib import Path


def image_digest(image):  # 3 用法
    image = image.convert("RGB")
    header = f"RGB:{image.width}:{image.height}:".encode()
    return hashlib.sha256(header + image.tobytes()).hexdigest()


def validate_indices(values, size):  # 4 用法
    if not isinstance(values, list) or not values:
        raise ValueError("Query indices must be a nonempty JSON list.")
    if any(type(i) is not int or i < 0 or i >= size for i in values):
        raise ValueError(f"Indices must be integers in [0, {size}).")
    if len(set(values)) != len(values):
        raise ValueError(f"Repeated query indices are not allowed.")
    return values


def write_json(path, value):  # 2 用法
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")

def checkpoint_digest(path):  # 3 用法
    """Byte identity for model-count auditing; metadata/log files are excluded."""
    from review_exp.common.runtime import resolve_checkpoint
    source = resolve_checkpoint(path)
    if not source.is_file():
        raise ValueError(f"No complete checkpoint tensor file found at {source}.")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bundle(args):  # 2 用法
    from review_exp.common.runtime import load_data
    bundle = load_data(args)
    return bundle


def _read_index_file(path):  # 1 个用法
    value = json.loads(Path(path).read_text())
    if isinstance(value, dict):
        for key in ("query_indices", "indices", "selected_indices"):
            if key in value:
                return value[key]
        raise ValueError("Index JSON needs query_indices, indices or selected_indices.")
    return value

def prepare(args):  # 12 个用法 (11 个动态)
    bundle = _bundle(args)
    raw = bundle["dataset"][args.split]
    indices = validate_indices(_read_index_file(args.indices), len(raw))
    digests, labels = [], []
    for idx in indices:
        example = raw[idx]
        digests.append(image_digest(example["image"]))
        labels.append(int(example["labels"]))
    if len(set(digests)) != len(digests):
        raise ValueError("Queries contain duplicate raw images; remove duplicates before splitting C data.")
    manifest = {
        "schema": "holmes_queries_v1", "dataset_name": args.dataset_name,
        "split": args.split, "dataset_size": len(raw),
        "class_names": list(bundle["labels"]), "query_indices": indices,
        "query_ids": ["sha256:" + x for x in digests],
        "raw_image_sha256": digests, "labels": labels,
        "preprocessing": repr(bundle["val_transforms"]),
        "query_source": str(Path(args.indices).resolve()),
    }
    write_json(args.output, manifest)
    print(f"Frozen {len(indices)} raw-image query identities: {args.output}")

def _check_manifest(manifest, raw, bundle, allow_preprocess_difference=False):
    if manifest.get("schema") != "holmes_queries_v1":
        raise ValueError("Unsupported query-manifest schema.")
    if list(bundle["labels"]) != manifest["class_names"]:
        raise ValueError("Dataset class names/order differ from the frozen queries.")
    if len(raw) != manifest["dataset_size"]:
        raise ValueError("Dataset split size changed since query preparation.")
    indices = validate_indices(manifest["query_indices"], len(raw))
    if not (len(indices) == len(manifest["query_ids"]) == len(manifest["labels"])):
        raise ValueError("Manifest columns have different lengths.")
    if len(set(manifest["query_ids"])) != len(indices):
        raise ValueError("Query IDs must be unique.")
    for position, idx in enumerate(indices):
        row = raw[idx]
        if "sha256:" + image_digest(row["image"]) != manifest["query_ids"][position]:
            raise ValueError(f"Raw image changed/reordered at query position {position} (dataset index {idx}).")
        if int(row["labels"]) != manifest["labels"][position]:
            raise ValueError(f"Query label changed at position {position}.")
    if repr(bundle["val_transforms"]) != manifest["preprocessing"] and not allow_preprocess_difference:
        raise ValueError("Preprocessing changed. Use the same foundation path/settings, or explicitly "
                         "pass --allow-preprocess-difference for a model requiring its own native preprocessing.")
    return indices

def collect(args):
    import numpy as np
    import torch
    from accelerate import Accelerator
    from torch.utils.data import DataLoader, Dataset
    from review_exp.common.runtime import load_model, classification_logits

    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    bundle = _bundle(args)
    manifest = json.loads(Path(args.query_manifest).read_text())
    raw = bundle["dataset"][manifest["split"]]
    indices = _check_manifest(manifest, raw, bundle, args.allow_preprocess_difference)
    transform = bundle["val_transforms"]

    class Queries(Dataset):
        def __len__(self):
            return len(indices)

        def __getitem__(self, position):
            row = raw[indices[position]]
            return {"pixel_values": transform(row["image"].convert("RGB")),
                    "labels": int(row["labels"]), "position": position}

    # num_workers=0 also works on macOS spawn and avoids copying a local Dataset class.
    loader = DataLoader(Queries(), batch_size=args.query_batch_size, shuffle=False, num_workers=0)
    model = load_model(args, accelerator, bundle["labels"])
    if getattr(model, "review_incomplete_legacy_diagnostic", False):
        raise ValueError("An incomplete legacy model cannot produce formal query outputs; rebuild complete checkpoints first.")
    checkpoint_metadata = getattr(model, "review_checkpoint_metadata", {})
    if args.expected == "independent" and checkpoint_metadata.get("initial_checkpoint"):
        raise ValueError("This checkpoint was initialized from another personalized checkpoint; it cannot be labelled independent from F.")
    model, loader = accelerator.prepare(model, loader)
    model.eval()
    rows = {}
    with torch.inference_mode():
        for batch in loader:
            with accelerator.autocast():
                logits = classification_logits(model(batch["pixel_values"]))
            positions, outputs, labels = accelerator.gather_for_metrics(
                (batch["position"], logits.float(), batch["labels"]))
            if accelerator.is_main_process:
                for position, output, label in zip(positions.cpu().tolist(), outputs.cpu().numpy(), labels.cpu().tolist()):
                    if position in rows and not np.allclose(rows[position][0], output, rtol=1e-5, atol=1e-5):
                        raise ValueError("Distributed gather produced inconsistent duplicate outputs.")
                    rows[position] = (output, label)
    if accelerator.is_main_process:
        if sorted(rows) != list(range(len(indices))):
            raise ValueError("Distributed export did not cover every query exactly once.")
        logits = np.asarray([rows[i][0] for i in range(len(indices))], dtype=np.float32)
        labels = np.asarray([rows[i][1] for i in range(len(indices))], dtype=np.int64)
        if logits.shape != (len(indices), len(bundle["labels"])) or not np.isfinite(logits).all():
            raise ValueError("Non-finite logits or incorrect output class dimension.")
        if labels.tolist() != manifest["labels"]:
            raise ValueError("Gathered labels do not match the frozen manifest.")
        metadata = {
            "schema": "holmes_outputs_v1", "checkpoint": str(Path(args.checkpoint).resolve()),
            "checkpoint_sha256": checkpoint_digest(args.checkpoint),
            "model_id": args.model_id, "expected": args.expected,
            "query_manifest": str(Path(args.query_manifest).resolve()),
            "query_manifest_sha256": hashlib.sha256(Path(args.query_manifest).read_bytes()).hexdigest(),
            "preprocessing": repr(transform), "preprocess_difference_allowed": args.allow_preprocess_difference,
            "accuracy": float((logits.argmax(1) == labels).mean()),
            "num_processes": accelerator.num_processes,
            "mixed_precision": accelerator.mixed_precision,
            "training_seed": checkpoint_metadata.get("seed"),
            "train_count": checkpoint_metadata.get("train_count"),
            "training_optimizer_steps": checkpoint_metadata.get("optimizer_steps"),
            "training_effective_batch_size": checkpoint_metadata.get("effective_batch_size"),
            "task_validation_accuracy": checkpoint_metadata.get("validation_accuracy"),
            "initialization": checkpoint_metadata.get("initialization"),
            "initial_checkpoint": checkpoint_metadata.get("initial_checkpoint"),
        }
        output = Path(args.output)
        if output.suffix != ".npz":
            raise ValueError("Output must have the .npz extension.")
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, logits=logits, labels=labels,
                            query_ids=np.asarray(manifest["query_ids"]),
                            class_names=np.asarray(bundle["labels"]),
                            metadata_json=json.dumps(metadata, ensure_ascii=False))
        write_json(output.with_suffix(".json"), metadata)
        print(f"Saved aligned logits {logits.shape}; query accuracy={metadata['accuracy']:.6f}: {output}")
    accelerator.wait_for_everyone()

def main(argv=None):  # 6 个用法 (5 个动态)
    from review_exp.common.runtime import add_runtime_arguments
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="Freeze exact original-image identities and order.")
    add_runtime_arguments(prep)
    prep.add_argument("--indices", required=True, help="JSON list of ORIGINAL split indices (e.g. actual Ds).")
    prep.add_argument("--split", default="train")
    prep.add_argument("--output", required=True)
    prep.set_defaults(func=prepare)
    export = sub.add_parser("collect", help="Query a COMPLETE checkpoint on the frozen images.")
    add_runtime_arguments(export)
    export.add_argument("--query-manifest", required=True)
    export.add_argument("--query-batch-size", type=int, default=128)
    export.add_argument("--output", required=True)
    export.add_argument("--model-id", required=True)
    export.add_argument("--expected", choices=["independent", "stolen", "reference"], default="reference")
    export.add_argument("--allow-preprocess-difference", action="store_true")
    export.set_defaults(func=collect)
    args = parser.parse_args(argv)
    if args.command == "collect" and not args.checkpoint:
        parser.error("collect requires --checkpoint with a complete checkpoint, including the trained classifier.")
    if args.command == "collect" and args.allow_incomplete_legacy_diagnostic:
        parser.error("collect forbids incomplete legacy diagnostics: a random classifier is invalid for verification.")
    if args.command == "collect" and args.query_batch_size <= 0:
        parser.error("--query-batch-size must be positive")
    args.func(args)

if __name__ == "__main__":
    main()