"""Rebuild complete V/P/B checkpoints without relying on incomplete legacy student.pt.

Use the executable run.sh from any directory; all artifacts stay in review_exp.
"""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def write_json(path, value):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def validate_indices(values, size):
    if not isinstance(values, list) or not values:
        raise ValueError("Ds indices must be a nonempty JSON list.")
    if any(type(i) is not int or i < 0 or i >= size for i in values):
        raise ValueError(f"Ds indices must be integers in [0, {size}].")
    if len(set(values)) != len(values):
        raise ValueError("Ds indices must be unique.")
    return values

def label_digest(labels):
    return hashlib.sha256(json.dumps(labels, separators=(",", ":")).encode()).hexdigest

def protocol_indices(labels, ordered_indices, poison_fraction, drop_fraction, target_label, supplied=None):
    """Select before filtering target labels, exactly as the original cube protocol."""
    size = len(labels)
    if sorted(ordered_indices) != list(range(size)):
        raise ValueError("The loss ranking must cover every original training index exactly once.")
    if not 0 < poison_fraction < 1 or not 0 <= drop_fraction < 1:
        raise ValueError("Require 0 < poison_fraction < 1 and 0 <= drop_fraction < 1.")
    candidates = (validate_indices(supplied, size) if supplied is not None
                  else ordered_indices[:int(size * poison_fraction)])
    actual_rank_order = [index for index in candidates if labels[index] != target_label]
    if not actual_rank_order:
        raise ValueError("No non-target samples remain in Ds; change selection/target_label.")
    actual_set = set(actual_rank_order)
    b_dropped = ordered_indices[:int(size * drop_fraction)]
    dropped_set = set(b_dropped)
    return {
        "candidate_indices": candidates,
        "actual_query_rank_order": actual_rank_order,
        "query_indices": sorted(actual_rank_order),
        "poisoned_training_order": [i for i in range(size) if i not in actual_set] + actual_rank_order,
        "benign_dropped_indices": b_dropped,
        "benign_retained_indices": [i for i in range(size) if i not in dropped_set],
    }

def black_square(image, size, location):
    """Match legacy cube's uint8(256)==0 result at (row=location, col=location)."""
    from PIL import Image
    import numpy as np

    image = image.convert("RGB")
    if size <= 0 or location < 0 or location + size > min(image.height, image.width):
        raise ValueError(f"Square size={size}, location={location} does not fit raw RGB image {image.size}.")
    pixels = np.array(image, dtype=np.uint8, copy=True)
    pixels[location:location + size, location:location + size, :] = 0
    return Image.fromarray(pixels)

def _labels(raw):  # 4 用法
    try:
        return [int(value) for value in raw["labels"]]
    except (KeyError, TypeError):
        return [int(raw[i]["labels"]) for i in range(len(raw))]

def _read_supplied(path):  # 1 个用法
    if not path:
        return None
    value = json.loads(Path(path).read_text())
    if isinstance(value, dict):
        for key in ("query_indices", "indices", "selected_indices"):
            if key in value:
                return value[key]
        raise ValueError("Ds JSON object must contain query_indices, indices or selected_indices.")
    return value

def rank(args):
    if args.allow_incomplete_legacy_diagnostic:
        raise ValueError("Formal baseline ranking forbids incomplete legacy diagnostics.")
    import numpy as np
    import torch
    from accelerate import Accelerator
    from torch.utils.data import DataLoader, Dataset
    from review_exp.common.runtime import load_data, load_model, classification_logits

    if not args.checkpoint:
        raise ValueError("rank requires --checkpoint with the COMPLETE V checkpoint.")
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    bundle = load_data(args)
    raw = bundle["dataset"]["train"]
    transform = bundle["val_transforms"]

    class RankingDataset(Dataset):
        def __len__(self):
            return len(raw)

        def __getitem__(self, index):
            row = raw[index]
            return {"pixel_values": transform(row["image"].convert("RGB")),
                    "labels": int(row["labels"]), "index": index}

    loader = DataLoader(RankingDataset(), batch_size=args.rank_batch_size,
                        shuffle=False, num_workers=0)
    model = load_model(args, accelerator, bundle["labels"])
    model, loader = accelerator.prepare(model, loader)
    model.eval()
    rows = {}
    duplicate_count = 0
    with torch.inference_mode():
        for batch in loader:
            with accelerator.autocast():
                logits = classification_logits(model, batch["pixel_values"])
            losses = torch.nn.functional.cross_entropy(logits.float(), batch["labels"], reduction="none")
            indices, values = accelerator.gather_for_metrics((batch["index"], losses))
            if accelerator.is_main_process:
                for index, value in zip(indices.cpu().tolist(), values.cpu().tolist()):
                    if index in rows:
                        duplicate_count += 1
                    rows[index] = value
    if accelerator.is_main_process:
        if duplicate_count or sorted(rows) != list(range(len(raw))):
            raise ValueError("Loss gather must cover every original training index exactly once; "
                             f"size={len(raw)}, gathered={len(rows)}, duplicates={duplicate_count}.")
        if not all(math.isfinite(value) for value in rows.values()):
            raise ValueError("Non-finite per-sample loss; the ranking was not saved.")
        target = Path(args.baseline_dir)
        target.mkdir(parents=True, exist_ok=True)
        # String IDs and dict format intentionally match the existing load_order API.
        np.save(target / "loss_order.npy", {str(i): float(rows[i]) for i in range(len(raw))})
        labels = _labels(rows)
        write_json(target / "losses.json", {
            "schema": "holmes_loss_ranking_v1", "dataset_name": args.dataset_name,
            "dataset_size": len(raw), "num_train": len(raw), "split": "train",
            "dataset_fingerprint": getattr(raw, "_fingerprint", label_digest(labels)),
            "labels_sha256": label_digest(labels), "class_names": list(bundle["labels"]),
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "preprocessing": repr(transform), "num_processes": accelerator.num_processes,
            "ordered_indices": sorted(rows, key=lambda i: (rows[i], i)),
            "losses": {str(i): float(rows[i]) for i in range(len(raw))},
        })
        print(f"Saved deterministic full-training-set ranking: {target / 'loss_order.npy'}")
    accelerator.wait_for_everyone()

def prepare(args):
    if args.allow_incomplete_legacy_diagnostic:
        raise ValueError("Formal baseline selection forbids incomplete legacy diagnostics.")
    if args.target_label < 0 or args.trigger_location < 0 or args.trigger_size <= 0:
        raise ValueError("Target/location must be nonnegative and trigger_size must be positive.")
    from accelerate import Accelerator
    from review_exp.common.runtime import load_data

    accelerator = Accelerator()
    bundle = load_data(args)
    raw = bundle["dataset"]["train"]
    labels = _labels(raw)
    if not 0 <= args.target_label < len(bundle["labels"]):
        raise ValueError("target_label is outside the dataset class space.")
    target = Path(args.baseline_dir)
    ranking = json.loads((target / "losses.json").read_text())
    if (ranking.get("num_train") != len(raw) or ranking.get("labels_sha256") != label_digest(labels)
            or ranking.get("class_names") != list(bundle["labels"])
            or ranking.get("dataset_fingerprint") != getattr(raw, "_fingerprint", label_digest(labels))):
        raise ValueError("Dataset differs from losses.json; rerun the ranking stage on this exact dataset.")
    losses = ranking["losses"]
    if not isinstance(losses, dict) or {int(key) for key in losses} != set(range(len(raw))):
        raise ValueError("loss_order.npy must be a complete original-index-to-loss dictionary.")
    clean_losses = {int(key): float(value) for key, value in losses.items()}
    if not all(math.isfinite(value) for value in clean_losses.values()):
        raise ValueError("loss_order.npy contains a non-finite loss.")
    order = sorted(clean_losses, key=lambda i: (clean_losses[i], i))
    supplied = _read_supplied(args.ds_indices)
    selected = protocol_indices(labels, order, args.poison_fraction, args.drop_fraction,
                                args.target_label, supplied)
    # Fail before expensive training if a trigger does not fit any selected raw image.
    for index in selected["query_indices"]:
        width, height = raw[index]["image"].size
        if args.trigger_location + args.trigger_size > min(width, height):
            raise ValueError(f"Trigger does not fit original training image {index}: {(width, height)}.")
    metadata = {
        "schema": "holmes_complete_baseline_selection_v1", "dataset_name": args.dataset_name,
        "dataset_size": len(raw), "class_names": list(bundle["labels"]),
        "dataset_fingerprint": getattr(raw, "_fingerprint", label_digest(labels)),
        "labels_sha256": label_digest(labels), "poison_fraction": args.poison_fraction,
        "drop_fraction": args.drop_fraction, "target_label": args.target_label,
        "trigger_size": args.trigger_size, "trigger_location": args.trigger_location,
        "trigger_value": 0, "trigger_coordinates": "raw RGB row=location, column=location",
        "ds_indices_source": str(Path(args.ds_indices).resolve()) if args.ds_indices else None,
        "candidate_count": len(selected["candidate_indices"]),
        "actual_poison_count": len(selected["query_indices"]),
        "actual_poison_fraction": len(selected["query_indices"]) / len(raw),
        "benign_retained_count": len(selected["benign_retained_indices"]),
        **selected,
    }
    if accelerator.is_main_process:
        write_json(target / "selection.json", metadata)
        write_json(target / "query_indices.json", selected["query_indices"])
        write_json(target / "benign_retained_indices.json", selected["benign_retained_indices"])
        print(f"Prepared P replacements={metadata['actual_poison_count']}/{len(raw)}, "
              f"B retained={metadata['benign_retained_count']}/{len(raw)}: {target / 'selection.json'}")
    accelerator.wait_for_everyone()

def _load_selection(args, bundle):  # 1 个用法
    metadata = json.loads((Path(args.baseline_dir) / "selection.json").read_text())
    raw = bundle["dataset"]["train"]
    if metadata.get("schema") != "holmes_complete_baseline_selection_v1":
        raise ValueError("Invalid selection schema; rerun prepare.")
    if (metadata["dataset_size"] != len(raw) or metadata["class_names"] != list(bundle["labels"])
            or metadata["dataset_fingerprint"] != getattr(raw, "_fingerprint", label_digest(_labels(raw)))
            or metadata["labels_sha256"] != label_digest(_labels(raw))):
        raise ValueError("Dataset order, labels or class mapping changed since prepare.")
    return metadata

def train(args):  # 13 个用法 (12 个动态)
    from torch.utils.data import Dataset
    from review_exp.common.runtime import load_data
    from review_exp.common.train import train_model

    bundle = load_data(args)
    run_args = copy.copy(args)
    selection = {"stage": args.stage, "protocol": "complete_V_P_B_rebuild_v1"}
    if args.stage in ("victim", "benign") and args.checkpoint:
        raise ValueError(f"{args.stage} must start from F; omit --checkpoint. "
                         "Use the poisoned stage for initialization from V.")
    if args.stage == "poisoned" and not args.checkpoint:
        raise ValueError("poisoned requires --checkpoint with the COMPLETE V checkpoint.")
    if not getattr(run_args, "output_dir", None):
        run_args.output_dir = str(Path(args.baseline_dir) / args.stage)
    overrides = None
    if args.stage in ("poisoned", "benign"):
        metadata = _load_selection(args, bundle)
        raw = bundle["dataset"]["train"]
        poisoned = args.stage == "poisoned"
        indices = (metadata["poisoned_training_order"] if poisoned
                   else metadata["benign_retained_indices"])
        affected = set(metadata["query_indices"]) if poisoned else set()

        class PreparedDataset(Dataset):
            def __init__(self, source, order, transform, poison_ids=None):
                self.source, self.order, self.transform = source, order, transform
                self.poison_ids = poison_ids or set()

            def __len__(self):
                return len(self.order)

            def __getitem__(self, position):
                index = self.order[position]
                row = self.source[index]
                image, label = row["image"].convert("RGB"), int(row["labels"])
                if index in self.poison_ids:
                    image = black_square(image, metadata["trigger_size"], metadata["trigger_location"])
                    label = metadata["target_label"]
                return {"pixel_values": self.transform(image), "labels": label, "id": index}

        validation = bundle["dataset"]["validation"]
        overrides = {
            "train": PreparedDataset(raw, indices, bundle["train_transforms"], affected),
            "validation": PreparedDataset(validation, list(range(len(validation))), bundle["val_transforms"]),
        }
        selection.update({
            "selection_file": str((Path(args.baseline_dir) / "selection.json").resolve()),
            "selection_sha256": hashlib.sha256((Path(args.baseline_dir) / "selection.json").read_bytes()).hexdigest(),
            "training_indices": indices, "actual_query_indices": metadata["query_indices"],
            "dataset_size_before_selection": len(raw), "num_training_samples": len(indices),
            "poisoned_replacements": len(affected), "target_label": metadata["target_label"] if poisoned else None,
            "trigger": {"style": "legacy_black_cube", "size": metadata["trigger_size"],
                        "location": metadata["trigger_location"], "pixel_value": 0} if poisoned else None,
            "initialization": "complete_V_checkpoint" if poisoned else "foundation_F",
        })
    else:
        selection.update({"initialization": "foundation_F", "num_training_samples": len(bundle["dataset"]["train"])})
    checkpoint = train_model(run_args, dataset_override=overrides,
                             selection_metadata=selection, data_bundle=bundle)
    print(f"Complete {args.stage} checkpoint: {checkpoint}")

def main(argv=None):  # 6 个用法 (5 个动态)
    from review_exp.common.train import add_training_arguments

    parser = argparse.ArgumentParser(description=__doc__)
    add_training_arguments(parser)
    parser.add_argument("--stage", choices=["victim", "rank", "prepare", "poisoned", "benign"], required=True)
    parser.add_argument("--baseline-dir", default="review_exp/00_prepare_baseline/results")
    parser.add_argument("--rank-batch-size", type=int, default=128)
    parser.add_argument("--poison-fraction", type=float, default=0.1)
    parser.add_argument("--drop-fraction", type=float, default=0.5)
    parser.add_argument("--target-label", type=int, default=0)
    parser.add_argument("--trigger-size", type=int, default=4)
    parser.add_argument("--trigger-location", type=int, default=18)
    parser.add_argument("--ds-indices", help="Optional original-training-index JSON for the pre-filter Ds candidate.")
    args = parser.parse_args(argv)
    if args.rank_batch_size <= 0 or args.trigger_size <= 0 or args.trigger_location < 0:
        parser.error("Batch size and trigger size must be positive; trigger location must be nonnegative.")
    if args.stage == "rank":
        rank(args)
    elif args.stage == "prepare":
        prepare(args)
    else:
        train(args)

if __name__ == "__main__":
    main()