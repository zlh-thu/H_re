"""R5.2: independent training from the same F on fixed D, with optional real Dprime."""
import argparse
import copy
import hashlib
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from review_exp.common import runtime
from review_exp.common.train import add_training_arguments, train_model

selection = importlib.import_module("review_exp.02_data_overlap.selection")

def image_hash(row : (__getitem__, sha256)):  # 2 用法
    image = row["image"].convert("RGB")
    return hashlib.sha256(f"{image.width}:{image.height}:RGB:".encode() + image.tobytes()).hexdigest()

def load_extra_pool(args, bundle):  # 1 个用法
    from datasets import DatasetDict, load_from_disk
    if not args.extra_pool_dir or not args.extra_pool_provenance:
        raise ValueError("Overlap below 1 needs --extra_pool_dir and --extra_pool_provenance. Never partition D and call it disjoint from V.")
    provenance = selection.validate_pool_provenance(json.loads(Path(args.extra_pool_provenance).read_text(encoding="utf-8")))
    pool = load_from_disk(args.extra_pool_dir)
    if isinstance(pool, DatasetDict):
        if set(pool) != {"train"}:
            raise ValueError("Extra saved DatasetDict must contain only train; test/validation data is not an extra training pool.")
        pool = pool["train"]
    renames = {}
    if "image" not in pool.column_names and "img" in pool.column_names:
        renames["img"] = "image"
    if "labels" not in pool.column_names and "label" in pool.column_names:
        renames["label"] = "labels"
    if renames:
        pool = pool.rename_columns(renames)
    if not {"image", "labels"}.issubset(pool.column_names):
        raise ValueError("Extra pool needs original image and ground-truth labels columns.")
    names = getattr(pool.features["labels"], "names", None)
    if names is not None and list(names) != list(bundle["labels"]):
        raise ValueError("Extra pool ClassLabel names/order mismatch.")
    if any(type(label) is not int or not 0 <= label < len(bundle["labels"]) for label in pool["labels"]):
        raise ValueError("Extra pool label IDs must be integers in the victim class space.")
    # Pixel identity is checked against all original train AND held-out splits.
    # Provenance still matters: hashing cannot prove near-duplicate/source origin.
    forbidden = set()
    for raw in bundle["dataset"].values():
        forbidden.update(image_hash(row) for row in raw)
    seen = set()
    for index, row in enumerate(pool):
        digest = image_hash(row)
        if digest in forbidden:
            raise ValueError(f"Extra pool row {index} duplicates original D or held-out test/validation image.")
        if digest in seen:
            raise ValueError(f"Extra pool duplicate raw image at row {index}.")
        seen.add(digest)
    return pool, provenance

def main(argv=None):  # 6 个用法 (5 个动态)
    parser = argparse.ArgumentParser(description=__doc__)
    add_training_arguments(parser)
    parser.add_argument("--seeds", type=int, nargs="+", default=[43, 44, 45])
    parser.add_argument("--overlap_rates", type=float, nargs="+", default=[1.0])
    parser.add_argument("--data_seed", type=int, default=2026,
                        help="Fix D/Dprime membership across training seeds; never use a model seed to redefine D.")
    parser.add_argument("--extra_pool_dir", default=None, help="HF Dataset.save_to_disk with real labelled, disjoint in-distribution train images.")
    parser.add_argument("--extra_pool_provenance", default=None, help="JSON source/ground_truth/class_names/disjoint attestation.")
    args = parser.parse_args(argv)
    if args.checkpoint:
        raise ValueError("All models in this experiment must start from F, not from any victim checkpoint.")
    if len(set(args.seeds)) != len(args.seeds) or not args.seeds:
        raise ValueError("Training seeds must be nonempty and unique.")
    if len(set(args.overlap_rates)) != len(args.overlap_rates) or any(not 0 <= x <= 1 for x in args.overlap_rates):
        raise ValueError("Overlap rates must be unique values in [0,1].")
    from accelerate import Accelerator
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    bundle = runtime.load_data(args)
    raw = bundle["dataset"]["train"]
    labels = [int(value) for value in raw["labels"]]
    extra, provenance = (None, None)
    if any(rate < 1 for rate in args.overlap_rates):
        extra, provenance = load_extra_pool(args, bundle)
    extra_labels = [int(value) for value in extra["labels"]] if extra is not None else []
    output_root = Path(args.output_dir).resolve()
    entries = []
    for overlap in args.overlap_rates:
        plan = selection.select_overlap(labels, extra_labels, overlap, args.data_seed)
        plan.update({"schema": "holmes_data_overlap_v1", "experiment": "R5.2_same_F_data_overlap",
                     "base_dataset_fingerprint": getattr(raw, "_fingerprint", None),
                     "split_seed": args.split_seed, "provenance": provenance,
                     "base_labels": [labels[index] for index in plan["base_indices"]],
                     "extra_labels": [extra_labels[index] for index in plan["extra_indices"]],
                     "extra_pool_fingerprint": getattr(extra, "_fingerprint", None),
                     "extra_pool_dir": str(Path(args.extra_pool_dir).resolve()) if extra is not None else None,
                     "verifier_policy": "V/P/C/threshold frozen; evaluation models never enter C training"})
        rate_name = format(overlap, ".6g").replace(".", "p")
        if accelerator.is_main_process:
            output_root.mkdir(parents=True, exist_ok=True)
            (output_root / f"selection_overlap{rate_name}.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
        accelerator.wait_for_everyone()
        if overlap == 1.0:
            train_rows = raw  # Keep the exact same D and its order for every seed.
        else:
            from datasets import concatenate_datasets
            selected_base = raw.select(plan["base_indices"])
            selected_extra = extra.select(plan["extra_indices"])
            # Restrict both sources to matching raw fields; never include V logits.
            selected_base = selected_base.select_columns(["image", "labels"])
            selected_extra = selected_extra.select_columns(["image", "labels"])
            # Cast labels to the same ClassLabel schema when source uses integers.
            selected_extra = selected_extra.cast(selected_base.features)
            train_rows = concatenate_datasets([selected_base, selected_extra])
        for seed in args.seeds:
            run_args = copy.copy(args)
            run_args.seed = seed
            run_args.checkpoint = None
            model_id = f"independent_overlap{rate_name}_seed{seed}"
            run_args.output_dir = str(output_root / model_id)
            checkpoint = train_model(run_args,
                                     dataset_override={"train": train_rows,
                                                       "validation": bundle["dataset"]["validation"]},
                                     selection_metadata=plan, data_bundle=bundle)
            entries.append({"model_id": model_id, "group": f"overlap_{overlap}", "expected": "independent",
                            "seed": seed, "data_seed": args.data_seed, "checkpoint": checkpoint,
                            "requested_overlap": overlap, "actual_overlap": plan["actual_overlap"],
                            "train_count": len(train_rows)})
            if accelerator.is_main_process:
                (output_root / "models.json").write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    if accelerator.is_main_process:
        print(f"Independent models to evaluate: {output_root / 'models.json'}")

if __name__ == "__main__":
    main()