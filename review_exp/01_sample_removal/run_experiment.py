import argparse
import copy
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from review_exp.common import runtime
from review_exp.common.train import add_training_arguments, train_model

selection = importlib.import_module("review_exp.01_sample_removal.selection")


def read_losses(path, raw, trust_legacy):  # 1 个用法
    if Path(path).suffix == ".npy":
        if not trust_legacy:
            raise ValueError("Old pickled npy input requires --trust_legacy_order. Prefer audited 00 losses.json.")
        import numpy as np
        payload = np.load(path, allow_pickle=True).item()
    else:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and payload.get("dataset_fingerprint"):
        fingerprint = getattr(raw, "_fingerprint", None)
        if fingerprint and payload["dataset_fingerprint"] != fingerprint:
            raise ValueError("Loss file dataset fingerprint differs from current D; rebuild loss order with the identical dataset/split.")
    return selection.validate_loss_mapping(payload, len(raw))

def main(argv=None):  # 6 个用法 (5 个动态)
    parser = argparse.ArgumentParser(description=__doc__)
    add_training_arguments(parser)
    parser.add_argument("--loss_file", "--loss-file", required=True, help="00_prepare_baseline/losses.json from frozen V on D.")
    parser.add_argument("--trust_legacy_order", action="store_true")
    parser.add_argument("--drop_fraction", type=float, default=0.5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[43, 44, 45])
    parser.add_argument("--selection_seed", type=int, default=2026,
                        help="Random-drop sample seed; fixed across model-training seeds.")
    parser.add_argument("--groups", nargs="+", choices=("full", "random_drop", "low_loss", "victim_filtered"),
                        default=["full", "random_drop", "low_loss"])
    parser.add_argument("--low_loss_checkpoint", default=None,
                        help="Optional existing COMPLETE B checkpoint; records one reference instead of retraining low_loss.")
    parser.add_argument("--victim_checkpoint", default=None,
                        help="Complete V needed only for optional victim_filtered positive control.")
    args = parser.parse_args(argv)
    if args.checkpoint:
        raise ValueError("Independent groups must start from F: omit --checkpoint. Use --victim_checkpoint only for victim_filtered.")
    if len(args.seeds) != len(set(args.seeds)) or not args.seeds:
        raise ValueError("Training seeds must be nonempty and unique.")
    if "victim_filtered" in args.groups and not args.victim_checkpoint:
        raise ValueError("victim_filtered requires --victim_checkpoint (complete V).")
    from accelerate import Accelerator
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    bundle = runtime.load_data(args)
    raw = bundle["dataset"]["train"]
    labels = [int(value) for value in raw["labels"]]
    losses = read_losses(args.loss_file, raw, args.trust_legacy_order)
    low_keep = selection.low_loss_keep(labels, losses, args.drop_fraction)
    random_keep = selection.random_keep_matching_classes(labels, low_keep, args.selection_seed)
    all_keep = list(range(len(raw)))
    if not low_keep:
        raise ValueError("Low-loss removal leaves no training samples.")
    output_root = Path(args.output_dir).resolve()
    plan = {"schema": "holmes_sample_removal_v1", "drop_fraction": args.drop_fraction,
            "selection_seed": args.selection_seed, "split_seed": args.split_seed,
            "dataset_fingerprint": getattr(raw, "_fingerprint", None),
            "num_train": len(raw), "loss_file": str(Path(args.loss_file).resolve()),
            "low_loss_keep_indices": low_keep, "random_drop_keep_indices": random_keep,
            "low_loss_keep_class_counts": selection.class_counts(labels, low_keep),
            "random_drop_keep_class_counts": selection.class_counts(labels, random_keep),
            "low_loss_keep_labels": [labels[index] for index in low_keep],
            "random_drop_keep_labels": [labels[index] for index in random_keep],
            "budget_policy": "equal optimizer steps" if args.max_train_steps else "equal epochs; steps differ with set size"}
    if accelerator.is_main_process:
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "selection.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    accelerator.wait_for_everyone()
    entries = []
    for group in dict.fromkeys(args.groups):
        if group == "low_loss" and args.low_loss_checkpoint:
            payload = runtime.load_complete_checkpoint(args.low_loss_checkpoint, labels=bundle["labels"], model_name_or_path=args.model_name_or_path)
            saved_selection = payload["metadata"].get("selection", {})
            saved_indices = saved_selection.get("original_indices", saved_selection.get("training_indices"))
            if saved_indices is None or sorted(saved_indices) != low_keep:
                raise ValueError("Existing low_loss checkpoint does not certify this exact low-loss retained D; retrain it or use the matching baseline B.")
            entries.append({"model_id": "low_loss_reference", "group": group, "expected": "reference",
                            "checkpoint": str(runtime.resolve_checkpoint(args.low_loss_checkpoint)),
                            "seed": payload["metadata"].get("seed"), "existing_reference": True})
            continue
        keep = {"full": all_keep, "random_drop": random_keep,
                "low_loss": low_keep, "victim_filtered": low_keep}[group]
        for seed in args.seeds:
            run_args = copy.copy(args)
            run_args.seed = seed
            run_args.checkpoint = args.victim_checkpoint if group == "victim_filtered" else None
            model_id = f"{group}_seed{seed}"
            run_args.output_dir = str(output_root / model_id)
            selected = {"train": raw.select(keep), "validation": bundle["dataset"]["validation"]}
            metadata = {"experiment": "R5.1_sample_removal", "group": group,
                        "original_indices": keep, "original_labels": [labels[index] for index in keep],
                        "selection_seed": args.selection_seed, "drop_fraction": args.drop_fraction,
                        "class_counts": selection.class_counts(labels, keep),
                        "selection_file": str(output_root / "selection.json"),
                        "verifier_policy": "V/P/C/threshold frozen; these S models are evaluation only"}
            checkpoint = train_model(run_args, dataset_override=selected, selection_metadata=metadata, data_bundle=bundle)
            entries.append({"model_id": model_id, "group": group, "seed": seed,
                            "checkpoint": checkpoint,
                            "expected": "stolen" if group == "victim_filtered" else ("reference" if group == "low_loss" else "independent"),
                            "train_count": len(keep), "selection_seed": args.selection_seed})
            if accelerator.is_main_process:
                (output_root / "models.json").write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    if accelerator.is_main_process:
        (output_root / "models.json").write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Models to evaluate with the frozen verifier: {output_root / 'models.json'}")

if __name__ == "__main__":
    main()