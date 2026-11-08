"""Reuse R5.1 full-D independent checkpoints for R5.2 without retraining/count inflation."""
import argparse
import json
from pathlib import Path

def main(argv=None):  # 6 个用法 (5 个动态)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", required=True, help="01_sample_removal models.json")
    parser.add_argument("--victim-metadata", required=True, help="Complete V metadata.json")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    source = Path(args.models).resolve()
    entries = json.loads(source.read_text())
    if isinstance(entries, dict):
        entries = entries["models"]
    victim = json.loads(Path(args.victim_metadata).read_text())
    reused, seeds = [], set()
    for entry in entries:
        if entry.get("group") != "full":
            continue
        checkpoint = Path(entry["checkpoint"])
        if not checkpoint.is_absolute():
            checkpoint = source.parent / checkpoint
        if not checkpoint.is_file():
            raise ValueError(f"Missing complete full-D checkpoint: {checkpoint}")
        metadata = json.loads(checkpoint.with_name("metadata.json").read_text())
        if not metadata.get("complete_model") or not metadata.get("classifier_saved"):
            raise ValueError("Only complete full-D checkpoints can be reused.")
        if metadata.get("initial_checkpoint") or metadata.get("initialization") != "foundation_model_F":
            raise ValueError("The full group must have started independently from F.")
        for key in ("dataset_train_fingerprint", "labels", "train_count"):
            if not metadata.get(key) or metadata[key] != victim.get(key):
                raise ValueError(f"Full-D checkpoint differs from victim in {key}.")
        if Path(metadata["model_name_or_path"]).name != Path(victim["model_name_or_path"]).name:
            raise ValueError("Full-D checkpoint uses a different foundation model.")
        if metadata.get("selection", {}).get("original_indices") != list(range(victim["train_count"])):
            raise ValueError("Source checkpoint does not certify the complete original D in original order.")
        seed = metadata["seed"]
        if seed in seeds:
            raise ValueError("Repeated training seeds in full-D source manifest.")
        seeds.add(seed)
        reused.append({**entry, "checkpoint": str(checkpoint.resolve()), "expected": "independent",
                       "group": "overlap_1.0", "requested_overlap": 1.0, "actual_overlap": 1.0,
                       "reused_from": str(source), "is_new_training_run": False})
    if not reused:
        raise ValueError("No certified full-D independent models in source manifest.")
    target = Path(args.output_dir)
    if (target / "models.json").exists():
        raise ValueError("Use a new output directory to avoid overwriting a model manifest.")
    target.mkdir(parents=True, exist_ok=True)
    (target / "models.json").write_text(json.dumps(reused, ensure_ascii=False, indent=2) + "\n")
    (target / "reuse_note.json").write_text(json.dumps({
        "source": str(source), "model_count": len(reused), "additional_training_model_count": 0,
        "note": "These are the same full-D checkpoints used for R5.1; never count them twice in a combined denominator."},indent=2)+"\n")
    print(f"Reused {len(reused)} existing full-D checkpoints; no additional models were trained: {target / 'models.json'}")

if __name__ == "__main__":
    main()