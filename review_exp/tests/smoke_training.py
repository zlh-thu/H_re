"""Server CPU smoke for the shared Accelerate training/checkpoint loop.

Uses a tiny synthetic classifier and tensor data, never downloads foundation
weights and never produces scientific results. Run in the original server env.
"""
import argparse
from pathlib import Path
import tempfile
from unittest.mock import patch


def main():  # 6 个用法 (5 个动态)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    import torch
    from review_exp.common import runtime
    from review_exp.common.train import add_training_arguments, train_model

    class TinyClassifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.classifier = torch.nn.Linear(12, 4)

        def forward(self, pixels : {flatten} ):
            return self.classifier(pixels.flatten(1))

    rows = [{"pixel_values": torch.full((3, 2, 2), float(i) / 8), "labels": i % 4, "id": i}
            for i in range(8)]
    bundle = {"dataset": {"train": rows, "validation": rows[:4]},
              "labels": ["a", "b", "c", "d"], "train_transforms": None, "val_transforms": None}
    with tempfile.TemporaryDirectory(prefix="holmes_training_smoke_") as directory:
        argument_parser = argparse.ArgumentParser()
        add_training_arguments(argument_parser)
        args = argument_parser.parse_args([
            "--output_dir", directory, "--mixed_precision", "no", "--batch_size", "4",
            "--eval_batch_size", "4", "--gradient_accumulation_steps", "2",
            "--max_train_steps", "2", "--num_warmup_steps", "0", "--lr", "0.01"])
        with patch.object(runtime, "load_model", side_effect=lambda *unused: TinyClassifier()):
            checkpoint = train_model(args, data_bundle=bundle)
        payload = runtime.load_complete_checkpoint(checkpoint, labels=bundle["labels"])
        if payload["metadata"]["optimizer_steps"] != 2:
            raise AssertionError("Gradient accumulation did not produce the requested update budget.")
        restored = TinyClassifier()
        restored.load_state_dict(payload["model_state"], strict=True)
        if not torch.isfinite(restored(torch.zeros(1, 3, 2, 2))).all():
            raise AssertionError("Restored classifier produced non-finite logits.")
        if not (Path(directory) / "history.json").is_file():
            raise AssertionError("Training history was not saved.")
        print("PASS: two accumulated optimizer updates -> complete checkpoint -> classifier reload (synthetic CPU data).")

if __name__ == "__main__":
    main()