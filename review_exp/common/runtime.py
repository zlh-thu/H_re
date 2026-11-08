"""Shared, lazy-import runtime for review experiments.

Only the standard library is imported at module import time, so help and the
selection tests also run on a machine without the server's ML dependencies.
"""

import argparse
import copy
import hashlib
import json
import logging
import random
from pathlib import Path


CHECKPOINT_FORMAT = "holmes_review_complete_v1"


def add_runtime_arguments(parser):  # 3 用法
    parser.add_argument("--model_name_or_path", "--model-name-or-path", default="/path_to_model/clip-vit-base-patch32")
    parser.add_argument("--dataset_name", "--dataset-name", default="./dataset/cifar10.py")
    parser.add_argument("--train_dir", "--train-dir", default=None)
    parser.add_argument("--validation_dir", "--validation-dir", default=None)
    parser.add_argument("--use_pt_imagefolder", "--use-pt-imagefolder", action="store_true")
    parser.add_argument("--train_val_split", "--train-val-split", type=float, default=0.1)
    parser.add_argument("--split_seed", "--split-seed", type=int, default=42,
                        help="Only controls a missing validation split; never use the model seed here.")
    parser.add_argument("--checkpoint", "--init_checkpoint", "--init-checkpoint", default=None,
                        help="Complete checkpoint.pt (or directory containing it), including classifier.")
    parser.add_argument("--allow_incomplete_legacy_diagnostic", "--allow-incomplete-legacy-diagnostic", action="store_true",
                        help="DIAGNOSTIC ONLY: load old student.pt with an untrained head; invalid for paper results.")
    parser.add_argument("--mixed_precision", "--mixed-precision", choices=("no", "fp16", "bf16"), default="bf16")
    parser.add_argument("--num_workers", "--num-workers", type=int, default=0)
    parser.add_argument("--num_student_layers", "--num-student-layers", type=int, default=None)
    parser.add_argument("--student_l_pad", type=int, default=0)
    parser.add_argument("--student_r_pad", type=int, default=0)
    parser.add_argument("--train_module", choices=("all", "student", "adapter"), default="all")
    parser.add_argument("--use_pretrained_torch_model", action="store_true")
    parser.add_argument("--ignore_mismatched_sizes", action=argparse.BooleanOptionalAction, default=True)
    return parser


def set_random_seed(seed):  # 1 个用法
    random.seed(seed)
    from accelerate.utils import set_seed
    set_seed(seed)

def load_data(args : (copy) ):  # 7 用法
    """Use the repository loader/transforms with an explicit, fixed split seed."""
    from offsite_tuning.prepare_dataset import get_raw_dataset
    options = copy.copy(args)
    # Stop the old loader's unseeded fallback; split explicitly after loading.
    requested_split = options.train_val_split
    options.train_val_split = 0
    options.use_pseudo_logits = False
    options.use_pseudo_target = False
    if options.use_pt_imagefolder:
        # The old ImageFolder branch loads HF rows but later accesses the
        # torchvision-only `.classes` attribute. Keep its transforms, fix only
        # that interface in this isolated runtime.
        from datasets import load_dataset
        from offsite_tuning.prepare_dataset import get_img_size, get_normalize, get_transforms
        if not options.train_dir:
            raise ValueError("--use_pt_imagefolder requires --train_dir containing train/test subdirectories.")
        loaded = load_dataset("imagefolder", data_dir=options.train_dir, trust_remote_code=True)
        label_feature = loaded["train"].features.get("labels", loaded["train"].features.get("label"))
        names = getattr(label_feature, "names", None)
        if names is None:
            raise ValueError("ImageFolder needs ClassLabel class names.")
        size, normalize = get_img_size(options), get_normalize(options)
        train_transform, val_transform = get_transforms(options, normalize, size)
        bundle = {"dataset": loaded, "labels": list(names), "train_transforms": train_transform,
                  "val_transforms": val_transform, "size": size, "normalize": normalize,
                  "id2label": {str(i): x for i, x in enumerate(names)},
                  "label2id": {x: str(i) for i, x in enumerate(names)}}
    else:
        bundle = get_raw_dataset(options, logging.getLogger(__name__))
    dataset = bundle["dataset"]
    if "validation" not in dataset:
        if "test" in dataset:
            dataset["validation"] = dataset["test"]
        elif requested_split and 0 < requested_split < 1:
            split = dataset["train"].train_test_split(test_size=requested_split, seed=args.split_seed)
            dataset["train"], dataset["validation"] = split["train"], split["test"]
        else:
            raise ValueError("A separate validation/test split or --train_val_split in (0,1) is required.")
    for split_name in list(dataset):
        names = dataset[split_name].column_names
        renames = {}
        if "image" not in names and "img" in names:
            renames["img"] = "image"
        if "labels" not in names and "label" in names:
            renames["label"] = "labels"
        if renames:
            dataset[split_name] = dataset[split_name].rename_columns(renames)
    return bundle

def resolve_checkpoint(path):  # 6 用法
    path = Path(path).expanduser().resolve()
    return path / "checkpoint.pt" if path.is_dir() else path


def sha256_file(path):  # 1 个用法
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _torch_load(path):  # 1 个用法
    import torch
    # Only locally produced, trusted checkpoints should be supplied. Explicit
    # weights_only=False retains compatibility across old/new server PyTorch.
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")

def load_complete_checkpoint(path, *, labels=None, model_name_or_path=None):  # 3 用法
    path = resolve_checkpoint(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing complete checkpoint: {path}. Old student.pt is incomplete; rebuild via 00_prepare_baseline.")
    payload = _torch_load(path)
    if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("Refusing incomplete/legacy checkpoint: expected full model state including classifier and metadata. Rebuild via 00_prepare_baseline.")
    if not isinstance(payload.get("model_state"), dict) or not payload["model_state"]:
        raise ValueError("Empty/missing complete model_state.")
    metadata = payload.get("metadata", {})
    if not metadata.get("complete_model", False) or not metadata.get("classifier_saved", False):
        raise ValueError("Checkpoint metadata does not certify a complete model and classifier.")
    if labels is not None and metadata.get("labels") != list(labels):
        raise ValueError("Checkpoint class meaning/order differs from this dataset. Outputs cannot be subtracted.")
    if model_name_or_path is not None:
        # Allow the same pretrained model to be mounted under another directory.
        recorded = metadata.get("model_name_or_path", "")
        if Path(recorded).name != Path(model_name_or_path).name:
            raise ValueError(f"Model architecture/source mismatch: {recorded}. vs {model_name_or_path}")
    return payload

def load_model(args, accelerator, labels):  # 9 个用法 (6 个动态)
    from offsite_tuning.prepare_model import get_model
    options = copy.copy(args)
    options.load_student = None
    options.student_layer_selection_strategy = "uniform"
    options.magnitude_pruning_ratio = 0.0
    options.weight_quantization_bits = None
    options.freeze_bottom = False
    checkpoint = getattr(args, "checkpoint", None)
    payload = None
    diagnostic = False
    if checkpoint:
        resolved = resolve_checkpoint(checkpoint)
        if resolved.exists() and resolved.name != "student.pt":
            payload = load_complete_checkpoint(resolved, labels=labels, model_name_or_path=args.model_name_or_path)
        elif getattr(args, "allow_incomplete_legacy_diagnostic", False):
            legacy = Path(checkpoint)
            legacy = legacy / "student.pt" if legacy.is_dir() else legacy
            if legacy.name != "student.pt" or not legacy.exists():
                raise ValueError("Legacy diagnostic accepts only an existing student.pt or its parent directory.")
            if "clip" not in args.model_name_or_path.lower():
                raise ValueError("Legacy head-missing diagnostic is implemented only for the original CLIP format.")
            options.load_student = str(legacy.parent)
            diagnostic = True
            logging.warning("INCOMPLETE LEGACY DIAGNOSTIC: classifier is random; all outputs are invalid for rebuttal conclusion.")
        else:
            raise ValueError("Refusing old student.pt / missing complete checkpoint.pt: the old file omitted the classifier. Rebuild 00_prepare_baseline or explicitly request diagnostic-only loading.")
    result = get_model(options, accelerator, labels=labels,
                       id2label={str(i): x for i, x in enumerate(labels)},
                       label2id={x: str(i) for i, x in enumerate(labels)})
    model = result["model"]
    # The old loader mixes fp16 backbone and fp32 head. Accelerate autocast needs
    # a coherent fp32 master model, particularly for bf16 training.
    model.float()
    if payload is not None:
        expected = model.state_dict()
        actual = payload["model_state"]
        if set(expected) != set(actual):
            raise ValueError("Complete checkpoint parameter keys differ (including head); use identical layer/pad configuration.")
        model.load_state_dict(actual, strict=True)
    model.review_checkpoint_metadata = payload["metadata"] if payload else {}
    model.review_incomplete_legacy_diagnostic = diagnostic
    return model

def classification_logits(model_or_output, pixel_values=None):  # 4 用法
    """Accept either (model, pixels) or an already computed model output."""
    output = model_or_output(pixel_values) if pixel_values is not None else model_or_output
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, dict):
        return output["logits"]
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def make_collator():  # 1 个用法
    def collate(examples):
        import torch
        return {"pixel_values": torch.stack([x["pixel_values"] for x in examples]),
                "labels": torch.tensor([int(x["labels"]) for x in examples], dtype=torch.long),
                "id": torch.tensor([int(x.get("id", i)) for i, x in enumerate(examples)], dtype=torch.long)}
    return collate

def build_data_loader(dataset, transform, batch_size, shuffle=False, num_workers=0, seed=42):
    import torch
    from torch.utils.data import DataLoader, Dataset

    class TransformedRows(Dataset):
        def __len__(self):
            return len(dataset)

        def __getitem__(self, index):
            row = dataset[index]
            if "pixel_values" in row:
                return {"pixel_values": row["pixel_values"], "labels": int(row["labels"]), "id": int(row.get("id", index))}
            image = row["image"].convert("RGB")
            return {"pixel_values": transform(image), "labels": int(row["labels"]), "id": int(row.get("id", index))}

    generator = torch.Generator().manual_seed(seed)
    return DataLoader(TransformedRows(), batch_size=batch_size, shuffle=shuffle,
                      collate_fn=make_collator(), num_workers=num_workers, generator=generator)

def save_complete_checkpoint(path, model, metadata, accelerator):
    import torch
    if getattr(accelerator.unwrap_model(model), "review_incomplete_legacy_diagnostic", False):
        raise ValueError("Diagnostic model with missing legacy head cannot become a formal complete checkpoint.")
    state = accelerator.get_state_dict(model)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        full_metadata = dict(metadata, complete_model=True, classifier_saved=True)
        torch.save({"format": CHECKPOINT_FORMAT,
                    "model_state": {key: value.detach().cpu() for key, value in state.items()},
                    "metadata": full_metadata}, path)
        path.with_name("metadata.json").write_text(json.dumps(full_metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    accelerator.wait_for_everyone()
    return str(Path(path).resolve())