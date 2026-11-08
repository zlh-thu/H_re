import copy
import json
import math
from pathlib import Path

from review_exp.common import runtime


def add_training_arguments(parser):  # 4 用法
    runtime.add_runtime_arguments(parser)
    parser.add_argument("--output_dir", "--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", "--num_train_epochs", dest="epochs", type=int, default=20)
    parser.add_argument("--lr", "--learning_rate", dest="lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", "--per_device_train_batch_size", dest="batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", "--per_device_eval_batch_size", dest="eval_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--classifier_lr_multiplier", type=float, default=10.0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--optimizer", choices=("adamw", "sgd"), default="adamw")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--num_warmup_steps", type=int, default=1000)
    parser.add_argument("--lr_scheduler_type", choices=("cosine", "linear", "constant"), default="cosine")
    parser.add_argument("--max_train_steps", type=int, default=None,
                        help="Optimizer-update budget; overrides epochs. Useful for equal-step sensitivity runs.")
    return parser

def train_model(args, *, dataset_override=None, selection_metadata=None, data_bundle=None):
    import torch
    import torch.nn.functional as functional
    from accelerate import Accelerator
    from transformers import get_scheduler
    from offsite_tuning.prepare_model import get_optimizer

    if args.allow_incomplete_legacy_diagnostic:
        raise ValueError("Training formal review models forbids incomplete legacy diagnostics.")
    if min(args.epochs, args.batch_size, args.eval_batch_size, args.gradient_accumulation_steps) <= 0:
        raise ValueError("epochs, batch sizes and gradient accumulation must be positive.")
    if args.max_train_steps is not None and args.max_train_steps <= 0:
        raise ValueError("max_train_steps must be positive.")
    accelerator = Accelerator(mixed_precision=args.mixed_precision,
                              gradient_accumulation_steps=args.gradient_accumulation_steps)
    runtime.set_random_seed(args.seed)
    bundle = data_bundle if data_bundle is not None else runtime.load_data(args)
    dataset = dataset_override if dataset_override is not None else bundle["dataset"]
    if not len(dataset["train"]):
        raise ValueError("The selected training set is empty.")
    train_loader = runtime.build_data_loader(dataset["train"], bundle["train_transforms"],
                                             args.batch_size, True, args.num_workers, args.seed)
    eval_loader = runtime.build_data_loader(dataset["validation"], bundle["val_transforms"],
                                            args.eval_batch_size, False, args.num_workers, args.seed)
    model = runtime.load_model(args, accelerator, bundle["labels"])
    options = copy.copy(args)
    options.learning_rate = args.lr
    optimizer = get_optimizer(options, model)["optimizer"]
    model, optimizer, train_loader, eval_loader = accelerator.prepare(model, optimizer, train_loader, eval_loader)
    updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    max_steps = args.max_train_steps if args.max_train_steps is not None else args.epochs * updates_per_epoch
    epochs = math.ceil(max_steps / updates_per_epoch)
    # Deliberately step this scheduler exactly once per synchronized optimizer
    # update. Do not prepare it through Accelerate (which may multiply steps).
    scheduler = get_scheduler(args.lr_scheduler_type, optimizer=optimizer,
                              num_warmup_steps=args.num_warmup_steps, num_training_steps=max_steps)

    history = []
    completed = 0
    skipped_updates = 0
    examples_seen = 0
    optimizer.zero_grad()
    for epoch in range(epochs):
        model.train()
        if hasattr(train_loader, "set_epoch"):
            train_loader.set_epoch(epoch)
        loss_total = 0.0
        batches = 0
        for batch in train_loader:
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    logits = runtime.classification_logits(model, batch["pixel_values"])
                    loss = functional.cross_entropy(logits, batch["labels"])
                accelerator.backward(loss)
                optimizer.step()
                if accelerator.sync_gradients and not accelerator.optimizer_step_was_skipped:
                    scheduler.step()
                    completed += 1
                elif accelerator.sync_gradients:
                    skipped_updates += 1
                optimizer.zero_grad()
            loss_total += float(loss.detach())
            batches += 1
            examples_seen += len(batch["labels"]) * accelerator.num_processes
            if completed >= max_steps:
                break
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in eval_loader:
                with accelerator.autocast():
                    prediction = runtime.classification_logits(model, batch["pixel_values"]).argmax(-1)
                prediction, target = accelerator.gather_for_metrics((prediction, batch["labels"]))
                correct += int((prediction == target).sum())
                total += len(target)
        epoch_loss = accelerator.reduce(torch.tensor(loss_total / max(1, batches), device=accelerator.device), reduction="mean")
        record = {"epoch": epoch + 1, "optimizer_steps": completed,
                  "train_loss": epoch_loss, "validation_accuracy": correct / total if total else None,
                  "validation_count": total}
        history.append(record)
        accelerator.print(json.dumps(record))
        if completed >= max_steps:
            break
    if args.max_train_steps is not None and completed < max_steps:
        raise RuntimeError(f"Requested {max_steps} real optimizer updates, completed {completed} with "
                           f"{skipped_updates} skipped updates. No complete checkpoint was saved; "
                           "inspect overflow/precision settings before an equal-step experiment.")
    output_dir = Path(args.output_dir).resolve()
    safe_arguments = {}
    for key, value in vars(args).items():
        try:
            json.dumps(value)
            safe_arguments[key] = value
        except TypeError:
            safe_arguments[key] = repr(value)
    metadata = {"model_name_or_path": args.model_name_or_path, "dataset_name": args.dataset_name,
                "labels": list(bundle["labels"]), "seed": args.seed, "split_seed": args.split_seed,
                "train_count": len(dataset["train"]), "validation_count": len(dataset["validation"]),
                "dataset_train_fingerprint": getattr(bundle["dataset"]["train"], "_fingerprint", None),
                "initialization": "victim_checkpoint" if args.checkpoint else "foundation_model_F",
                "initial_checkpoint": str(runtime.resolve_checkpoint(args.checkpoint)) if args.checkpoint else None,
                "initial_checkpoint_sha256": runtime.sha256_file(runtime.resolve_checkpoint(args.checkpoint)) if args.checkpoint else None,
                "mixed_precision": args.mixed_precision, "master_parameter_dtype": "float32",
                "world_size": accelerator.num_processes,
                "per_device_batch_size": args.batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "effective_batch_size": args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps,
                "optimizer_steps": completed, "epochs_completed": len(history),
                "target_optimizer_steps": max_steps, "skipped_optimizer_updates": skipped_updates,
                "examples_seen_including_distributed_padding": examples_seen,
                "learning_rate": args.lr, "classifier_lr_multiplier": args.classifier_lr_multiplier,
                "num_warmup_steps": args.num_warmup_steps, "scheduler": args.lr_scheduler_type,
                "optimizer": args.optimizer, "weight_decay": args.weight_decay,
                "validation_accuracy": history[-1]["validation_accuracy"],
                "training_arguments": safe_arguments, "selection": selection_metadata or {}}
    checkpoint = runtime.save_complete_checkpoint(output_dir / "checkpoint.pt", model, metadata, accelerator)
    if accelerator.is_main_process:
        (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    accelerator.wait_for_everyone()
    # Sequential multi-model scripts must release each Accelerate wrapper.
    del model, optimizer, scheduler, train_loader, eval_loader
    accelerator.free_memory()
    return checkpoint