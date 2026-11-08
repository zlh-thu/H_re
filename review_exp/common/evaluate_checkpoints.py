"""Export and evaluate every complete checkpoint in a training models.json.

Subprocess arguments are lists; model names and paths are never evaluated as shell code.
"""

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]

def load_models(path):  # 2 用法
    manifest = Path(path).resolve()
    value = json.loads(manifest.read_text())
    entries = value.get("models") if isinstance(value, dict) else value
    if not isinstance(entries, list) or not entries:
        raise ValueError("models.json must be a nonempty list or an object with a nonempty models list.")
    seen = set()
    for entry in entries:
        if not isinstance(entry, dict) or not entry.get("model_id") or not entry.get("checkpoint"):
            raise ValueError("Each model needs model_id, checkpoint and expected.")
        identifier = entry["model_id"]
        if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", identifier):
            raise ValueError("model_id may contain ASCII letters, digits, underscores, dots and hyphens only.")
        if identifier in seen:
            raise ValueError(f"Duplicate model_id: {identifier}")
        seen.add(identifier)
        if entry.get("expected") not in {"independent", "stolen", "reference"}:
            raise ValueError(f"{identifier}: expected must be independent, stolen or reference.")
        checkpoint = Path(entry["checkpoint"])
        if not checkpoint.is_absolute():
            checkpoint = manifest.parent / checkpoint
        entry["checkpoint"] = str(checkpoint.resolve())
        if not isinstance(entry.get("collector_args", []), list) or not all(
                isinstance(x, str) for x in entry.get("collector_args", [])):
            raise ValueError("collector_args must be a list of literal CLI argument strings.")
    return entries

def build_commands(args, entry, output):
    launch = [sys.executable, "-m", "accelerate.commands.launch",
              "--num_processes", str(args.num_processes), "--num_machines", "1",
              "--machine_rank", "0", "--mixed_precision", args.mixed_precision,
              "--main_process_port", str(args.port), "--dynamo_backend", "no"]
    if args.num_processes > 1:
        launch.append("--multi_gpu")
    launch += ["-m", "review_exp.common.collect_outputs", "collect",
               "--model_name_or_path", entry.get("model_name_or_path", args.model_name_or_path),
               "--dataset_name", args.dataset_name,
               "--mixed_precision", args.mixed_precision,
               "--checkpoint", entry["checkpoint"], "--query-manifest", str(Path(args.query_manifest).resolve()),
               "--query-batch-size", str(args.query_batch_size), "--model-id", entry["model_id"],
               "--expected", entry["expected"], "--output", str(output / f"{entry['model_id']}.npz")]
    extras = shlex.split(args.collector_extra_args) + entry.get("collector_args", [])
    # These keys encode experiment identity and may not be overridden by convenience extras.
    protected = {"--checkpoint", "--init_checkpoint", "--init-checkpoint", "--query-manifest", "--query_manifest", "--output",
                 "--model-id", "--model_id", "--expected"}
    if any(token.split("=", 1)[0] in protected for token in extras):
        raise ValueError("collector extra arguments cannot override checkpoint/query/output/model identity.")
    launch.extend(extras)
    evaluate = [sys.executable, "-m", "review_exp.common.verifier", "evaluate",
                "--verifier", str(Path(args.verifier).resolve()),
                "--suspect", str(output / f"{entry['model_id']}.npz"),
                "--poison", str(Path(args.poison).resolve()),
                "--model-id", entry["model_id"], "--expected", entry["expected"],
                "--device", args.device, "--output", str(output / f"{entry['model_id']}.result.json")]
    return launch, evaluate

def main(argv=None):  # 6 个用法 (5 个动态)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", required=True)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--dataset_name", default="./dataset/cifar10.py")
    parser.add_argument("--query-manifest", required=True)
    parser.add_argument("--verifier", required=True)
    parser.add_argument("--poison", required=True, help="The EXACT P.npz bound when fitting/importing C.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-processes", type=int, default=1)
    parser.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="bf16")
    parser.add_argument("--port", type=int, default=29620)
    parser.add_argument("--device", default="auto", help="C inference device; e.g. auto, cuda:0 or cpu.")
    parser.add_argument("--query-batch-size", type=int, default=128)
    parser.add_argument("--collector-extra-args", default="", help="Additional literal runtime flags, parsed by shlex.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.num_processes < 1 or args.query_batch_size < 1 or not 1 <= args.port <= 65535:
        parser.error("num-processes/query-batch-size must be positive; port must be in 1..65535")
    entries = load_models(args.models)
    output = Path(args.output_dir).resolve()
    if not args.dry_run and output.exists() and any(output.iterdir()) and not args.overwrite:
        raise ValueError("Use a new empty output directory, or explicitly pass --overwrite for a rerun.")
    commands = [(entry, *build_commands(args, entry, output)) for entry in entries]
    if args.dry_run:
        for entry, collect, evaluate in commands:
            print(f"[{entry['model_id']}] collect: {shlex.join(collect)}")
            print(f"[{entry['model_id']}] evaluate: {shlex.join(evaluate)}")
        return
    for path in (args.query_manifest, args.poison, Path(args.verifier) / "verifier.json"):
        if not Path(path).is_file():
            raise ValueError(f"Required file does not exist: {path}")
    for entry in entries:
        if not Path(entry["checkpoint"]).exists():
            raise ValueError(f"Missing checkpoint: {entry['checkpoint']}")
    output.mkdir(parents=True, exist_ok=True)
    report_paths, groups = [], {}
    for entry, collect, evaluate in commands:
        subprocess.run(collect, cwd=REPO_ROOT, check=True)
        subprocess.run(evaluate, cwd=REPO_ROOT, check=True)
        result = output / f"{entry['model_id']}.result.json"
        report_paths.append(str(result))
        groups.setdefault(str(entry.get("group", "unspecified")), []).append(str(result))
    from review_exp.common.verifier import summarize_reports, write_json
    write_json(output / "summary.json", summarize_reports(report_paths))
    write_json(output / "summary_by_group.json", {
        group: summarize_reports(paths) for group, paths in groups.items()})
    print(f"Finished {len(entries)} model evaluations: {output / 'summary.json'}")

if __name__ == "__main__":
    main()