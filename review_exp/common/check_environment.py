"""Read-only server preflight; does not install packages or download models."""
import argparse
import importlib
import json
from pathlib import Path
import sys

def main():  # 6 个用法 (5 个动态)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name_or_path")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    errors, versions = [], {}
    for package in ("torch", "torchvision", "transformers", "datasets", "accelerate", "numpy", "scipy", "sklearn", "PIL"):
        try:
            module = importlib.import_module(package)
            versions[package] = getattr(module, "__version__", "unknown")
        except Exception as exc:
            errors.append(f"{package}: {type(exc).__name__}: {exc}")
    if "torch" in versions:
        import torch
        versions["cuda_available"] = torch.cuda.is_available()
        versions["cuda_runtime"] = torch.version.cuda
        versions["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        if args.require_cuda and not torch.cuda.is_available():
            errors.append("CUDA is required by this invocation but is not available.")
    if args.model_name_or_path:
        source = Path(args.model_name_or_path)
        if not source.is_dir():
            errors.append(f"Foundation model directory does not exist: {source}")
        else:
            versions["foundation_files"] = sorted(p.name for p in source.iterdir() if p.is_file())
            if not (source / "config.json").exists():
                errors.append("Missing foundation config.json")
    print(json.dumps({"python": sys.executable, "versions": versions, "errors": errors}, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()