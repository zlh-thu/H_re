"""Server CPU smoke test with synthetic logits; never use its numbers as paper results."""
import argparse
import json
from pathlib import Path
import tempfile


def main():  # 6 个用法 (5 个动态)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", help="Optional new empty directory; default is a temporary directory.")
    args = parser.parse_args()
    if args.output_dir:
        root = Path(args.output_dir)
        if root.exists() and any(root.iterdir()):
            raise ValueError("Smoke output directory must be empty.")
        root.mkdir(parents=True, exist_ok=True)
        run(root)
    else:
        with tempfile.TemporaryDirectory(prefix="holmes_review_smoke_") as directory:
            run(Path(directory))

def run(root):  # 5个用法 (3个动态)
    import numpy as np
    from review_exp.common import verifier
    rng = np.random.default_rng(42)
    n, k = 600, 10
    ids = np.asarray([f"synthetic:{i}" for i in range(n)])
    labels = np.arange(n) % k
    classes = np.asarray([f"class{i}" for i in range(k)])
    poison = rng.normal(size=(n, k)).astype(np.float32)
    for name, logits in (("P", poison), ("V", poison + 1), ("B", poison - 1)):
        np.savez(root / f"{name}.npz", logits=logits.astype(np.float32), labels=labels,
                 query_ids=ids, class_names=classes,
                 metadata_json=json.dumps({"model_id": "synthetic_" + name, "expected": "reference"}))
    verifier.main(["fit", "--victim", str(root / "V.npz"), "--benign", str(root / "B.npz"),
                  "--poison", str(root / "P.npz"), "--output", str(root / "C"),
                  "--epochs", "3", "--device", "cpu", "--sample-num", "20", "--repeats", "3"])
    common = ["evaluate", "--verifier", str(root / "C"), "--suspect", str(root / "V.npz"),
              "--poison", str(root / "P.npz"), "--device", "cpu", "--model-id", "synthetic_V"]
    verifier.main(common + ["--output", str(root / "first.json")])
    verifier.main(common + ["--output", str(root / "second.json")])
    first = json.loads((root / "first.json").read_text())
    second = json.loads((root / "second.json").read_text())
    if first["primary"] != second["primary"] or first["repeats"] != second["repeats"]:
        raise AssertionError("Frozen verifier reload changed predictions or sampled query subsets.")
    print("PASS: synthetic fit -> save -> reload -> deterministic evaluate. Synthetic numbers are not experiment results.")

if __name__ == "__main__":
    main()