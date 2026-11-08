#!/usr/bin/env python3
"""Freeze and evaluate the Holmes MLP5 verifier without query leakage.

All numerical/ML dependencies are imported lazily, so ``--help`` also works on
the handover machine without the server's PyTorch environment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys


SCHEMA_VERSION = 1


def _numpy():  # 11 用法
    import numpy as np
    return np


def read_json(path):  # 6 用法
    return json.loads(Path(path).read_text(encoding="utf-8"))

def write_json(path, value):  # 8 用法
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")


def array_digest(array):  # 3 用法
    """Hash dtype, shape and bytes; unlike ZIP bytes this is timestamp stable."""
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()

def load_outputs(path):
    """Load only pickle-free, fully identified output matrices."""
    np = _numpy()
    with np.load(path, allow_pickle=False) as data:
        missing = {"logits", "labels", "query_ids", "class_names"} - set(data.files)
        if missing:
            raise ValueError(f"{path}: missing NPZ fields {sorted(missing)}")
        result = {key: data[key].copy() for key in
                  ("logits", "labels", "query_ids", "class_names")}
        result["metadata"] = {}
        if "metadata_json" in data.files:
            if data["metadata_json"].ndim != 0:
                raise ValueError(f"{path}: metadata_json must be a scalar JSON string")
            result["metadata"] = json.loads(str(data["metadata_json"].item()))
            if not isinstance(result["metadata"], dict):
                raise ValueError(f"{path}: metadata_json must encode an object")
    logits, labels = result["logits"], result["labels"]
    ids, classes = result["query_ids"], result["class_names"]
    if logits.ndim != 2 or logits.dtype != np.float32 or not np.isfinite(logits).all():
        raise ValueError(f"{path}: logits must be finite float32 [N,K]")
    n, k = logits.shape
    if n == 0 or k < 4:
        raise ValueError(f"{path}: require N>0 and K>=4 for the original MLP5")
    if labels.shape != (n,) or labels.dtype.kind not in "iu":
        raise ValueError(f"{path}: labels must be integer [N]")
    if (labels < 0).any() or (labels >= k).any():
        raise ValueError(f"{path}: labels lie outside the output class space")
    if ids.shape != (n,) or ids.dtype.kind != "U" or len(set(ids.tolist())) != n:
        raise ValueError(f"{path}: query_ids must be unique Unicode strings [N]")
    if classes.shape != (k,) or classes.dtype.kind != "U" or len(set(classes.tolist())) != k:
        raise ValueError(f"{path}: class_names must be unique Unicode strings [K]")
    if any(not str(x) for x in ids) or any(not str(x) for x in classes):
        raise ValueError(f"{path}: empty query/class identifiers are not allowed")
    result["path"] = str(Path(path).resolve())
    return result


def assert_aligned(left, right):
    """No truncating, broadcasting or guessing mappings between class spaces."""
    np = _numpy()
    for field in ("query_ids", "labels", "class_names"):
        if not np.array_equal(left[field], right[field]):
            raise ValueError(f"{left['path']} and {right['path']}: {field} differ; "
                             "recollect the same ordered queries/classes, do not truncate or guess alignment")
    if left["logits"].shape != right["logits"].shape:
        raise ValueError("logit shapes differ")


def _softmax(value):
    np = _numpy()
    value = value - value.max(axis=1, keepdims=True)
    exponent = np.exp(value)
    return exponent / exponent.sum(axis=1, keepdims=True)


def differences(model, poison, config):
    assert_aligned(model, poison)
    a, b = model["logits"], poison["logits"]
    if config["softmax"]:
        a, b = _softmax(a), _softmax(b)
    return a - b

def normalize_rows(value : {__truediv__} ):  # 2 用法
    np = _numpy()
    norm = np.linalg.norm(value, axis=1, keepdims=True)
    return value / np.where(norm == 0, 1.0, norm)


def transform(value, config, scaler):  # 8 个用法 (5 个动态)
    np = _numpy()
    if config["norm"]:
        value = normalize_rows(value)
    if config["scale"]:
        value = (value - scaler["center"]) / scaler["scale"]
    return np.asarray(value, dtype=np.float32)


def _torch_device(requested):  # 3 用法
    import torch
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; use the configured server environment or --device cpu")
    return torch, device

def make_mlp(k, torch):  # 3 用法
    """Exact stealing_verification.mlp.MLP5 layers and state_dict keys."""
    class MLP5(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Sequential(
                torch.nn.Linear(k, k // 2), torch.nn.ReLU(inplace=True),
                torch.nn.Linear(k // 2, k // 4), torch.nn.ReLU(inplace=True),
                torch.nn.Linear(k // 4, 2), torch.nn.Tanh())

        def forward(self, value):
            return self.linear(value)
    return MLP5()


def _seed(seed, torch):  # 1 个用法
    np = _numpy()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _load_state(path, torch, device):  # 2 用法
    # Load only tensor state dictionaries, never arbitrary legacy Python objects.
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError as exc:
        raise RuntimeError("This utility requires torch.load(weights_only=True); update the "
                           "server PyTorch or export a tensor-only state_dict with a supported version") from exc
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    return state


def _protocol(args):  # 2 用法
    if not 0 < args.alpha < 1 or not math.isfinite(args.tau) or not 0 <= args.tau <= 1:
        raise ValueError("alpha must be in (0,1), tau must be in [0,1]")
    if args.sample_num < 2 or args.repeats < 1 or args.query_seed < 0:
        raise ValueError("sample-num>=2, repeats>=1, query-seed>=0 are required")
    return {"primary": args.primary, "alpha": args.alpha, "tau": args.tau,
            "sample_num": args.sample_num, "repeats": args.repeats,
            "query_seed": args.query_seed, "alternative": "greater",
            "decision_rule": "p_value < alpha AND delta_mu > tau",
            "repeated_harmonic_is_calibrated": False}

def _base_manifest(reference, poison, config, protocol, splits, source):  # 2 用法
    return {"schema_version": SCHEMA_VERSION, "architecture": "MLP5",
            "input_size": len(reference["class_names"]),
            "class_names": reference["class_names"].tolist(),
            "query_ids": reference["query_ids"].tolist(),
            "labels": reference["labels"].tolist(),
            "poison_logits_sha256": array_digest(poison["logits"]),
            "poison_metadata": poison["metadata"],
            "preprocessing": config, "protocol": protocol, "splits": splits,
            "source": source}
def fit(args):
    np = _numpy()
    victim, benign, poison = (load_outputs(path) for path in
                              (args.victim, args.benign, args.poison))
    assert_aligned(victim, benign)
    assert_aligned(victim, poison)
    protocol = _protocol(args)
    if not 0 < args.train_fraction < 1 or not 0 < args.validation_fraction < 1:
        raise ValueError("split fractions must lie in (0,1)")
    if args.train_fraction + args.validation_fraction >= 1:
        raise ValueError("train_fraction + validation_fraction must be < 1")
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        raise ValueError("epochs, batch-size and lr must be positive")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"{output}: use a new empty directory to avoid overwriting a frozen verifier")
    n = len(victim["query_ids"])
    order = np.random.default_rng(args.seed).permutation(n)
    n_train, n_val = int(n * args.train_fraction), int(n * args.validation_fraction)
    split_indices = {"train": order[:n_train], "validation": order[n_train:n_train+n_val],
                     "holdout": order[n_train+n_val:]}
    if min(len(x) for x in split_indices.values()) < 1:
        raise ValueError("all three query splits must be nonempty")
    if len(split_indices["holdout"]) < args.sample_num:
        raise ValueError(f"holdout contains {len(split_indices['holdout'])} queries, "
                         f"fewer than sample-num={args.sample_num}; collect more queries or preregister a smaller sample")
    config = {"norm": args.norm, "scale": args.scale, "softmax": args.softmax}
    vp, bp = differences(victim, poison, config), differences(benign, poison, config)
    train_idx = split_indices["train"]
    scaler = {"center": np.zeros(vp.shape[1]), "scale": np.ones(vp.shape[1])}
    if config["scale"]:
        from sklearn.preprocessing import RobustScaler
        train_values = np.concatenate([vp[train_idx], bp[train_idx]])
        if config["norm"]:
            train_values = normalize_rows(train_values)
        fitted_scaler = RobustScaler().fit(train_values)
        scaler = {"center": fitted_scaler.center_, "scale": fitted_scaler.scale_}
    prepared = {}
    for name in ("train", "validation"):
        idx = split_indices[name]
        values = np.concatenate([vp[idx], bp[idx]])
        labels = np.concatenate([np.ones(len(idx)), np.zeros(len(idx))]).astype(np.int64)
        prepared[name] = transform(values, config, scaler), labels
    torch, device = _torch_device(args.device)
    _seed(args.seed, torch)
    model = make_mlp(vp.shape[1], torch).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    train_x, train_y = prepared["train"]
    train_data = torch.utils.data.TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                         num_workers=0, generator=torch.Generator().manual_seed(args.seed))
    val_x, val_y = (torch.from_numpy(value).to(device) for value in prepared["validation"])
    best_key, best_state, best_epoch, history = None, None, None, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(y)
        model.eval()
        with torch.no_grad():
            logits = model(val_x)
            val_loss = float(torch.nn.functional.cross_entropy(logits, val_y).item())
            val_acc = float((logits.argmax(1) == val_y).float().mean().item())
        history.append({"epoch": epoch, "train_loss": total_loss / len(train_y),
                        "validation_loss": val_loss, "validation_accuracy": val_acc})
        key = (val_acc, -val_loss)
        if best_key is None or key > best_key:
            best_key, best_epoch = key, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"epoch={epoch} validation_accuracy={val_acc:.6f} validation_loss={val_loss:.6f}", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, output / "classifier.pt")
    np.savez(output / "preprocessing.npz", **scaler)
    splits = {name: victim["query_ids"][idx].tolist() for name, idx in split_indices.items()}
    source = {"kind": "fit_V_B_P_only", "victim": victim["path"], "benign": benign["path"],
              "poison": poison["path"], "victim_metadata": victim["metadata"],
              "benign_metadata": benign["metadata"], "seed": args.seed,
              "epochs": args.epochs, "best_epoch": best_epoch,
              "selection": "maximum validation accuracy; tie: minimum validation loss",
              "holdout_used_for_selection": False,
              "optimizer": {"name": "SGD", "lr": args.lr,
                            "momentum": args.momentum, "weight_decay": args.weight_decay}}
    write_json(output / "verifier.json", _base_manifest(victim, poison, config, protocol, splits, source))
    write_json(output / "training_history.json", history)
    print(f"Frozen verifier: {output.resolve()} (holdout={len(splits['holdout'])}, best_epoch={best_epoch})")


def _read_ids(path):  # 2 用法
    value = read_json(path)
    if not isinstance(value, list) or not value or not all(isinstance(x, str) and x for x in value):
        raise ValueError(f"{path}: expected a nonempty JSON list of query IDs")
    if len(set(value)) != len(value):
        raise ValueError(f"{path}: duplicate query IDs")
    return value

def import_legacy(args):
    """Import an existing C only when provenance can rule out query leakage."""
    np = _numpy()
    reference, poison = load_outputs(args.reference_query), load_outputs(args.poison)
    assert_aligned(reference, poison)
    protocol = _protocol(args)
    used, holdout = _read_ids(args.training_query_ids), _read_ids(args.holdout_query_ids)
    # "training" includes every query used for fitting, preprocessing, tuning or checkpoint selection.
    if set(used) & set(holdout):
        raise ValueError("legacy training/tuning IDs overlap holdout; recollect genuinely unseen queries")
    if not all(re.fullmatch(r"sha256:[0-9a-f]{64}", identifier)
               for identifier in used + holdout + reference["query_ids"].tolist()):
        raise ValueError("Legacy C provenance requires the same sha256:<raw RGB image digest> IDs for "
                         "training/tuning, holdout and reference; convert historical indices before import.")
    available = set(reference["query_ids"].tolist())
    if not set(holdout) <= available:
        raise ValueError("holdout IDs must occur in reference-query NPZ")
    if len(holdout) < args.sample_num:
        raise ValueError("not enough holdout queries for the frozen sample-num")
    document = read_json(args.preprocessing_json)
    if not all(type(document.get(key)) is bool for key in ("norm", "scale", "softmax")):
        raise ValueError("preprocessing-json must explicitly contain boolean norm, scale and softmax")
    config = {key: document[key] for key in ("norm", "scale", "softmax")}
    k = len(reference["class_names"])
    scaler = {"center": np.zeros(k), "scale": np.ones(k)}
    if config["scale"]:
        if not document.get("scaler_npz"):
            raise ValueError("scale=true requires original training-fitted scaler_npz with center and scale arrays")
        scaler_path = Path(document["scaler_npz"])
        if not scaler_path.is_absolute():
            scaler_path = Path(args.preprocessing_json).resolve().parent / scaler_path
        with np.load(scaler_path, allow_pickle=False) as saved:
            scaler = {key: saved[key].copy() for key in ("center", "scale")}
        if any(x.shape != (k,) or not np.isfinite(x).all() for x in scaler.values()) or (scaler["scale"] <= 0).any():
            raise ValueError("invalid original scaler; expected finite center[K], positive scale[K]")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("output must be a new empty verifier directory")
    torch, device = _torch_device(args.device)
    model = make_mlp(k, torch).to(device)
    model.load_state_dict(_load_state(args.checkpoint, torch, device), strict=True)
    output.mkdir(parents=True, exist_ok=True)
    torch.save({key: value.cpu() for key, value in model.state_dict().items()}, output / "classifier.pt")
    np.savez(output / "preprocessing.npz", **scaler)
    splits = {"train_and_tuning": used, "validation": [], "holdout": holdout}
    source = {"kind": "import_legacy", "checkpoint": str(Path(args.checkpoint).resolve()),
              "training_query_ids_evidence": str(Path(args.training_query_ids).resolve()),
              "holdout_query_ids_evidence": str(Path(args.holdout_query_ids).resolve()),
              "preprocessing_evidence": str(Path(args.preprocessing_json).resolve()),
              "holdout_used_for_selection": False,
              "warning": "IDs/scaler provenance is supplied by the experimenter; no filename-based guessing"}
    write_json(output / "verifier.json", _base_manifest(reference, poison, config, protocol, splits, source))
    print(f"Imported frozen verifier: {output.resolve()}")

def _one_sided_p(a, b, paired=False):  # 2 用法
    np = _numpy()
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    delta = float(np.mean(a - b))
    # Continuous-limit result for the degenerate constant distributions.
    degenerate = np.ptp(a - b) == 0 if paired else (np.ptp(a) == 0 and np.ptp(b) == 0)
    if degenerate:
        return (0.0 if delta > 0 else 1.0), True
    from scipy import stats
    test = stats.ttest_rel(a, b, alternative="greater") if paired else stats.ttest_ind(
        a, b, alternative="greater", equal_var=False)
    p = float(test.pvalue)
    if not math.isfinite(p):
        raise ValueError("non-finite t-test p-value; inspect saved query probabilities")
    return p, False

def compute_statistics(probabilities, query_ids, protocol):
    """Same per-query class-1 vs class-0 Welch comparison as the old notebook."""
    np = _numpy()
    n = len(query_ids)
    if n < protocol["sample_num"]:
        raise ValueError("fewer held-out queries than sample_num; refusing sampling with replacement")
    records = []
    for repeat in range(protocol["repeats"]):
        seed = protocol["query_seed"] + repeat
        indices = random.Random(seed).sample(range(n), protocol["sample_num"])
        a, b = probabilities[indices, 1], probabilities[indices, 0]
        # H0: E[class1 - class0] <= tau. The reported delta stays unshifted.
        welch, welch_degenerate = _one_sided_p(a.astype(np.float64) - protocol["tau"], b)
        paired, paired_degenerate = _one_sided_p(a.astype(np.float64) - protocol["tau"], b, paired=True)
        records.append({"repeat": repeat, "seed": seed,
                        "query_ids": [query_ids[i] for i in indices],
                        "delta_mu": float(np.mean(a.astype(np.float64) - b)),
                        "welch_p_value": welch, "paired_p_value": paired,
                        "degenerate_welch": welch_degenerate,
                        "degenerate_paired": paired_degenerate})
    p_values = [record["welch_p_value"] for record in records]
    harmonic = 0.0 if min(p_values) == 0 else len(p_values) / sum(1.0 / p for p in p_values)
    mean_delta = float(np.mean([record["delta_mu"] for record in records]))
    single = records[0]
    if protocol["primary"] == "legacy_harmonic":
        p_value, delta = harmonic, mean_delta
    else:
        p_value, delta = single["welch_p_value"], single["delta_mu"]
    stolen = p_value < protocol["alpha"] and delta > protocol["tau"]
    return {"primary": {"method": protocol["primary"], "p_value": p_value,
                        "delta_mu": delta, "decision": "stolen" if stolen else "independent",
                        "model_count": 1},
            "legacy_repeated_welch": {"harmonic_mean_p_value": harmonic,
                                      "mean_delta_mu": mean_delta,
                                      "calibrated_nominal_alpha_guarantee": False,
                                      "note": "Overlapping query repeats are descriptive resampling, not independent models."},
            "sensitivity": {"single_preregistered_welch": {"seed": single["seed"],
                                                           "p_value": single["welch_p_value"], "delta_mu": single["delta_mu"]},
                            "single_preregistered_paired": {"seed": single["seed"],
                                                            "p_value": single["paired_p_value"], "delta_mu": single["delta_mu"]},
                            "note": "Sensitivity tests do not replace the frozen primary decision."},
            "repeats": records}

def evaluate(args):  # 2 用法
    np = _numpy()
    root = Path(args.verifier)
    manifest = read_json(root / "verifier.json")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported verifier schema_version")
    suspect, poison = load_outputs(args.suspect), load_outputs(args.poison)
    assert_aligned(suspect, poison)
    for key in ("query_ids", "labels", "class_names"):
        if suspect[key].tolist() != manifest[key]:
            raise ValueError(f"{key} do not exactly match the frozen verifier; recollect all ordered reference queries")
    if array_digest(poison["logits"]) != manifest["poison_logits_sha256"]:
        raise ValueError("poison outputs differ from frozen P; the experiment must keep P fixed")
    with np.load(root / "preprocessing.npz", allow_pickle=False) as data:
        scaler = {key: data[key].copy() for key in ("center", "scale")}
    index = {identifier: i for i, identifier in enumerate(manifest["query_ids"])}
    holdout = manifest["splits"]["holdout"]
    indices = [index[identifier] for identifier in holdout]
    values = differences(suspect, poison, manifest["preprocessing"])[indices]
    values = transform(values, manifest["preprocessing"], scaler)
    torch, device = _torch_device(args.device)
    model = make_mlp(manifest["input_size"], torch).to(device)
    model.load_state_dict(_load_state(root / "classifier.pt", torch, device), strict=True)
    model.eval()
    probabilities = []
    with torch.no_grad():
        for start in range(0, len(values), args.batch_size):
            tensor = torch.from_numpy(values[start:start+args.batch_size]).to(device)
            probabilities.append(torch.softmax(model(tensor), dim=1).cpu().numpy())
    probabilities = np.concatenate(probabilities)
    report = compute_statistics(probabilities, holdout, manifest["protocol"])
    model_id = args.model_id or suspect["metadata"].get("model_id") or Path(args.suspect).stem
    expected = args.expected or suspect["metadata"].get("expected", "unknown")
    if expected not in {"independent", "stolen", "reference", "unknown"}:
        raise ValueError("expected must be independent, stolen, reference or unknown")
    report.update({"schema_version": SCHEMA_VERSION, "model_id": model_id, "expected": expected,
                   "verifier": str(root.resolve()), "verifier_sha256": hashlib.sha256(
                       (root / "verifier.json").read_bytes() + (root / "classifier.pt").read_bytes() +
                       (root / "preprocessing.npz").read_bytes()).hexdigest(),
                   "suspect_path": suspect["path"], "suspect_metadata": suspect["metadata"],
                   "protocol": manifest["protocol"], "holdout_query_count": len(holdout),
                   "task_accuracy_all_queries": float((suspect["logits"].argmax(1) == suspect["labels"]).mean()),
                   "task_validation_accuracy": suspect["metadata"].get("task_validation_accuracy"),
                   "holdout_probabilities": [{"query_id": identifier, "class0": float(prob[0]),
                                              "class1": float(prob[1])}
                                             for identifier, prob in zip(holdout, probabilities)]})
    report["model_level"] = {"model_id": model_id, "expected": expected,
                             **report["primary"],
                             "false_positive": expected == "independent" and report["primary"]["decision"] == "stolen"}
    write_json(args.output, report)
    print(json.dumps(report["model_level"], ensure_ascii=False))
    return report

def summarize_reports(paths):  # 5 用法
    reports = [read_json(path) for path in paths]
    seen, checkpoint_seen, rows, protocol_fingerprints = set(), set(), [], set()
    for report in reports:
        row = report["model_level"]
        key = (report["verifier_sha256"], row["model_id"])
        if key in seen:
            raise ValueError(f"duplicate model ID for the same verifier: {key}; repeats are not additional models")
        identity = report.get("suspect_metadata", {}).get("checkpoint_sha256") or report.get("suspect_path")
        if not identity:
            raise ValueError("report lacks checkpoint_sha256 and suspect_path; cannot audit duplicate models")
        checkpoint_key = (report["verifier_sha256"], identity)
        if checkpoint_key in checkpoint_seen:
            raise ValueError("the same checkpoint/output was renamed as multiple models; do not count query repeats as model seeds")
        checkpoint_seen.add(checkpoint_key)
        seen.add(key)
        protocol_fingerprints.add((report["verifier_sha256"], json.dumps(report["protocol"], sort_keys=True)))
        rows.append({**row, "verifier_sha256": report["verifier_sha256"]})
    if len(protocol_fingerprints) > 1:
        raise ValueError("summary requires one frozen verifier/protocol; summarize cross-dataset directions separately")
    independent = [row for row in rows if row["expected"] == "independent"]
    stolen = [row for row in rows if row["expected"] == "stolen"]
    fp = sum(row["false_positive"] for row in independent)
    tp = sum(row["decision"] == "stolen" for row in stolen)
    return {"schema_version": SCHEMA_VERSION, "models": rows,
            "independent_model_count": len(independent), "false_positive_model_count": fp,
            "model_level_false_positive_rate": fp / len(independent) if independent else None,
            "stolen_model_count": len(stolen), "true_positive_model_count": tp,
            "model_level_true_positive_rate": tp / len(stolen) if stolen else None,
            "excluded_reference_or_unknown_count": len(rows) - len(independent) - len(stolen),
            "note": "Training seeds count as models; repeated query subsets never increase the model denominator."}

def _add_protocol(parser):  # 2 用法
    parser.add_argument("--primary", choices=("legacy_harmonic", "single_welch"), default="legacy_harmonic",
                        help="Frozen primary decision; default reproduces old repeated Welch harmonic aggregation")
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--tau", type=float, default=0.0)
    parser.add_argument("--sample-num", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--query-seed", type=int, default=100)

def build_parser():  # 3 用法
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("fit", help="Fit C using V/B/P only, grouped query train/validation/holdout split")
    for flag in ("victim", "benign", "poison", "output"):
        train.add_argument("--" + flag, required=True)
    train.add_argument("--norm", action=argparse.BooleanOptionalAction, default=True)
    train.add_argument("--scale", action=argparse.BooleanOptionalAction, default=True)
    train.add_argument("--softmax", action=argparse.BooleanOptionalAction, default=False)
    train.add_argument("--train-fraction", type=float, default=0.6)
    train.add_argument("--validation-fraction", type=float, default=0.2)
    train.add_argument("--epochs", type=int, default=300)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--lr", type=float, default=0.01)
    train.add_argument("--momentum", type=float, default=0.9)
    train.add_argument("--weight-decay", type=float, default=5e-4)
    train.add_argument("--batch-size", type=int, default=128)
    train.add_argument("--device", default="auto")
    _add_protocol(train)
    train.set_defaults(func=fit)
    legacy = sub.add_parser("import-legacy", help="Import C only with explicit preprocessing and disjoint query evidence")
    for flag in ("checkpoint", "preprocessing-json", "training-query-ids", "holdout-query-ids",
                 "reference-query", "poison", "output"):
        legacy.add_argument("--" + flag, required=True)
    legacy.add_argument("--device", default="auto")
    _add_protocol(legacy)
    legacy.set_defaults(func=import_legacy)
    test = sub.add_parser("evaluate", help="Evaluate one S with frozen C, P, preprocessing, holdout and thresholds")
    for flag in ("verifier", "suspect", "poison", "output"):
        test.add_argument("--" + flag, required=True)
    test.add_argument("--device", default="auto")
    test.add_argument("--batch-size", type=int, default=256)
    test.add_argument("--model-id")
    test.add_argument("--expected", choices=("independent", "stolen", "reference", "unknown"))
    test.set_defaults(func=evaluate)
    summary = sub.add_parser("summary", help="Summarize independent model false positives; excludes reference/unknown")
    summary.add_argument("--inputs", nargs="+", required=True)
    summary.add_argument("--output", required=True)
    summary.set_defaults(func=lambda args: write_json(args.output, summarize_reports(args.inputs)))
    return parser

def main(argv=None):  # 9 个用法 (5 个动态)
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if getattr(args, "batch_size", 1) < 1:
            raise ValueError("batch-size must be positive")
        args.func(args)
    except (ValueError, KeyError, FileNotFoundError, ImportError, RuntimeError) as exc:
        parser.exit(2, f"error: {exc}\n")

if __name__ == "__main__":
    main()
