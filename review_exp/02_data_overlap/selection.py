"""Pure-Python overlap construction: keep class counts and fixed total size."""
import random
from collections import defaultdict


def select_overlap(base_labels, extra_labels, overlap, data_seed=2026):
    if not 0 <= overlap <= 1:
        raise ValueError("Overlap must be in [0,1].")
    if not base_labels:
        raise ValueError("Victim D cannot be empty.")
    base_by_class, extra_by_class = defaultdict(list), defaultdict(list)
    for index, label in enumerate(base_labels):
        base_by_class[int(label)].append(index)
    for index, label in enumerate(extra_labels):
        extra_by_class[int(label)].append(index)
    if set(extra_by_class) - set(base_by_class):
        raise ValueError("Extra pool has class IDs absent from victim D.")
    rng = random.Random(data_seed)
    base, extra, counts = [], [], {}
    for label, candidates in sorted(base_by_class.items()):
        n_shared = int(len(candidates) * overlap + 0.5)
        n_extra = len(candidates) - n_shared
        pool = extra_by_class[label]
        if len(pool) < n_extra:
            raise ValueError(f"Extra pool class {label} needs {n_extra} ground-truth rows, has {len(pool)}.")
        base.extend(rng.sample(candidates, n_shared))
        extra.extend(rng.sample(pool, n_extra))
        counts[str(label)] = {"total": len(candidates), "shared_D": n_shared, "extra_Dprime": n_extra}
    return {"base_indices": sorted(base), "extra_indices": sorted(extra), "class_counts": counts,
            "requested_overlap": overlap, "actual_overlap": len(base) / len(base_labels),
            "total_count": len(base_labels), "data_seed": data_seed}

def validate_pool_provenance(value, labels):  # 3 个用法 (3 个动态)
    if not isinstance(value, dict):
        raise ValueError("Pool provenance must be a JSON object.")
    if not isinstance(value.get("source"), str) or not value["source"].strip():
        raise ValueError("Describe the actual extra pool source.")
    if value.get("label_source") != "ground_truth":
        raise ValueError("Independent Dprime requires real ground-truth labels, never V pseudo-labels.")
    for key in ("in_distribution", "victim_disjoint", "not_test_data"):
        if value.get(key) is not True:
            raise ValueError(f"Pool provenance must explicitly attest {key}=true.")
    if value.get("class_names") != list(labels):
        raise ValueError("Pool class names/order must exactly match victim task.")
    return value