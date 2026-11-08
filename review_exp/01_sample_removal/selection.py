"""Pure-Python sample selection; indices always refer to fixed original D."""
import math
import random
from collections import Counter

def low_loss_keep(labels, losses, drop_fraction=0.5):  # 3 个用法 (3 个动态)
    if not 0 <= drop_fraction < 1:
        raise ValueError("drop_fraction must be in [0,1).")
    if len(labels) != len(losses) or not labels:
        raise ValueError("Need one loss per original training row.")
    if any(not math.isfinite(float(value)) for value in losses):
        raise ValueError("Loss values must be finite.")
    drop_count = int(len(labels) * drop_fraction)
    dropped = set(sorted(range(len(labels)), key=lambda index: (float(losses[index]), index))[:drop_count])
    return [index for index in range(len(labels)) if index not in dropped]


def class_counts(labels, indices):  # 5 个用法 (5 个动态)
    return {str(label): count for label, count in sorted(Counter(int(labels[index]) for index in indices).items())}


def random_keep_matching_classes(labels, reference_keep, seed):  # 3 个用法 (3 个动态)
    if len(set(reference_keep)) != len(reference_keep) or any(type(x) is not int or x < 0 or x >= len(labels) for x in reference_keep):
        raise ValueError("Reference indices must be unique, valid original D indices.")
    counts = Counter(int(labels[index]) for index in reference_keep)
    by_class = {}
    for index, label in enumerate(labels):
        by_class.setdefault(int(label), []).append(index)
    rng = random.Random(seed)
    result = []
    for label, indices in sorted(by_class.items()):
        result.extend(rng.sample(indices, counts.get(label, 0)))
    return sorted(result)

def validate_loss_mapping(value, num_train):  # 2 个用法 (2 个动态)
    """Accept dense losses, id/loss rows, or the new losses.json payload."""
    if isinstance(value, dict) and "losses" in value:
        if "num_train" in value and value["num_train"] != num_train:
            raise ValueError("Loss file num_train differs from fixed D.")
        value = value["losses"]
    if isinstance(value, list) and value and isinstance(value[0], dict):
        pairs = [(int(row["id"]), float(row["loss"])) for row in value]
        if len({key for key, _ in pairs}) != len(pairs):
            raise ValueError("Duplicate loss row IDs.")
        value = dict(pairs)
    if isinstance(value, dict):
        normalized = {int(key): float(loss) for key, loss in value.items()}
        if set(normalized) != set(range(num_train)):
            raise ValueError("Loss file must contain exactly indices 0..N-1 from fixed D.")
        value = [normalized[index] for index in range(num_train)]
    if not isinstance(value, (list, tuple)) or len(value) != num_train:
        raise ValueError("Loss file length differs from fixed D.")
    return [float(loss) for loss in value]