import json
from pathlib import Path

import numpy as np
from sklearn import metrics


def read_predicted_probs(path: Path) -> dict[str, tuple[int]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.loads(f.read())


def save_predicted_probs(predicted_probs: dict[str, tuple[int]], path: Path):
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(predicted_probs))


def calculate_metric(ground_truth_label_probs: dict[str, tuple[int]], predicted_probs: dict[str, tuple[int]]) -> float:
    diff = set(ground_truth_label_probs.keys()) ^ set(predicted_probs.keys())
    assert len(diff) == 0, f"Keys are different (N={len(diff)}): {diff}"

    keys = list(predicted_probs.keys())
    gt = np.array([
        ground_truth_label_probs[k]
        for k in keys
    ])
    pred = np.array([
        predicted_probs[k]
        for k in keys
    ])    
    return metrics.roc_auc_score(gt, pred)
