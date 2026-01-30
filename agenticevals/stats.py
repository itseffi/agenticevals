from __future__ import annotations

import random
from typing import Iterable


def bootstrap_ci(values: Iterable[float | bool], *, samples: int = 1000, confidence: float = 0.95, seed: int = 0) -> dict[str, float]:
    rows = [float(value) for value in values]
    if not rows:
        return {"mean": 0.0, "low": 0.0, "high": 0.0, "confidence": confidence, "samples": 0}
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        draw = [rows[rng.randrange(len(rows))] for _ in rows]
        means.append(sum(draw) / len(draw))
    means.sort()
    alpha = 1.0 - confidence
    low_index = max(0, min(len(means) - 1, int((alpha / 2) * len(means))))
    high_index = max(0, min(len(means) - 1, int((1 - alpha / 2) * len(means)) - 1))
    return {
        "mean": round(sum(rows) / len(rows), 6),
        "low": round(means[low_index], 6),
        "high": round(means[high_index], 6),
        "confidence": confidence,
        "samples": samples,
    }
