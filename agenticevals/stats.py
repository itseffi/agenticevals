from __future__ import annotations

import random
import statistics
from typing import Iterable


def wilson_ci(outcomes: Iterable[bool], *, confidence: float = 0.95) -> dict[str, float]:
    """Wilson score interval for a binomial proportion.

    Preferred over the percentile bootstrap for pass/fail rates: it stays
    non-degenerate at the extremes (a 5/5 run yields roughly [0.566, 1.0] rather
    than a misleading zero-width [1.0, 1.0]) and never leaves the unit interval.
    """
    rows = [1.0 if bool(value) else 0.0 for value in outcomes]
    n = len(rows)
    if n == 0:
        return {"mean": 0.0, "low": 0.0, "high": 0.0, "confidence": confidence, "n": 0}
    p = sum(rows) / n
    z = statistics.NormalDist().inv_cdf(1.0 - (1.0 - confidence) / 2.0)
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)) / denom
    return {
        "mean": round(p, 6),
        "low": round(max(0.0, centre - half), 6),
        "high": round(min(1.0, centre + half), 6),
        "confidence": confidence,
        "n": n,
    }


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
