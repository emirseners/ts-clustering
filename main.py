from __future__ import annotations

import random
from ts_clustering import TimeSeriesKMeans


def _generate_sine_series(n: int, length: int, noise: float, phase: float) -> list[list[float]]:
    import math

    data: list[list[float]] = []
    for _ in range(n):
        series = [math.sin(2 * math.pi * (i / length) + phase) for i in range(length)]
        series = [x + random.gauss(0.0, noise) for x in series]
        data.append(series)
    return data


def demo() -> None:
    # Create a tiny synthetic dataset: two sine-wave clusters with phase shift
    X = _generate_sine_series(n=10, length=50, noise=0.1, phase=0.0)
    X += _generate_sine_series(n=10, length=50, noise=0.1, phase=1.0)

    model = TimeSeriesKMeans(n_clusters=2, random_state=42)
    labels = model.fit_predict(X)

    counts = {0: 0, 1: 0}
    for lbl in labels:
        counts[lbl] = counts.get(lbl, 0) + 1

    print("Cluster counts:", counts)


if __name__ == "__main__":
    demo()


