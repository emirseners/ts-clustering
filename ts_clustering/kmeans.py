from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import math
import random


def _euclidean_distance_squared(a: list[float], b: list[float]) -> float:
    """Compute squared Euclidean distance between two equal-length sequences.

    Parameters
    ----------
    a : list[float]
        First sequence.
    b : list[float]
        Second sequence.

    Returns
    -------
    float
        The squared Euclidean distance.
    """
    if len(a) != len(b):
        raise ValueError("Sequences must have equal length for Euclidean distance.")
    return sum((x - y) * (x - y) for x, y in zip(a, b))


def _mean_series(series_list: list[list[float]]) -> list[float]:
    """Compute the element-wise mean of a list of equal-length sequences."""
    if not series_list:
        raise ValueError("Cannot compute mean of an empty list.")
    length = len(series_list[0])
    for s in series_list:
        if len(s) != length:
            raise ValueError("All series must have equal length.")
    return [
        sum(s[i] for s in series_list) / float(len(series_list))
        for i in range(length)
    ]


@dataclass
class TimeSeriesKMeans:
    """K-Means clustering for univariate time series with Euclidean distance.

    This is a minimal, dependency-light implementation intended for educational
    and small-scale use. It assumes all input series have the same length.

    Parameters
    ----------
    n_clusters : int
        Number of clusters.
    max_iter : int, optional
        Maximum number of iterations. Default is 100.
    tol : float, optional
        Convergence threshold on centroid movement (squared distance). Default 1e-4.
    random_state : Optional[int], optional
        Random seed for reproducibility.

    Notes
    -----
    - Input format `X` is a list of time series, where each time series is a
      list of floats with equal length.
    - Distance metric is squared Euclidean distance across aligned time points.
    - This implementation uses a simple random initialization.
    """

    n_clusters: int
    max_iter: int = 100
    tol: float = 1e-4
    random_state: Optional[int] = None

    centroids_: Optional[list[list[float]]] = None
    labels_: Optional[list[int]] = None

    def _init_random_centroids(self, X: list[list[float]]) -> list[list[float]]:
        rng = random.Random(self.random_state)
        return [list(ts) for ts in rng.sample(X, self.n_clusters)]

    def fit(self, X: list[list[float]]) -> "TimeSeriesKMeans":
        """Fit the model on a dataset of equal-length time series.

        Parameters
        ----------
        X : list[list[float]]
            Dataset of shape (n_series, series_length).

        Returns
        -------
        TimeSeriesKMeans
            The fitted estimator.
        """
        if len(X) < self.n_clusters:
            raise ValueError("n_clusters must be <= number of time series in X")
        length = len(X[0])
        if any(len(ts) != length for ts in X):
            raise ValueError("All time series must have the same length.")

        self.centroids_ = self._init_random_centroids(X)

        for _ in range(self.max_iter):
            # Assign step
            labels = [self._closest_centroid_index(ts) for ts in X]

            # Update step
            new_centroids: list[list[float]] = []
            for k in range(self.n_clusters):
                members = [X[i] for i, lbl in enumerate(labels) if lbl == k]
                if members:
                    new_centroids.append(_mean_series(members))
                else:
                    # If a cluster lost all members, reinitialize it randomly
                    new_centroids.append(list(self.centroids_[k]))

            # Check convergence
            shift = sum(
                _euclidean_distance_squared(c_old, c_new)
                for c_old, c_new in zip(self.centroids_, new_centroids)
            )
            self.centroids_ = new_centroids
            self.labels_ = labels
            if shift <= self.tol:
                break

        return self

    def predict(self, X: list[list[float]]) -> list[int]:
        """Assign each series in `X` to the nearest learned centroid.

        Parameters
        ----------
        X : list[list[float]]
            Dataset of shape (n_series, series_length).

        Returns
        -------
        list[int]
            Cluster label for each input series.
        """
        if self.centroids_ is None:
            raise RuntimeError("Model is not fitted. Call fit(X) first.")
        length = len(self.centroids_[0])
        if any(len(ts) != length for ts in X):
            raise ValueError("All time series must have the same length as centroids.")
        return [self._closest_centroid_index(ts) for ts in X]

    def fit_predict(self, X: list[list[float]]) -> list[int]:
        """Fit the model to `X` and return the cluster labels."""
        return self.fit(X).labels_  # type: ignore[return-value]

    # ----- helper methods -----
    def _closest_centroid_index(self, ts: list[float]) -> int:
        if self.centroids_ is None:
            raise RuntimeError("Centroids are not initialized.")
        distances = [
            _euclidean_distance_squared(ts, centroid) for centroid in self.centroids_
        ]
        min_idx = min(range(len(distances)), key=lambda i: distances[i])
        return min_idx


