"""Top-level package for ts_clustering.

This package provides simple, dependency-light utilities for clustering
univariate time series. The primary entry point is
`ts_clustering.kmeans.TimeSeriesKMeans`.
"""

from .kmeans import TimeSeriesKMeans

__all__ = [
    "TimeSeriesKMeans",
]

__version__ = "0.1.0"


