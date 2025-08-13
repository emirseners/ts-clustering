"""
Top-level for simgon clustering package.

This package provides pattern-oriented behavior clustering for time-series data.
End users should use the main functions: read_time_series, perform_clustering, and related utilities.
"""

from .clusterer import (
    read_time_series,
    simulate_from_vensim,
    perform_clustering,
    Cluster
)

__all__ = [
    "read_time_series",
    "simulate_from_vensim", 
    "perform_clustering",
    "Cluster"
]

__version__ = "0.1.0"