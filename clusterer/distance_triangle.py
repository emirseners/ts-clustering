import numpy as np
from scipy.spatial.distance import pdist
from typing import Tuple, List, Dict

def distance_triangle(data: np.ndarray) -> Tuple[np.ndarray, List[Tuple[Dict[str, str], np.ndarray]]]:
    """
    Calculate pairwise triangle distances between all data sequences.

    Parameters
    ----------
    data : np.ndarray
        2D array of shape (n_samples, n_features) with equal-length sequences.

    Returns
    -------
    dRow : np.ndarray
        Condensed distance matrix as 1D array of length n_samples * (n_samples - 1) / 2.
    runLogs : List[Tuple[Dict[str, str], np.ndarray]]
        List of (index_dict, sequence) tuples for tracking original data.
    """
    runLogs = [({'Index': str(i)}, data[i]) for i in range(len(data))]

    cosine_distances = pdist(data, metric='cosine')
    dRow = 1 - cosine_distances
    
    return dRow, runLogs