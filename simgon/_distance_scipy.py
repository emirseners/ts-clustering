import numpy as np
from scipy.spatial.distance import pdist
from typing import Tuple, List, Dict, Any

def _distance_scipy(data: np.ndarray, metric: str = 'euclidean', **kwargs) -> Tuple[np.ndarray, List[Tuple[Dict[str, str], np.ndarray]]]:
    """
    Calculate pairwise distances between all data sequences using scipy's pdist function.
    
    This function provides access to all distance metrics available in scipy.spatial.distance.pdist.

    Parameters
    ----------
    data : np.ndarray
        2D array of shape (n_samples, n_features) with equal-length sequences.
    metric : str, optional
        Distance metric to use. Must be one of the metrics supported by scipy.spatial.distance.pdist.
        Default is 'euclidean'.
    **kwargs : dict
        Additional parameters for specific distance metrics:
        - For 'minkowski': p
        - For 'mahalanobis': VI
        - For 'seuclidean': V
        - For 'wminkowski': w
        - For 'sokalsneath': w

    Returns
    -------
    dRow : np.ndarray
        Condensed distance matrix as 1D array of length n_samples * (n_samples - 1) / 2.
    runLogs : List[Tuple[Dict[str, str], np.ndarray]]
        List of (index_dict, sequence) tuples for tracking original data.
    """
    runLogs = [({'Index': str(i)}, data[i]) for i in range(len(data))]

    try:
        dRow = pdist(data, metric=metric, **kwargs)
    except Exception as e:
        raise ValueError(f"Error computing {metric} distance: {str(e)}. "
                       f"Please check that the metric '{metric}' is supported by scipy and "
                       f"required parameters are provided.")

    return dRow, runLogs
