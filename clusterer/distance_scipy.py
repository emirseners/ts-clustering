import numpy as np
from scipy.spatial.distance import pdist
from typing import Tuple, List, Dict, Any

def distance_scipy(data: np.ndarray, metric: str = 'euclidean', **kwargs) -> Tuple[np.ndarray, List[Tuple[Dict[str, str], np.ndarray]]]:
    """
    Calculate pairwise distances between all data sequences using scipy's pdist function.
    
    This function provides access to all distance metrics available in scipy.spatial.distance.pdist,
    including but not limited to:
    - 'euclidean', 'sqeuclidean' (squared Euclidean)
    - 'cityblock' (Manhattan), 'chebyshev'
    - 'cosine', 'correlation'
    - 'hamming', 'jaccard'
    - 'canberra', 'braycurtis'
    - 'minkowski' (with p parameter)
    - 'mahalanobis' (with VI parameter)
    - 'seuclidean' (standardized Euclidean)
    - And many more...

    Parameters
    ----------
    data : np.ndarray
        2D array of shape (n_samples, n_features) with equal-length sequences.
    metric : str, optional
        Distance metric to use. Must be one of the metrics supported by scipy.spatial.distance.pdist.
        Default is 'euclidean'.
    **kwargs : dict
        Additional parameters for specific distance metrics:
        - For 'minkowski': p (norm parameter, default=2)
        - For 'mahalanobis': VI (inverse covariance matrix)
        - For 'seuclidean': V (variance vector)
        - For 'wminkowski': w (weight vector)
        - For 'sokalsneath': w (weight vector)

    Returns
    -------
    dRow : np.ndarray
        Condensed distance matrix as 1D array of length n_samples * (n_samples - 1) / 2.
    runLogs : List[Tuple[Dict[str, str], np.ndarray]]
        List of (metadata_dict, sequence) tuples for tracking original data.

    Examples
    --------
    # Basic usage with default Euclidean distance
    distances, logs = distance_scipy(data)
    
    # Using Manhattan distance
    distances, logs = distance_scipy(data, metric='cityblock')
    
    # Using Minkowski distance with p=3
    distances, logs = distance_scipy(data, metric='minkowski', p=3)
    
    # Using cosine distance
    distances, logs = distance_scipy(data, metric='cosine')
    
    # Using squared Euclidean (equivalent to old 'mse' function)
    distances, logs = distance_scipy(data, metric='sqeuclidean')
    
    # Using triangle distance (equivalent to old 'triangle' function)
    distances, logs = distance_scipy(data, metric='cosine')
    distances = 1 - distances  # Convert to similarity
    """
    runLogs = [({'Index': str(i)}, data[i]) for i in range(len(data))]

    try:
        dRow = pdist(data, metric=metric, **kwargs)
    except Exception as e:
        raise ValueError(f"Error computing {metric} distance: {str(e)}. "
                       f"Please check that the metric '{metric}' is supported by scipy and "
                       f"required parameters are provided.")

    return dRow, runLogs
