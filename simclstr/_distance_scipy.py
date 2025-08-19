import numpy as np
from scipy.spatial.distance import pdist
from typing import Tuple, List, TYPE_CHECKING

if TYPE_CHECKING:
    from simclstr.clusterer import TimeSeries

def _distance_scipy(list_of_ts_objects: List['TimeSeries'], metric: str = 'euclidean', **kwargs) -> Tuple[np.ndarray, List['TimeSeries']]:
    """
    Calculate pairwise distances between all data sequences using scipy's pdist function.
    
    This function provides access to all distance metrics available in scipy.spatial.distance.pdist.

    Parameters
    ----------
    list_of_ts_objects : List['TimeSeries']
        List of TimeSeries objects.
    metric : str, optional
        Distance metric to use. Must be one of the metrics supported by scipy.spatial.distance.pdist.
        Default is 'euclidean'.
    **kwargs : dict
        Additional parameters for specific distance metrics (passed through to
        scipy.spatial.distance.pdist):
        - For 'minkowski': p
        - For 'mahalanobis': VI
        - For 'seuclidean': V

    Returns
    -------
    dRow : np.ndarray
        Condensed distance matrix as 1D array of length n_samples * (n_samples - 1) / 2.
    list_of_ts_objects : List['TimeSeries']
        List of TimeSeries objects with updated index and feature vector.
    """
    # For scipy distance metrics, the feature vector is the data itself
    for i, each_ts in enumerate(list_of_ts_objects):
        each_ts.feature_vector = each_ts.data
        each_ts.index = i

    # Convert list of arrays to 2D numpy array for distance functions
    data = np.array([ts.data for ts in list_of_ts_objects])

    try:
        dRow = pdist(data, metric=metric, **kwargs)
    except Exception as e:
        raise ValueError(f"Error computing {metric} distance: {str(e)}. "
                       f"Please check that the metric '{metric}' is supported by scipy and "
                       f"required parameters are provided.")

    return dRow, list_of_ts_objects