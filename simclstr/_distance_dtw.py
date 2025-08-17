import numpy as np
from numba import njit
from typing import Tuple, List, Dict

def _distance_dtw(data: np.ndarray) -> Tuple[np.ndarray, List[Tuple[Dict[str, str], np.ndarray]]]:
    """
    Calculate pairwise Dynamic Time Warping (DTW) distances between all data sequences.

    Dynamic Time Warping is a technique for measuring similarity between two temporal
    sequences that may vary in speed or timing. Unlike Euclidean distance, DTW can
    handle sequences of different lengths and finds the optimal alignment between them
    by allowing stretching and compression of the time axis.
    
    Parameters
    ----------
    data : np.ndarray
        2D array of shape (n_samples, n_features) containing time series sequences to compare.
        Sequences can have different lengths, making DTW particularly suitable for
        comparing time series with temporal variations.

    Returns
    -------
    dRow : np.ndarray
        Condensed distance matrix as 1D array of length n_samples * (n_samples - 1) / 2.
        Each element represents the DTW distance between a pair of sequences.
    runLogs : List[Tuple[Dict[str, str], np.ndarray]]
        List of (metadata_dict, sequence) tuples for tracking original data:
        
        - metadata_dict: Dictionary containing sequence index information
        - sequence: Original time series data

    This implementation uses absolute difference as the local distance measure
    between individual points: |x_i - y_j|.
    """
    dRow = np.zeros(shape=(np.sum(np.arange(len(data))), ))

    runLogs = [({'Index': str(i)}, data[i]) for i in range(len(data))]

    dRow = compute_dtw_distances(data, dRow)

    return dRow, runLogs

@njit
def compute_dtw_distances(data: np.ndarray, dRow: np.ndarray) -> np.ndarray:
    """
    Compute DTW distances between all pairs using Numba for performance.

    Parameters
    ----------
    data : np.ndarray
        Input sequences array.
    dRow : np.ndarray
        Array to store distances.

    Returns
    -------
    np.ndarray
        Array filled with DTW distances.
    """
    index = -1
    for i in range(len(data)):            
        for j in range(i+1, len(data)):
            index += 1
            
            sample1 = data[i]
            sample2 = data[j]
            
            dtw = np.zeros((sample1.shape[0] + 1, sample2.shape[0] + 1))
            dtw[:, 0] = np.inf
            dtw[0, :] = np.inf
            dtw[0, 0] = 0
            
            for k in range(sample1.shape[0]):
                for l in range(sample2.shape[0]):
                    cost = np.absolute(sample1[k] - sample2[l])
                    dtw[k + 1, l + 1] = cost + min(dtw[k + 1, l], dtw[k, l + 1], dtw[k, l])
            
            distance = dtw[sample1.shape[0], sample2.shape[0]]
            dRow[index] = distance
            
    return dRow