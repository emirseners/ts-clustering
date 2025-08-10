import numpy as np
from numba import njit
from typing import Tuple, List, Dict

def distance_dtw(data: np.ndarray) -> Tuple[np.ndarray, List[Tuple[Dict[str, str], np.ndarray]]]:
    """
    Calculate pairwise Dynamic Time Warping (DTW) distances between all data sequences.

    DTW can handle sequences of different lengths by finding optimal alignment.

    Parameters
    ----------
    data : np.ndarray
        2D array of shape (n_samples, n_features) containing sequences to compare.

    Returns
    -------
    dRow : np.ndarray
        Condensed distance matrix as 1D array of length n_samples * (n_samples - 1) / 2.
    runLogs : List[Tuple[Dict[str, str], np.ndarray]]
        List of (index_dict, sequence) tuples for tracking original data.
    """
    dRow = np.zeros(shape=(np.sum(np.arange(len(data))), ))

    runLogs = [({'Index': str(i)}, data[i]) for i in range(len(data))]

    dRow = compute_dtw_distances_numba(data, dRow)

    return dRow, runLogs

@njit
def compute_dtw_distances_numba(data: np.ndarray, dRow: np.ndarray) -> np.ndarray:
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
            distance = dtw_dist_numba(data[i], data[j]) 
            dRow[index] = distance
    return dRow

@njit
def dtw_dist_numba(sample1: np.ndarray, sample2: np.ndarray) -> float:
    """
    Calculate DTW distance between two sequences using dynamic programming.

    Parameters
    ----------
    sample1 : np.ndarray
        First sequence.
    sample2 : np.ndarray
        Second sequence.

    Returns
    -------
    float
        DTW distance between the sequences.
    """
    dtw = np.zeros((sample1.shape[0] + 1, sample2.shape[0] + 1))
    dtw[:, 0] = np.inf  # infinity is assigned instead of 1000
    dtw[0, :] = np.inf  # infinity is assigned instead of 1000
    dtw[0, 0] = 0
    for i in range(sample1.shape[0]):
        for j in range(sample2.shape[0]):
            cost = np.absolute(sample1[i] - sample2[j])
            dtw[i + 1, j + 1] = cost + min(dtw[i + 1, j], dtw[i, j + 1], dtw[i, j])

    return dtw[sample1.shape[0], sample2.shape[0]]