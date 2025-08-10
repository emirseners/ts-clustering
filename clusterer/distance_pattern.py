import numpy as np
from typing import Tuple, List, Dict
from .behavior_splitter import construct_features

def distance_same_length(series1: np.ndarray, series2: np.ndarray, wDim1: float, wDim2: float) -> float:
    """
    Calculate the distance between two feature vectors of the same size.

    Parameters
    ----------
    series1 : np.ndarray
        Feature vector 1 (2-dimensional numpy array).
    series2 : np.ndarray
        Feature vector 2 (2-dimensional numpy array).
    wDim1 : float
        Weight of the error between the 1st dimensions of the two 
        feature vectors (i.e. Slope).
    wDim2 : float
        Weight of the error between the 2nd dimensions of the two 
        feature vectors (i.e. Curvature).
        
    Returns
    -------
    float
        Weighted distance between the feature vectors.
    """
    diff_sq = np.square(series1 - series2)
    weighted_error = wDim1 * diff_sq[0] + wDim2 * diff_sq[1]
    return np.sum(weighted_error) / series1.shape[1]


def distance_different_length(series1: np.ndarray, series2: np.ndarray, wDim1: float, wDim2: float, sisterCount: int) -> float:
    """
    Calculate the distance between two feature vectors of different sizes.
    
    Parameters
    ----------
    series1 : np.ndarray
        Feature vector 1 (2-dimensional numpy array).
    series2 : np.ndarray
        Feature vector 2 (2-dimensional numpy array).
    wDim1 : float
        Weight of the error between the 1st dimensions of the two 
        feature vectors (i.e. Slope).
    wDim2 : float
        Weight of the error between the 2nd dimensions of the two 
        feature vectors (i.e. Curvature).
    sisterCount : int
        Number of long-versions that will be created for the 
        short vector.
        
    Returns
    -------
    float
        Minimum distance among all generated sister vectors.
    """
    length1 = series1.shape[1]
    length2 = series2.shape[1]
    
    if length1 > length2:
        shortFV = series2
        longFV = series1
    else:
        shortFV = series1
        longFV = series2

    sisters = create_sisters(shortFV, longFV.shape, sisterCount)
    error = np.square(sisters - longFV.T[np.newaxis, :, :])
    weights = np.array([wDim1, wDim2], dtype=np.float64)

    error = error * weights[np.newaxis, np.newaxis, :]
    total_error = np.sum(error, axis=(1, 2))

    return np.min(total_error) / longFV.shape[1]


def create_sisters(shortFV: np.ndarray, desired_shape: Tuple[int, int], sister_count: int) -> np.ndarray:
    """
    Create new feature vectors behaviorally identical to the given short feature vector.
    
    Creates a set of new feature vectors that are behaviorally identical to the given 
    short feature vector (shortFV), and that have the stated number of segments.
    
    Parameters
    ----------
    shortFV : np.ndarray
        The feature vector to be extended.
    desired_shape : Tuple[int, int]
        The desired shape (2-by-number of sections) of the extended feature vectors (i.e. sisters).
    sister_count : int
        The desired number of sisters to be created.
        
    Returns
    -------
    np.ndarray
        Array of sister vectors with shape (sister_count, desired_shape[1], 2).
    """
    to_add = desired_shape[1] - shortFV.shape[1]
    short_length = shortFV.shape[1]
    
    indices = np.empty((sister_count, desired_shape[1]), dtype=np.int32)
    
    random_indices = np.random.randint(0, short_length, size=(sister_count, to_add))
    
    base_indices = np.arange(short_length)
    
    indices[:, :to_add] = random_indices
    indices[:, to_add:] = base_indices[np.newaxis, :]

    indices.sort(axis=1)
    
    sisters = shortFV.T[indices, :] 

    return sisters


def distance_pattern(data: np.ndarray, significanceLevel: float = 0.01, sisterCount: int = 50, 
                    wSlopeError: float = 1, wCurvatureError: float = 1) -> Tuple[np.ndarray, List[Tuple[Dict[str, str], np.ndarray]]]:
    """
    Calculate pairwise pattern distances between all data sequences.

    The distance measures the proximity of data series in terms of their 
    qualitative pattern features. In other words, it quantifies the proximity 
    between two different dynamic behaviour modes.

    It is designed to work mainly on non-stationary data. Its current version 
    does not perform well in catching the proximity of two cyclic/repetitive 
    patterns with different number of cycles (e.g. oscillation with 4 cycles 
    versus oscillation with 6 cycles).

    Parameters
    ----------
    data : np.ndarray
        2D array of shape (n_samples, n_features) containing sequences to compare.
    significanceLevel : float, optional
        The threshold value to be used in filtering out fluctuations in the slope 
        and the curvature (default=0.01).
    sisterCount : int, optional
        Number of long-versions that will be created for the short vector while 
        comparing two data series with unequal feature vector lengths (default=50).
    wSlopeError : float, optional
        Weight of the error between the 1st dimensions of the two feature vectors 
        (i.e. Slope) (default=1).
    wCurvatureError : float, optional
        Weight of the error between the 2nd dimensions of the two feature vectors 
        (i.e. Curvature) (default=1).

    Returns
    -------
    dRow : np.ndarray
        Condensed distance matrix as 1D array of length n_samples * (n_samples - 1) / 2.
    runLogs : List[Tuple[Dict[str, str], np.ndarray]]
        List of (metadata_dict, sequence) tuples for tracking original data and feature vectors.
    """
    features = construct_features(data, significanceLevel)

    n = len(data)
    dRow = np.zeros(shape=(n * (n - 1) // 2,))

    data_w_desc = [({'Index': str(i), 'Feature vector': str(features[i])}, data[i]) 
                   for i in range(n)]

    index = 0
    for i in range(n):
        feature_i = features[i]
        for j in range(i+1, n):
            feature_j = features[j]

            if feature_i.shape[1] == feature_j.shape[1]:
                distance = distance_same_length(feature_i, feature_j, wSlopeError, wCurvatureError)
            else:
                distance = distance_different_length(feature_i, feature_j, wSlopeError, wCurvatureError, sisterCount)

            dRow[index] = distance
            index += 1

    return dRow, data_w_desc