"""
Pattern-oriented behavior clustering module.

This module provides methods for importing, pre-processing, clustering, and
post-processing of bundles of time-series data.
"""

import numpy as np
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import fcluster, linkage
import os
import pandas as pd
from typing import List, Tuple, Dict, Any, Union, Callable
from .distance_pattern import distance_pattern
from .distance_pattern_dtw import distance_pattern_dtw
from .distance_scipy import distance_scipy
from .distance_dtw import distance_dtw
from .plotting import plot_clusters, plot_dendrogram
import pysd
import itertools

distance_functions: Dict[str, Callable] = {
    'pattern': distance_pattern, 
    'pattern_dtw': distance_pattern_dtw,
    'dtw': distance_dtw,
    'scipy': distance_scipy,
    'euclidean': lambda data, **kwargs: distance_scipy(data, metric='euclidean', **kwargs),
    'manhattan': lambda data, **kwargs: distance_scipy(data, metric='cityblock', **kwargs),
    'cityblock': lambda data, **kwargs: distance_scipy(data, metric='cityblock', **kwargs),
    'mse': lambda data, **kwargs: distance_scipy(data, metric='sqeuclidean', **kwargs),
    'sse': lambda data, **kwargs: distance_scipy(data, metric='sqeuclidean', **kwargs),
    'triangle': lambda data, **kwargs: distance_scipy(data, metric='cosine', **kwargs),
    'cosine': lambda data, **kwargs: distance_scipy(data, metric='cosine', **kwargs),
    'correlation': lambda data, **kwargs: distance_scipy(data, metric='correlation', **kwargs),
    'chebyshev': lambda data, **kwargs: distance_scipy(data, metric='chebyshev', **kwargs),
    'canberra': lambda data, **kwargs: distance_scipy(data, metric='canberra', **kwargs),
    'braycurtis': lambda data, **kwargs: distance_scipy(data, metric='braycurtis', **kwargs),
    'hamming': lambda data, **kwargs: distance_scipy(data, metric='hamming', **kwargs),
    'jaccard': lambda data, **kwargs: distance_scipy(data, metric='jaccard', **kwargs),
    'minkowski': lambda data, **kwargs: distance_scipy(data, metric='minkowski', **kwargs),
    'mahalanobis': lambda data, **kwargs: distance_scipy(data, metric='mahalanobis', **kwargs),
    'seuclidean': lambda data, **kwargs: distance_scipy(data, metric='seuclidean', **kwargs)
}

def import_data(file_path: str, withClusters: bool = False) -> Union[List[Tuple[str, np.ndarray]], Tuple[List[Tuple[str, np.ndarray]], List[str]]]:
    """
    Import time series data from .xlsx or .csv files.

    The data file must have the following structure:
    
    **For Excel files (.xlsx):**
    - Sheet 'data': Column A contains labels/names for each time series, Column B onwards contains time series data values
    - Sheet 'clusters' (optional, only if withClusters=True): Column A contains cluster labels
    
    **For CSV files (.csv):**
    - Column A: Labels/names for each time series (e.g., "Run 1", "Scenario A", "Parameter Set 1")
    - Column B onwards: Time series data values
    - Row 1: First time series (label in A1, data values in B1, C1, D1, ...)
    - Row 2: Second time series (label in A2, data values in B2, C2, D2, ...)
    - And so on...
    
    Note: CSV files do not support multiple sheets, so cluster information cannot be imported
    when reading from CSV files (withClusters parameter will be ignored).
    
    Example data structure:
    ```
    | Label    | Time 1 | Time 2 | Time 3 | ... |
    |----------|--------|--------|--------|-----|
    | Run 1    | 10.5   | 12.3   | 15.7   | ... |
    | Run 2    | 11.2   | 13.1   | 16.2   | ... |
    | Run 3    | 9.8    | 11.9   | 14.5   | ... |
    ```

    Parameters
    ----------
    file_path : str
        Path to the .xlsx or .csv file (can be relative or absolute path).
    withClusters : bool, optional
        If True, also reads cluster information from 'clusters' sheet (default=False).
        Note: This parameter is ignored for CSV files as they don't support multiple sheets.

    Returns
    -------
    Union[List[Tuple[str, np.ndarray]], Tuple[List[Tuple[str, np.ndarray]], List[str]]]
        List of (label, data_array) tuples, or (data_list, clusters_list) if withClusters=True and reading from Excel.

    Raises
    ------
    FileNotFoundError
        If the specified file cannot be found.
    ValueError
        If the file doesn't have .xlsx or .csv extension.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find file: {file_path}")
    
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension not in ['.xlsx', '.csv']:
        raise ValueError("File must have .xlsx or .csv extension")

    # Read data based on file type
    if file_extension == '.xlsx':
        df_data = pd.read_excel(file_path, sheet_name='data')
    else:  # .csv
        df_data = pd.read_csv(file_path)
        # For CSV files, ignore withClusters parameter as they don't support multiple sheets
        if withClusters:
            print("Warning: CSV files do not support multiple sheets. Cluster information cannot be imported.")
            withClusters = False
    
    noRuns = len(df_data)
    
    # dataSet is a 3D list. Each entry is a 2D list object. 
    # First dimension is a string that keeps the label of the data series, 
    # and the second dimension is a numpy array that keeps the actual data
    all_rows = df_data.values.tolist()
    
    labels = [row[0] for row in all_rows]
    data_arrays = [np.array(row[1:]) for row in all_rows]
    
    data_w_desc = list(zip(labels, data_arrays))
    
    if withClusters and file_extension == '.xlsx':
        try:
            df_clusters = pd.read_excel(file_path, sheet_name='clusters')
            clusters_original = df_clusters.iloc[:, 0].tolist()
        except (FileNotFoundError, KeyError, ValueError):
            clusters_original = ['NA'] * noRuns
        return data_w_desc, clusters_original 
    else:
        return data_w_desc


def import_from_pysd(model_path: str, parameter_set: Dict[str, Union[float, List[float]]], output_variable: str) -> List[Tuple[str, np.ndarray]]:
    """
    Import and run Vensim models with parameter sets.

    Parameters
    ----------
    model_path : str
        Path to the .mdl file (can be relative or absolute path).
    parameter_set : Dict[str, Union[float, List[float]]]
        Dict of parameters and values to test.
    output_variable : str
        Output variable to extract.

    Returns
    -------
    List[Tuple[str, np.ndarray]]
        List of (description, time_series_data) tuples.

    Raises
    ------
    FileNotFoundError
        If the .mdl file cannot be found.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Could not find model file: {model_path}")
    
    if not model_path.endswith('.mdl'):
        raise ValueError("Model file must have .mdl extension")

    sd_model = pysd.read_vensim(model_path)
    param_names = list(parameter_set.keys())
    param_values = []

    for key in param_names:
        value = parameter_set[key]
        if not isinstance(value, (list, tuple)):
            param_values.append([value])
        else:
            param_values.append(value)

    param_combinations = list(itertools.product(*param_values))
    
    data_w_desc = []

    for combination in param_combinations:
        current_params = dict(zip(param_names, combination))
        simulation_results = sd_model.run(params=current_params)
        time_series_data = simulation_results[output_variable].values
        description = ', '.join([f"{name}={value}" for name, value in current_params.items()])
        data_w_desc.append((description, time_series_data))
    
    return data_w_desc


def cluster(data_w_labels: List[Tuple[str, np.ndarray]], distance: str = 'pattern_dtw', 
            interClusterDistance: str = 'complete', cMethod: str = 'inconsistent', 
            cValue: float = 1.5, plotDendrogram: bool = False, **kwargs) -> Tuple[np.ndarray, List['Cluster'], np.ndarray]:
    """
    Cluster time series data using hierarchical clustering.

    Parameters
    ----------
    data_w_labels : List[Tuple[str, np.ndarray]]
        List of (label, data_array) tuples.
    distance : str, optional
        Distance metric to use for clustering. Available options include:
        
        **Pattern-based distances:**
        - 'pattern': Pattern distance using behavioral features
        - 'pattern_dtw': Pattern distance with Dynamic Time Warping
        
        **Traditional distances (via scipy):**
        - 'euclidean': Standard Euclidean distance (default for scipy)
        - 'manhattan' or 'cityblock': Manhattan/L1 distance
        - 'mse' or 'sse': Mean/Squared Euclidean distance
        - 'cosine': Cosine distance
        - 'correlation': Correlation distance
        - 'chebyshev': Chebyshev distance
        - 'canberra': Canberra distance
        - 'braycurtis': Bray-Curtis distance
        - 'hamming': Hamming distance
        - 'jaccard': Jaccard distance
        - 'minkowski': Minkowski distance (use p parameter)
        - 'mahalanobis': Mahalanobis distance (use VI parameter)
        - 'seuclidean': Standardized Euclidean distance (use V parameter)
        
        **Custom scipy metric:**
        - 'scipy': Use any scipy.spatial.distance.pdist metric with custom parameters
        
        Default is 'pattern_dtw'.
    interClusterDistance : str, optional
        Linkage method ('complete', 'single', 'average', 'ward') (default='complete').
    cMethod : str, optional
        Cutoff method ('inconsistent', 'distance', 'maxclust', 'monocrit') (default='inconsistent').
    cValue : float, optional
        Cutoff value for clustering criterion (default=1.5).
    plotDendrogram : bool, optional
        If True, displays dendrogram (default=False).
    **kwargs : dict
        Additional distance function parameters.

    Returns
    -------
    Tuple[np.ndarray, List['Cluster'], np.ndarray]
        Tuple of (distances, cluster_list, cluster_assignments).
    """
    # Construct a list that includes only the data part. Gets rid of the label string in dataSet[i][0]
    data_wo_labels = [item[1] for item in data_w_labels]
    
    # Convert list of arrays to 2D numpy array for distance functions
    data_wo_labels_array = np.array(data_wo_labels)
    
    # Construct a list with distances. This list is the upper triangle
    # of the distance matrix
    try:
        dRow, data_w_desc = distance_functions[distance](data_wo_labels_array, **kwargs)
    except KeyError:
        print(f'Unknown distance "{distance}" is used.')
        raise ValueError(f'Unknown distance: {distance}')

    # Allocate individual runs into clusters using hierarchical agglomerative 
    # clustering. clusterSetup is a dictionary that customizes the clustering 
    # algorithm to be used.
    
    clusters, data_w_desc = flatcluster(dRow, data_w_desc, plotDendrogram=plotDendrogram, 
                                        interClusterDistance=interClusterDistance, cMethod=cMethod, cValue=cValue)

    clusterList = create_cluster_list(clusters, dRow, data_w_desc)
    return dRow, clusterList, clusters


def create_cluster_list(clusters: np.ndarray, distRow: np.ndarray, 
                       data_w_desc: List[Dict[str, Any]]) -> List['Cluster']:
    """
    Create Cluster objects from clustering results.

    Parameters
    ----------
    clusters : np.ndarray
        Array of cluster assignments.
    distRow : np.ndarray
        Condensed distance matrix.
    data_w_desc : List[Dict[str, Any]]
        List of data descriptors.

    Returns
    -------
    List['Cluster']
        List of Cluster objects.
    """
    
    nr_clusters = np.max(clusters)
    cluster_list = []
    total_size = clusters.shape[0]
    
    for i in range(1, nr_clusters+1):
        # Determine the indices for cluster i
        indices = np.where(clusters==i)[0]
        cluster_size = indices.shape[0]
        
        if cluster_size == 1:
            originalIndex = indices[0]
            cluster = Cluster(i,  indices, data_w_desc[originalIndex], [data_w_desc[originalIndex]])
            cluster_list.append(cluster)
            continue
        
        q_flat, r_flat = np.triu_indices(cluster_size, k=1)
        
        indices_q = indices[q_flat]
        indices_r = indices[r_flat]
        
        i_vals = indices_r
        j_vals = indices_q
        drow_indices = total_size * j_vals - j_vals * (j_vals + 1) // 2 + i_vals - j_vals - 1
        
        #make a distance matrix
        dist_clust = distRow[drow_indices]
        dist_matrix = squareform(dist_clust)

        #sum across the rows
        row_sum = dist_matrix.sum(axis=0)
        
        #get the index of the result with the lowest sum of distances
        min_cIndex = row_sum.argmin()
    
        # convert this cluster specific index back to the overall cluster list 
        # of indices
        originalIndex = indices[min_cIndex]

        indices_list = indices.astype(int).tolist()
        
        cluster = Cluster(i, indices, data_w_desc[originalIndex], [data_w_desc[idx] for idx in indices_list])
        cluster_list.append(cluster)
        
    return cluster_list


def flatcluster(dRow: np.ndarray, data: List[Tuple[Dict[str, Any], np.ndarray]], 
                interClusterDistance: str = 'complete', plotDendrogram: bool = True, 
                cMethod: str = 'inconsistent', cValue: float = 2.5) -> Tuple[np.ndarray, List[Tuple[Dict[str, Any], np.ndarray]]]:
    """
    Perform flat clustering using hierarchical clustering.

    Parameters
    ----------
    dRow : np.ndarray
        Condensed distance matrix.
    data : List[Tuple[Dict[str, Any], np.ndarray]]
        List of (metadata_dict, data_array) tuples.
    interClusterDistance : str, optional
        Linkage method (default='complete').
    plotDendrogram : bool, optional
        If True, displays dendrogram (default=True).
    cMethod : str, optional
        Clustering criterion (default='inconsistent').
    cValue : float, optional
        Threshold value (default=2.5).

    Returns
    -------
    Tuple[np.ndarray, List[Tuple[Dict[str, Any], np.ndarray]]]
        Tuple of (clusters, updated_data).
    """
    z = linkage(dRow, method=interClusterDistance)

    if plotDendrogram:
        plot_dendrogram(z)
    
    clusters = fcluster(z, t=cValue, criterion=cMethod)

    cluster_strings = [str(cluster_id) for cluster_id in clusters]
    for i, (metadata, _) in enumerate(data):
        metadata['Cluster'] = cluster_strings[i]
    
    return clusters, data


class Cluster:
    """
    Container for clustering results.

    Attributes
    ----------
    no : int
        Cluster number/index.
    indices : np.ndarray
        Original indices of cluster members.
    sample : Tuple[str, np.ndarray]
        Representative data series.
    members : List[Tuple[str, np.ndarray]]
        List of all cluster members.
    size : int
        Number of members in cluster.
    """

    def __init__(self, cluster_no: int, all_ds_indices: np.ndarray, sample_ds: Tuple[str, np.ndarray], member_dss: List[Tuple[str, np.ndarray]]):
        self.no = cluster_no
        self.indices = all_ds_indices
        self.sample = sample_ds
        self.size = self.indices.size
        self.members = member_dss


if __name__ == '__main__':
    inputFileName = 'TestSet_wo_Osc'
    data_set = import_data(inputFileName)

    #results = cluster(data_set, distance='manhattan')
    #results = cluster(data_set, cValue=10000, distance='manhattan', cMethod='distance')
    results = cluster(data_set, cValue=10, cMethod='maxclust', plotDendrogram=True)

    print('Distances:', results[0])
    print('Number of members in each cluster:', [results[1][i].size for i in range(len(results[1]))])
    print('Clusters:', results[2])

    plot_clusters(results[1], 'pattern', mode='show')