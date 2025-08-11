'''
Pattern-oriented behavior clustering module.

This module provides methods for importing, pre-processing, clustering, and
post-processing of bundles of time-series data (e.g., from a sensitivity
analysis).
'''

import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import fcluster, linkage, dendrogram
import os
import pandas as pd
from .distance_pattern import distance_pattern
from .distance_pattern_dtw import distance_pattern_dtw
from .distance_mse import distance_mse
from .distance_sse import distance_sse
from .distance_dtw import distance_dtw
from .distance_manhattan import distance_manhattan
import pysd
import itertools

distance_functions = {'pattern': distance_pattern, 'pattern_dtw': distance_pattern_dtw,'mse': distance_mse, 'sse': distance_sse, 'dtw': distance_dtw, 'manhattan': distance_manhattan}

def import_data(inputFileName, withClusters = False):
    '''
    Method that imports dataseries to be analyzed from .xlsx files. Unless specified otherwise, looks for the file in the datasets folder of the project. Optionally it can also read the original clusters of the dataseries. For that, the input file should contain a sheet names *clusters*, and the order of the dataseries in this sheet should be identical to the sorting in the data sheet

    :param inputFileName: The name of the .xlsx file that contains the dataset
    :param withClusters: If True, checks the sheet names *clusters* and returns also the original clusters/classess of dataseries
    :returns: Two lists. The first one contains 2D lists, each corresponding to a single dataseries. The first entry is a string that keeps the label of the sample, and the second entry is a numpy array that keeps the data. The second list is optional, and returns when *withClusters* is True. It contains the original clusters of the input data  
    :rtype: List (3D)
    '''
    possible_paths = ['datasets', '../datasets']
    file_path = None
    
    for relPathFileFolder in possible_paths:
        test_path = relPathFileFolder+'/'+inputFileName+'.xlsx'
        if os.path.exists(test_path):
            file_path = test_path
            break
    
    if file_path is None:
        raise FileNotFoundError(f"Could not find {inputFileName}.xlsx")
    
    # Read the data sheet using pandas
    df_data = pd.read_excel(file_path, sheet_name='data')
    noRuns = len(df_data)
    
    #dataSet is a 3D list. Each entry is a 2D list object. First dimension is a string that keeps the label of the data series, and the second dimension is a numpy array that keeps the actual data
    all_rows = df_data.values.tolist()
    
    # Vectorized separation of labels and data
    labels = [row[0] for row in all_rows]
    data_arrays = [np.array(row[1:]) for row in all_rows]
    
    data_w_desc = list(zip(labels, data_arrays))
    
    if withClusters:
        try:
            df_clusters = pd.read_excel(file_path, sheet_name='clusters')
            clusters_original = df_clusters.iloc[:, 0].tolist()
        except:
            clusters_original = ['NA'] * noRuns
        return data_w_desc, clusters_original 
    else:
        return data_w_desc

def import_from_pysd(model_name, parameter_set, output_variable, models_folder='vensim models'):
    """
    Import a model from pysd and run it with a given parameter set.
    :param model_name: The name of the model to import
    :param parameter_set: A dictionary of parameters, where keys are the parameter names and values are the parameter values or a list of parameter values
    :param output_variable: The name of the output variable to extract from simulation results
    :param models_folder: The folder containing the Vensim models (default: 'models')
    :return: A list of tuples in the same format as import_data: [(description, time_series_data), ...]
    """
    possible_paths = [models_folder, '../' + models_folder, 'vensim_models', '../vensim_models']
    model_path = None
    
    for rel_path in possible_paths:
        test_path = os.path.join(rel_path, model_name + '.mdl')
        if os.path.exists(test_path):
            model_path = test_path
            break

    if model_path is None:
        model_path = model_name + '.mdl'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Could not find {model_name}.mdl.")

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
    
    results = []

    for combination in param_combinations:
        current_params = dict(zip(param_names, combination))
        simulation_results = sd_model.run(params=current_params)
        time_series_data = simulation_results[output_variable].values
        description = ', '.join([f"{name}={value}" for name, value in current_params.items()])
        results.append((description, time_series_data))
    
    return results


def cluster(data_w_labels,  distance='pattern_dtw', interClusterDistance='complete',
            cMethod='inconsistent', cValue=1.5, plotDendrogram=False, **kwargs):
    '''
    Method that clusters time-series data based on the specified distance measure using a hierarchical clustering algorithm. Optionally the method also plots the dendrogram generated by the clustering algorithm
    
    :param data: A list of lists. Each entry of the master list corresponds to a dataseries. The second order lists have two entries: The first entry is the label of the dataseries, and the second entry is a numpy array that keeps the data
    :param str distance: The distance metric to be used. Default value is *'pattern'*
    :param str interClusterDistance: How to calculate inter cluster distance.
                                 see `linkage <http://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html#scipy.cluster.hierarchy.linkage>`_ 
                                 for details. Default value is *'inconsistent'*
    :param cMethod: Cutoff method, 
                    see `fcluster <http://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.fcluster.html#scipy.cluster.hierarchy.fcluster>`_ 
                    for details.
    :param cValue: Cutoff value, see 
                   `fcluster <http://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.fcluster.html#scipy.cluster.hierarchy.fcluster>`_ 
                   for details.
    :param plotDendogram: Boolean, if true, plot dendogram.
    :returns: A tuple containing the list of distances (i.e. dRow), the list of Cluster objects (i.e. clusterList : for each cluster a Cluster object that contains basic info about the cluster), 
            and a list that gives the index of the cluster each data series is allocated (i.e. clusters).     
    :rtype: Tuple
    
    The remainder of the arguments are passed on to the specified distance 
    function.
    
    Pattern Distance:
    
    * 'distance': String that specifies the distance to be used. 
                  Options: pattern (default), mse, sse, triangle
    * 'filter?': Boolean that specifies whether the data series will be 
                 filtered (for bmd distance)
    * 'slope filter': A float number that specifies the filtering threshold 
                     for the slope (for every data point if change__in_the_
                     outcome/average_value_of_the_outcome < threshold, 
                     consider slope = 0) (for bmd distance)
    * 'curvature filter': A float number that specifies the filtering 
                          threshold for the curvature (for every data point if 
                          change__in_the_slope/average_value_of_the_slope < 
                          threshold, consider curvature = 0) (for bmd distance)
    * 'no of sisters': 50 (for pattern distance)
    '''
    # Construct a list that includes only the data part. Gets rid of the label string in dataSet[i][0]
    data_wo_labels = [item[1] for item in data_w_labels]
    
    # Construct a list with distances. This list is the upper triangle
    # of the distance matrix
    dRow, data_w_desc = construct_distances(data_wo_labels, distance, **kwargs)

    # Allocate individual runs into clusters using hierarchical agglomerative 
    # clustering. clusterSetup is a dictionary that customizes the clustering 
    # algorithm to be used.
    
    clusters, data_w_desc = flatcluster(dRow, data_w_desc, plotDendrogram=plotDendrogram, 
                                        interClusterDistance=interClusterDistance, cMethod=cMethod, cValue=cValue)

    clusterList = create_cluster_list(clusters, dRow, data_w_desc)
    return dRow, clusterList, clusters


def create_cluster_list(clusters, distRow, data_w_desc):
    '''  
    Given the results of a clustering, the method creates Cluster objects for each of the identified clusters. Each cluster object contains member data series, as well as a sample/representative dataseries
    
    :param clusters: A list that contains the cluster number of the corresponding dataseries in the dataset(If the clusters[5] is 12, data[5] belongs to cluster 12 
    :param distRow: The row of distances coming from the distance function
    :param data_w_desc: The list that contains the raw data as well as the descriptor dictionary for each data series 
    
    :returns: A list of Cluster objects
    :rtype: List
    '''
    
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


def construct_distances(data_wo_labels, distance='pattern', **kwargs):
    """ 
    Constructs a row vector of distances (a condensed version of a n-by-n matrix of distances) for n data-series in data 
    according to the specified distance.
    
    Distance argument specifies the distance measure to be used. Options are as follows;
        * pattern: a distance based on qualitative dynamic pattern features 
        * sse: regular sum of squared errors
        * mse: regular mean squared error
        * triangle: triangular distance 
        * dtw: Dynamic time warping distance
    
    :param data: The list of dataseries to be clustered. Each entry is a numpy array that stores the data for a timeseries. 
    :param distance: The distance type to be used in calculating the pairwise distances. Default is *'pattern'* 
    :returns: A row vector of distances, and a list that stores the original data with distance-relevant dataseries descriptor 
    :rtype: Tuple (2 lists)
    """
    # Sets up the distance function according to user specification
    try:
        return distance_functions[distance](data_wo_labels, **kwargs)
    except KeyError:
        print(f'Unknown distance "{distance}" is used.')
        raise ValueError(f'Unknown distance: {distance}')


def flatcluster(dRow, data, interClusterDistance='complete', plotDendrogram=True, cMethod='inconsistent', cValue=2.5):
    z = linkage(dRow, method=interClusterDistance)
    
    if plotDendrogram:
        plotdendrogram(z)
    
    clusters = fcluster(z, t=cValue, criterion=cMethod)
        
    # Debug information to show clustering results
    # unique_clusters, cluster_counts = np.unique(clusters, return_counts=True)
    # print('Total number of clusters:', noClusters)
    # print('Clustering method:', cMethod, 'with value:', cValue)
    # for cluster_id, count in zip(unique_clusters, cluster_counts):
    #     print(f"Cluster {cluster_id}: {count} members")
    
    cluster_strings = [str(cluster_id) for cluster_id in clusters]
    for i, log in enumerate(data):
        log[0]['Cluster'] = cluster_strings[i]
    
    return clusters, data


def plotdendrogram(z):
    dendrogram(z, truncate_mode='lastp', show_leaf_counts=True, show_contracted=True)
    plt.show()


def plot_clusters(cluster_list, dist, mode='show',fname='results'):
    '''
    Takes a list of Cluster objects as an input. Plots the members of each cluster on a seperate plot
    
    :param clusterList: deneme
    :param dataset:
    :param groupPlot:
    :param mode: default is show, use save to save the figure in png format
    :param fname: if mode=save option is used this is used as the filename, type without extension
    :rtype: Matplotlib graph
    '''
    main_fig = plt.figure(figsize=(14,10))
    #main_fig.suptitle('deneme')
    main_fig.canvas.manager.set_window_title(dist + ' distance')
    no_plots = len(cluster_list)
    no_cols = 4
    no_rows = int(math.ceil(float(no_plots) / no_cols))
    i = 1
    
    for clust in cluster_list:
        sub_plot = main_fig.add_subplot(no_rows, no_cols, i)
        i = i + 1
               
        #=======================================================================
        # # For plotting only the sample of each cluster 
        # t = np.array(range(clust.sample[1].shape[0]))
        # sub_plot.plot(t, clust.sample[1], linewidth=2)
        #=======================================================================
         
        for j in clust.members:
            t = np.arange(j[1].shape[0])
            sub_plot.plot(t, j[1], linewidth=2)
        
        plt.title('Cluster no: ' + str(clust.no), weight='bold')
        #plt.ylim(0, 100)
    plt.tight_layout()
    if mode=='show':
        plt.show()
    elif mode=='save':
        plt.savefig('{0}.png'.format(fname))


class Cluster(object):
    '''
    Cluster container for time-series results.

    Basic attributes of a cluster object are as follows:

    - no: Cluster number/index
    - indices: Original indices of the dataseries that are in the cluster
    - sample: Original index of the dataseries that is the representative of the cluster (i.e., median element)
    - members: Members of the cluster
    - size: Number of elements (i.e., dataseries) in the cluster
    '''

    def __init__(self, cluster_no, all_ds_indices, sample_ds, member_dss):
        self.no = cluster_no
        self.indices = all_ds_indices
        self.sample = sample_ds
        self.size = self.indices.size
        self.members = member_dss


if __name__ == '__main__':
    inputFileName = 'TestSet_wo_Osc'

    #experiment_controller(inputFileName, distanceMethod='pattern_dtw', flatMethod='complete',
    #       transform='normalize', cMethod='maxclust', note="wSLope=6", cValue=9, replicate=1)

    data_set = import_data(inputFileName)

    #results = cluster(data_set, distance='manhattan')
    #results = cluster(data_set, cValue=10000, distance='manhattan', cMethod='distance')
    results = cluster(data_set, cValue=10, cMethod='maxclust', plotDendrogram=True)

    print('Distances:', results[0])
    print('Number of members in each cluster:', [results[1][i].size for i in range(len(results[1]))])
    print('Clusters:', results[2])

    plot_clusters(results[1], 'pattern', mode='show')

    #vensim_result_data = import_from_pysd(model_name = 'constantflow', parameter_set = {'constant flow value':[5, 10, 20]}, output_variable = 'OI')
    #vensim_result = cluster(vensim_result_data, cValue=2, cMethod='maxclust')
    #print('Vensim result:', vensim_result) Returns 1, 1, 1