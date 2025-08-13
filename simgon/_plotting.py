"""
Plotting utilities for clustering results.

This module provides functions for visualizing clustering results including
dendrograms and cluster plots.
"""
import matplotlib.pyplot as plt
import numpy as np
import math
from typing import List, TYPE_CHECKING
from scipy.cluster.hierarchy import dendrogram

if TYPE_CHECKING:
    from .clusterer import Cluster

def _plot_dendrogram(z: np.ndarray) -> None:
    """
    Plot hierarchical clustering dendrogram.
    
    Parameters
    ----------
        z : np.ndarray
            Linkage matrix
    """

    dendrogram(z, truncate_mode='lastp', show_leaf_counts=True, show_contracted=True)
    plt.show()


def _plot_clusters(cluster_list: List["Cluster"], dist: str, mode: str = 'show', fname: str = 'results') -> None:
    """
    Plot cluster members on separate subplots.

    Parameters
    ----------
        cluster_list : List[Cluster]
            List of Cluster objects
        dist : str
                Distance metric name for title
        mode : str
            'show' or 'save'
        fname : str
            Filename for saving (without extension)
    """
    main_fig = plt.figure(figsize=(14,10))
    main_fig.canvas.manager.set_window_title(dist + ' distance')
    no_plots = len(cluster_list)
    no_cols = 4
    no_rows = int(math.ceil(float(no_plots) / no_cols))
    i = 1
    
    for clust in cluster_list:
        sub_plot = main_fig.add_subplot(no_rows, no_cols, i)
        i = i + 1

        for j in clust.members:
            t = np.arange(j[1].shape[0])
            sub_plot.plot(t, j[1], linewidth=2)
        
        plt.title('Cluster no: ' + str(clust.no), weight='bold')

    plt.tight_layout()
    if mode=='show':
        plt.show()
    elif mode=='save':
        plt.savefig('{0}.png'.format(fname))
