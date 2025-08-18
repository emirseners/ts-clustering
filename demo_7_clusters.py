import numpy as np
from simclstr.clusterer import Cluster
from simclstr.plotting import interactive_plot_clusters
from simclstr.clusterer import read_time_series, perform_clustering
import os

def main():
    model_path_stella2 = os.path.expanduser('~/Desktop/SESDYN_Basic_A/Basic_A_Stella2.xlsx')

    read_data_stella2 = read_time_series(model_path_stella2)

    clustering_results_stella2 = perform_clustering(read_data_stella2, cMethod='maxclust', cValue = 7, plotDendrogram=True)

    interactive_plot_clusters(clustering_results_stella2[1], 'pattern_dtw')


if __name__ == "__main__":
    main()