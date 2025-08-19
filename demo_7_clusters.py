import numpy as np
from simclstr.plotting import interactive_plot_clusters, multiple_tabs_interactive_clustering, plot_clusters
from simclstr.clusterer import read_time_series, perform_clustering
import os

def main():
    model_path_vensim = os.path.expanduser('~/Desktop/SESDYN_Basic_A/Basic_A_Vensim.xlsx')

    read_data_vensim = read_time_series(model_path_vensim)

    clustering_results_vensim = perform_clustering(read_data_vensim, cMethod='maxclust', cValue = 5, plotDendrogram=True)

    multiple_tabs_interactive_clustering(clustering_results_vensim[1], 'euclidean')

if __name__ == "__main__":
    main()