import numpy as np
import simclstr

def create_sample_data():
    np.random.seed(42)
    
    data_with_labels = []
    
    # Group 1: Sine waves with noise
    for i in range(5):
        t = np.linspace(0, 4*np.pi, 100)
        y = np.sin(t + i*0.2) + 0.1*np.random.randn(100)
        data_with_labels.append((f"Sine_wave_{i+1}", y))
    
    # Group 2: Exponential decay
    for i in range(4):
        t = np.linspace(0, 5, 100)
        y = np.exp(-(t + i*0.1)) + 0.05*np.random.randn(100)
        data_with_labels.append((f"Exp_decay_{i+1}", y))
    
    # Group 3: Linear trends
    for i in range(3):
        t = np.linspace(0, 10, 100)
        y = (i+1)*t + 0.1*np.random.randn(100)
        data_with_labels.append((f"Linear_{i+1}", y))
    
    return data_with_labels

def main():
    data = create_sample_data()

    distances, clusters, assignments = simclstr.perform_clustering(data, distance='pattern_dtw', cMethod='maxclust',
        cValue=6, plotDendrogram=False)

    output_file = simclstr.interactive_plot_clusters(clusters, 'pattern_dtw', auto_open=True)

if __name__ == "__main__":
    main()
