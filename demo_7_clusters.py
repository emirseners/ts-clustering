import numpy as np
from simclstr.clusterer import Cluster
from simclstr.plotting import interactive_plot_clusters

def create_seven_clusters():    
    np.random.seed(42)
    time_length = 100
    clusters = []
    
    cluster_patterns = [
        ("Sine", lambda t, i: np.sin(t + i*0.5) + np.random.normal(0, 0.1, time_length)),
        ("Cosine", lambda t, i: np.cos(t + i*0.3) + np.random.normal(0, 0.1, time_length)),
        ("Linear", lambda t, i: 0.1 * t + i*2 + np.random.normal(0, 0.2, time_length)),
        ("Exponential", lambda t, i: np.exp(-t * (0.5 + i*0.2)) + np.random.normal(0, 0.05, time_length)),
        ("Polynomial", lambda t, i: 0.001 * (t - 50)**2 + i + np.random.normal(0, 0.15, time_length)),
        ("Damped", lambda t, i: np.exp(-t*0.1) * np.sin(t + i) + np.random.normal(0, 0.08, time_length)),
        ("Step", lambda t, i: np.where(t > 50, 2+i, 0.5+i) + np.random.normal(0, 0.1, time_length))
    ]
    
    for cluster_id, (pattern_name, pattern_func) in enumerate(cluster_patterns, 1):
        members = []
        member_count = np.random.randint(2, 5)  # 2-4 members per cluster
        
        for i in range(member_count):
            if pattern_name in ["Linear", "Polynomial"]:
                t = np.linspace(0, 10, time_length)
            elif pattern_name == "Step":
                t = np.arange(time_length)
            else:
                t = np.linspace(0, 4*np.pi, time_length)
                
            ts_data = pattern_func(t, i)
            name = f"{pattern_name}_{i+1}"
            members.append((name, ts_data))
        
        cluster = Cluster(
            cluster_id=cluster_id,
            indices_of_members=np.arange(len(members)),
            list_of_members=members,
            best_representative_member=members[0]
        )
        clusters.append(cluster)
    
    return clusters

def main():
    clusters = create_seven_clusters()
    interactive_plot_clusters(clusters, "pattern dtw")

if __name__ == "__main__":
    main()