import random
import numpy as np
from sklearn.metrics import pairwise_distances


class CLARANS:
    def __init__(self, num_clusters, max_neighbors, num_local_minima):
        self.num_clusters = num_clusters
        self.max_neighbors = max_neighbors
        self.num_local_minima = num_local_minima

    def fit(self, data):
        num_points = len(data)
        min_cost = float('inf')
        best_medoids = None

        for i in range(self.num_local_minima):
            current_medoids = self._initialize_medoids(data)
            current_cost = self._calculate_total_cost(data, current_medoids)

            for j in range(self.max_neighbors):
                new_medoids = self._get_random_neighbor(current_medoids, data)
                new_cost = self._calculate_total_cost(data, new_medoids)

                if new_cost < current_cost:
                    current_medoids = new_medoids
                    current_cost = new_cost
                    j = 0  # Reset neighbor counter

            if current_cost < min_cost:
                min_cost = current_cost
                best_medoids = current_medoids

        self.medoids = best_medoids
        self.labels = self._assign_points_to_clusters(data, best_medoids)

    def _initialize_medoids(self, data):
        indices = random.sample(range(len(data)), self.num_clusters)
        return data[indices]

    def _calculate_total_cost(self, data, medoids):
        distances = pairwise_distances(data, medoids, metric='euclidean')
        return np.sum(np.min(distances, axis=1))

    def _get_random_neighbor(self, medoids, data):
        new_medoids = medoids.copy()
        random_index = random.randint(0, len(medoids) - 1)
        new_medoid_index = random.randint(0, len(data) - 1)
        new_medoids[random_index] = data[new_medoid_index]
        return new_medoids

    def _assign_points_to_clusters(self, data, medoids):
        distances = pairwise_distances(data, medoids, metric='euclidean')
        return np.argmin(distances, axis=1)

    def predict(self, data):
        return self._assign_points_to_clusters(data, self.medoids)


# Test function
def test_clarans(num_clusters = 3, max_neighbors = 10, num_local_minima = 5, num_samples = 300):
    # Generate synthetic data
    from sklearn.datasets import make_blobs
    import matplotlib.pyplot as plt

    # Generate sample data
    random_seed = random.randint(0, 10000)
    data, _ = make_blobs(n_samples=num_samples, centers=num_clusters, cluster_std=1.0, random_state=random_seed)

    # Apply CLARANS
    clarans = CLARANS(num_clusters=num_clusters, max_neighbors=max_neighbors, num_local_minima=num_local_minima)
    clarans.fit(data)
    labels = clarans.predict(data)

    # Plot results
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', marker='o', alpha=0.6)
    plt.scatter(clarans.medoids[:, 0], clarans.medoids[:, 1], c='red', marker='x')
    plt.title('CLARANS Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()
