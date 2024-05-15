import numpy as np
import random
import matplotlib.pyplot as plt
from typing import List, Dict

from sklearn.datasets import make_blobs


def clarans(data_to_cluster: np.ndarray, num_clusters_input: int, num_local_input: int, max_neighbor_input: int) -> List[np.ndarray]:
    # Step 1.1: Input parameters numlocal and maxneighbor.
    def calculate_cost(medoids: List[np.ndarray], data: np.ndarray) -> float:
        cost = 0.0
        for point in data:
            cost += min([np.linalg.norm(point - medoid) for medoid in medoids])
        return cost

    def get_neighbors(current: List[np.ndarray], data: np.ndarray, num_clusters: int) -> List[List[np.ndarray]]:
        neighbors = []
        for i in range(num_clusters):
            for j in range(len(data)):
                if not np.array_equal(current[i], data[j]):
                    neighbor = current.copy()
                    neighbor[i] = data[j]
                    neighbors.append(neighbor)
        return neighbors

    best_node = None

    # Step 1.2: Initialize i to 1, and min_cost to a large number.
    min_cost = np.inf #calculate_cost(best_nodes, data_to_cluster)
    for i in range(num_local_input):
        # Step 2: Set current to an arbitrary node in Gn;k.
        current = random.sample(list(data_to_cluster), num_clusters_input)
        current_cost = calculate_cost(current, data_to_cluster)
        # Step 3: Set j to 0.
        j = 0

        # Step 4: Inner loop for maxneighbor iterations
        while j < max_neighbor_input:
            # Step 4: Consider a random neighbor S (neighbors) of current, and based
            #         on Step 5, calculate the cost differential of the two nodes.
            neighbors = get_neighbors(current, data_to_cluster, num_clusters_input)
            random_neighbor = random.choice(neighbors)
            random_neighbor_cost = calculate_cost(random_neighbor, data_to_cluster)

            # Step 5: If S has a lower cost, set current to S, and go to Step 3 (restart loop)
            if random_neighbor_cost < current_cost:
                current = random_neighbor
                current_cost = random_neighbor_cost
                j = 0
                continue # unnecessary but clearer
            # Step 6: Otherwise, increment j by 1. If j maxneighbor, go to Step 4 (continue loop).
            else:
                j += 1
                continue # unnecessary but clearer

        # Step 7: Otherwise, when j > maxneighbor (checked in while loop), compare the cost
        #         of current with mincost. If the former is less than
        #         mincost, set mincost to the cost of current and set
        #         bestnode to current.
        if current_cost < min_cost:
            best_node = current
            min_cost = current_cost

        # Step 8: Increment i by 1. If i > numlocal, output bestnode and halt.
        #         Otherwise, go to Step 2. Handled by for loop.

    return best_node

def gen_data(num_samples: int, num_clusters: int,  num_local: int, max_neighbor: int, uniform: bool = False) ->np.ndarray:
    np.random.seed(0)
    random_seed = random.randint(0, 10000)
    if uniform:
        data: np.ndarray = np.random.rand(num_samples, 2)
    else:
        data, _ = make_blobs(n_samples=num_samples, centers=num_clusters, cluster_std=1.0, random_state=random_seed)

    return data

def plot_results(data: np.ndarray, medoids: List[np.ndarray], num_clusters: int) -> None:
    clusters: Dict[int, List[np.ndarray]] = {i: [] for i in range(num_clusters)}
    for point in data:
        distances: List[float] = [np.linalg.norm(point - medoid) for medoid in medoids]
        cluster_index: int = distances.index(min(distances))
        clusters[cluster_index].append(point)

    # Plot the results
    for cluster_index, cluster in clusters.items():
        cluster_array: np.ndarray = np.array(cluster)
        plt.scatter(cluster_array[:, 0], cluster_array[:, 1], label=f'Cluster {cluster_index + 1}')
    for medoid in medoids:
        plt.scatter(medoid[0], medoid[1], color='black', marker='x')
    plt.legend()
    plt.title('CLARANS Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()



if __name__ == '__main__':
    num_clusters: int = 3
    num_local: int = 5
    max_neighbor: int = 10
    num_samples: int = 300

    data = gen_data(num_samples, num_clusters, num_local, max_neighbor, uniform=False)

    medoids: List[np.ndarray] = clarans(data, num_clusters, num_local, max_neighbor)

    plot_results(data, medoids, num_clusters)

