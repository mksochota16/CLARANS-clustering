import numpy as np
import random
import matplotlib.pyplot as plt
from typing import Dict, Union, Any

from numpy import ndarray, dtype
from sklearn.datasets import make_blobs
from scipy.io import arff
import time
from typing import List, Tuple
import requests
import zipfile
import io
import os


def preprocess_data(data: Union[np.ndarray, List[List[Union[str, float, int]]]]) -> tuple[
    ndarray[Any, dtype[Any]], dict[int, dict[int, tuple[int, Any]]]]:
    if isinstance(data, list):
        data_array = np.array(data, dtype=object)
    else:
        data_array = data

    # Convert int and float to their NumPy representations
    for i in range(data_array.shape[0]):
        for j in range(data_array.shape[1]):
            if isinstance(data_array[i, j], int):
                data_array[i, j] = np.int64(data_array[i, j])
            elif isinstance(data_array[i, j], float):
                data_array[i, j] = np.float64(data_array[i, j])

    # Initialize processed_data with the same shape
    processed_data = np.empty(data_array.shape, dtype=float)
    decoding_dict = {}

    for col in range(data_array.shape[1]):
        if all(isinstance(x, (int, float, np.number)) for x in data_array[:, col]):
            processed_data[:, col] = data_array[:, col].astype(float)
        else:
            unique_values = np.unique(data_array[:, col])
            value_to_number = {value: idx for idx, value in enumerate(unique_values)}
            number_to_value = {idx: value for idx, value in enumerate(unique_values)}
            processed_data[:, col] = np.vectorize(value_to_number.get)(data_array[:, col])
            decoding_dict[col] = number_to_value

    return processed_data, decoding_dict

def decode_data(encoded_data: np.ndarray, decoding_dict: Dict[int, Dict[int, str]]) -> np.ndarray:
    decoded_data = encoded_data.copy()
    for col, mapping in decoding_dict.items():
        reverse_mapping = np.vectorize(mapping.get)
        decoded_data[:, col] = reverse_mapping(encoded_data[:, col].astype(int))
    return decoded_data

def clarans(data_to_cluster: np.ndarray, num_clusters_input: int, num_local_input: int, max_neighbor_input: int) -> List[np.ndarray]:
    # Step 1.1: Input parameters numlocal and maxneighbor.
    def calculate_cost(medoids: List[np.ndarray], data: np.ndarray) -> float:
        cost = 0.0
        for point in data:
            cost += min([np.linalg.norm(point - medoid) for medoid in medoids]) # do przepisania
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

    def assign_points_to_medoids(medoids: List[np.ndarray], data: np.ndarray) -> List[np.ndarray]:
        clusters = [[] for _ in range(len(medoids))]
        for point in data:
            distances = [np.linalg.norm(point - medoid) for medoid in medoids]
            closest_medoid_index = np.argmin(distances)
            clusters[closest_medoid_index].append(point)
        return [np.array(cluster) for cluster in clusters]

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

    return assign_points_to_medoids(best_node, data_to_cluster)

def gen_data(num_samples: int, num_clusters: int,  num_local: int, max_neighbor: int, uniform: bool = False) ->np.ndarray:
    np.random.seed(0)
    random_seed = random.randint(0, 10000)
    if uniform:
        data: np.ndarray = np.random.rand(num_samples, 2)
    else:
        data, _ = make_blobs(n_samples=num_samples, centers=num_clusters, cluster_std=1.0, random_state=random_seed)

    return data


def plot_results(clusters: List[np.ndarray]) -> None:
    # Plot the results
    for cluster_index, cluster in enumerate(clusters):
        cluster_array: np.ndarray = np.array(cluster)
        plt.scatter(cluster_array[:, 0], cluster_array[:, 1], label=f'Cluster {cluster_index + 1}')
        # Plot the medoids (assumed to be the mean of the cluster points)
        medoid = np.mean(cluster_array, axis=0)
        plt.scatter(medoid[0], medoid[1], color='black', marker='x')

    #plt.legend()
    plt.title('CLARANS Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

def measure_time(func, *args):
    start_time = time.time()
    result = func(*args)
    end_time = time.time()
    execution_time = end_time - start_time
    return result, execution_time


# Test 1: Generated data with np.random.rand and sklearn.datasets.make_blobs
def test_generated_data_simple(num_clusters, num_local, max_neighbor, n_samples_list=None):
    #print("Testing on randomly generated data using np.random.rand")
    if n_samples_list is None:
        n_samples_list = [100, 400, 800, 1600, 3200, 6400, 12800, 25600]
    rand_results = {}
    for n_samples in n_samples_list:
        #print(f"  Testing on {n_samples} samples")
        data_rand = np.random.rand(n_samples, 2)
        preprocessed_data_result, prep_time_rand = measure_time(preprocess_data, data_rand.tolist())
        preprocessed_data_rand, decode_dict_rand = preprocessed_data_result
        result_rand, clarans_time_rand = measure_time(clarans, preprocessed_data_rand, num_clusters, num_local,
                                                      max_neighbor)
        rand_results[n_samples] = [prep_time_rand, clarans_time_rand]

    #print("Testing on blobs generated using sklearn.datasets.make_blobs")
    blob_results = {}
    for n_samples in n_samples_list:
        #print(f"  Testing on {n_samples} samples")
        data_blobs, _ = make_blobs(n_samples=n_samples, centers=num_clusters, n_features=2, random_state=42)
        preprocessed_data_result, prep_time_blobs = measure_time(preprocess_data, data_blobs.tolist())
        preprocessed_data_blobs, decode_dict_blobs = preprocessed_data_result
        result_blobs, clarans_time_blobs = measure_time(clarans, preprocessed_data_blobs, num_clusters, num_local,
                                                        max_neighbor)
        blob_results[n_samples] = [prep_time_blobs, clarans_time_blobs]

    return rand_results, blob_results


def test_generated_data_dimensions(num_clusters, num_local, max_neighbor):
    # Compare execution times for varying dimensions
    results = {}
    for dims in [2, 3, 4, 8, 16, 32]:
        data_high_dim, _ = make_blobs(n_samples=1000, centers=3, n_features=dims, random_state=42)
        preprocessed_data_high_dim, _ = preprocess_data(data_high_dim.tolist())

        _, clarans_time_high_dim = measure_time(clarans, preprocessed_data_high_dim, num_clusters, num_local,
                                                max_neighbor)
        results[dims] = clarans_time_high_dim
        print(f"CLARANS Time for {dims} dimensions: {clarans_time_high_dim:.4f} seconds")

    return results


# Test 2: Real-world data from clustering-benchmark
def test_real_world_data(num_local, max_neighbor, benchmark_name="3-spiral.arff", num_clusters=None):
    # url = "https://github.com/deric/clustering-benchmark/archive/refs/heads/master.zip"
    # response = requests.get(url)
    # with zipfile.ZipFile(io.BytesIO(response.content)) as z:
    #     z.extractall("clustering-benchmark")

    base_dir = "clustering-benchmark/clustering-benchmark-master/src/main/resources/datasets/artificial"
    datasets = [benchmark_name]

    for dataset in datasets:
        print(f"\nTesting on real-world dataset: {dataset}")
        file_path = os.path.join(base_dir, dataset)

        # Assuming ARFF file handling is available via scipy.io or similar package

        data, meta = arff.loadarff(file_path)
        if num_clusters is None:
            try:
                num_clusters = len(meta['class'][1])
            except:
                num_clusters = len(meta['CLASS'][1])
        data = np.array(data.tolist(), dtype=object)
        preprocessed_data, _ = preprocess_data(data.tolist())

        result, prep_time = measure_time(preprocess_data, data.tolist())
        result, clarans_time = measure_time(clarans, preprocessed_data, num_clusters, num_local, max_neighbor)

        print(f"Preprocessing Time ({dataset}): {prep_time:.4f} seconds")
        print(f"CLARANS Time ({dataset}): {clarans_time:.4f} seconds")
        return result, prep_time, clarans_time


if __name__ == '__main__':
    num_clusters: int = 3
    num_local: int = 5
    max_neighbor: int = 10
    num_samples: int = 300

    data = gen_data(num_samples, num_clusters, num_local, max_neighbor, uniform=False)

    clusters: List[np.ndarray] = clarans(data, num_clusters, num_local, max_neighbor)

    plot_results(clusters)

