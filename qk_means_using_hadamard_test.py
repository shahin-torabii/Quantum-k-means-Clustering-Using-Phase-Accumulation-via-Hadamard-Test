import numpy as np
from qiskit import QuantumCircuit, transpile, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.visualization import circuit_drawer

from qiskit.circuit.library import MCMT, RYGate
from sklearn.datasets import make_blobs, make_moons, load_iris, load_wine, load_diabetes, load_breast_cancer
from sklearn.preprocessing import StandardScaler, Normalizer
import matplotlib.pyplot as plt
from sklearn.metrics import v_measure_score, silhouette_score, adjusted_rand_score, normalized_mutual_info_score, \
    davies_bouldin_score, homogeneity_score, adjusted_mutual_info_score, completeness_score, accuracy_score,\
    confusion_matrix,precision_score, recall_score, f1_score,fowlkes_mallows_score
from sklearn.cluster import KMeans
import time
from scipy.stats import mode
from sklearn.decomposition import PCA
import math
from qiskit_aer.noise import (
    NoiseModel, 
    depolarizing_error, 
    thermal_relaxation_error, 
    ReadoutError
)
from math import sqrt
from qiskit_ibm_runtime.fake_provider import FakeGuadalupeV2
from scipy.optimize import linear_sum_assignment
import logging
import random
from collections import defaultdict
import seaborn as sns




logging.getLogger('qiskit_aer').setLevel(logging.ERROR)
logging.getLogger('qiskit').setLevel(logging.ERROR)

np.random.seed(42)
random.seed(42)


def init_centroids(data, k):
    """k-means++ centroid initialization"""

    centroids = []
    first_centroid = data[np.random.choice(data.shape[0], replace=False)]
    centroids.append(first_centroid)
    for _ in range(k - 1):
        dists = np.min([np.linalg.norm(data - c, axis=1) for c in centroids], axis=0)
        probabilities = dists ** 2 / np.sum(dists ** 2)
        new_centroid = data[np.random.choice(data.shape[0], p=probabilities)]
        centroids.append(new_centroid)
    return centroids



def encode_with_angles(qc, qubits, data):
    """Encodes classical data using angle encoding via RY rotations."""
    theta_list = []
    for i, value in enumerate(data):
        theta = 2 * np.arcsin(np.clip(value, -1, 1))
        theta_list.append(theta)
        qc.ry(theta, qubits[i])
    return theta_list

def hadamard_test(qc, record_qubit, centroid_qubit, ancilla_qubit, theta):
    """Performs the Hadamard test using Angle Encoding."""
    qc.h(ancilla_qubit)

    qc.cry(theta, ancilla_qubit, record_qubit)

    qc.h(ancilla_qubit)
 

def get_noisy_backend_for_kmeans():
    """Add noise to the simulator to simulate real condition"""

    noise_model = NoiseModel()

    # ---- Depolarizing gate errors ----
    # before: 0.001 / 0.01
    p1q = 0.0005      # mild single-qubit noise
    p2q = 0.005       # mild CX noise

    error_1q = depolarizing_error(p1q, 1)
    error_2q = depolarizing_error(p2q, 2)

    single_qubit = ['rz', 'sx', 'x', 'h', 'ry']
    two_qubit = ['cx']

    noise_model.add_all_qubit_quantum_error(error_1q, single_qubit)
    noise_model.add_all_qubit_quantum_error(error_2q, two_qubit)

    # ---- Thermal relaxation ----
    # increase coherence relative to gate duration
    t1, t2 = 200e3, 160e3   # longer coherence times
    gate_time_1q = 45        # faster gates
    gate_time_2q = 220

    relax_1q = thermal_relaxation_error(t1, t2, gate_time_1q)
    relax_single = thermal_relaxation_error(t1, t2, gate_time_2q)
    relax_2q = relax_single.tensor(relax_single)

    noise_model.add_all_qubit_quantum_error(relax_1q, single_qubit)
    noise_model.add_all_qubit_quantum_error(relax_2q, two_qubit)

    # ---- Readout error ----
    # before: 2%
    p_flip = 0.008

    readout = ReadoutError([
        [1 - p_flip, p_flip],
        [p_flip, 1 - p_flip]
    ])

    noise_model.add_all_qubit_readout_error(readout)

    return AerSimulator(
        method="density_matrix",
        noise_model=noise_model,
        seed_simulator=42
    )

def distance_circuit_qf(record, centroid, shots=8192):
    """Computes the quantum distance between a record and a centroid using multiple ancillas."""
    n_data_qubits = len(record)  # Number of qubits required for data

    # Define Quantum Registers
    record_qreg = QuantumRegister(n_data_qubits, name="record")
    centroid_qreg = QuantumRegister(n_data_qubits, name="centroid")
    ancilla_qreg = QuantumRegister(n_data_qubits, name="ancilla")  # One ancilla per qubit pair
    creg = ClassicalRegister(n_data_qubits, name="c")  # Measure all ancillas

    qc = QuantumCircuit(record_qreg, ancilla_qreg, creg)

    record_theta = encode_with_angles(qc, record_qreg, record)
    centroid_theta = []
    for i, value in enumerate(centroid):
        theta = 2 * np.arcsin(np.clip(value, -1, 1))
        centroid_theta.append(theta)


    for i in range(n_data_qubits):
        theta = (record_theta[i] - centroid_theta[i])/2
        hadamard_test(qc, record_qreg[i], centroid_qreg[i], ancilla_qreg[i], theta)

    qc.measure(ancilla_qreg, creg)
    
    # simulator = AerSimulator(
    #     method="density_matrix",
    #     seed_simulator=42
    # )
    simulator = get_noisy_backend_for_kmeans()
    transpiled_qc = transpile(qc, simulator, optimization_level=3, seed_transpiler=42)
    result = simulator.run(transpiled_qc, shots=shots).result()
    #print(transpiled_qc.depth())
    counts = result.get_counts()




    #### Test with fake backend####

    # simulator = get_noisy_backend_for_kmeans()
    # backend = FakeGuadalupeV2()
    # # print(qc.num_qubits)
    # # print(qc.count_ops())
    # transpiled_qc = transpile(qc,backend, optimization_level=3, seed_transpiler=42)
    # result = simulator.run(transpiled_qc, shots=shots).result()
    # print(transpiled_qc.count_ops())
    # print(f"number of qubits is {transpiled_qc.num_qubits}")
    # print(f"depth is {transpiled_qc.depth()}")
    # counts = result.get_counts()
    # exit()



    total_prob_0 = 0
    for outcome, count in counts.items():

        num_zeros = outcome.count('0')  # Count number of 0s in the outcome
        total_prob_0 += (num_zeros / n_data_qubits) * (count / shots)

    average_overlap = 2 * total_prob_0 - 1
    distance = np.sqrt(2 - 2 * average_overlap)
    #print(f"Computed quantum distance: {distance}")
    return distance

def has_converged(old_centroids, new_centroids, threshold=0.0001):
    """Check if centroids have converged"""
    return np.linalg.norm(np.array(old_centroids) - np.array(new_centroids), ord='fro') <= threshold


def qk_means(data, k, max_iter=10, sc_thresh=0.0001, ground_truth=None, kmena_labels=None, v_measure=True):
    """Quantum k-means clustering with Quantum Forking for distance calculation."""
    centroids = init_centroids(data, k)
    labels = np.zeros(len(data), dtype=int)
    iteration_sims = 0
    iter_count = 0

    while max_iter > 0:
        clusters = [[] for _ in range(k)]
        for i, record in enumerate(data):
            min_distance = np.inf
            for idx, centroid in enumerate(centroids):
                distance = distance_circuit_qf(record, centroid)
                if distance < min_distance:
                    min_distance = distance
                    labels[i] = idx
            clusters[labels[i]].append(record)

        new_centroids = [np.mean(cluster, axis=0) if cluster else centroids[idx]
                         for idx, cluster in enumerate(clusters)]

        if ground_truth is not None:
            iteration_sims += adjusted_rand_score(labels, ground_truth)
        elif kmena_labels is not None:
            iteration_sims += adjusted_rand_score(labels, kmena_labels)

        if has_converged(centroids, new_centroids, sc_thresh):
            break

        centroids = new_centroids
        max_iter -= 1
        iter_count += 1

    return labels, centroids, clusters,iter_count


def k_means(data, k, max_iters=300, ground_truth=None):
    """Perform classic k-means on data"""
    kmeans = KMeans(n_clusters=k, max_iter=max_iters, n_init=1, random_state=42)

    kmeans.fit(data)

    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    iters = kmeans.n_iter_

    return labels, centroids

def hungarian_cluster_mapping(y_true, y_pred):
    """
    Uses Hungarian algorithm to find optimal 1-to-1 mapping
    between cluster labels and ground-truth labels.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    true_labels = np.unique(y_true)
    pred_labels = np.unique(y_pred)

    # build cost matrix
    cost_matrix = np.zeros((len(pred_labels), len(true_labels)))

    for i, p in enumerate(pred_labels):
        for j, t in enumerate(true_labels):
            cost_matrix[i, j] = -np.sum((y_pred == p) & (y_true == t))

    # Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    mapping = {pred_labels[row]: true_labels[col] for row, col in zip(row_ind, col_ind)}

    new_labels = np.array([mapping[label] for label in y_pred])

    return new_labels, mapping





def generate_blobs():
    """Generate synthetic blobs(blobs) data for clustering."""
    data, ground_cluster = make_blobs(n_samples=800, n_features=4, cluster_std=1.0, random_state=42)
    return data, ground_cluster


def generate_moons():
    """Generate synthetic data(noisy moon) for clustering."""
    points, ground_cluster = make_moons(n_samples=800, noise=1.2, random_state=42)
    return points, ground_cluster

def generate_breast_cancer():
    """Generate Breast Cancer dataset for clustering."""
    
    # Load dataset
    dataset = load_breast_cancer()
    points = dataset.data
    targets = dataset.target

    reduced_points = PCA(n_components=4).fit_transform(points)

    return reduced_points, targets


def generate_iris():
    """Generate iris dataset for clustering."""
    dataset = load_iris()
    np.random.seed(42)
    data = dataset.data
    targets = dataset.target
    #selected_indices = np.random.choice(data.shape[0] , 12, replace=False)

    #return data[selected_indices], targets[selected_indices]
    return data, targets

def generate_wine():
    """Generate wine dataset for clustering."""
    dataset = load_wine()
    data = dataset.data
    labels = dataset.target
    data = PCA(n_components=4).fit_transform(data)
    return data, labels


def noisy_iris():
    """Generate a noisy version of iris dataset for clustering."""
    dataset = load_iris()
    data = dataset.data
    targets = dataset.target
    data = PCA(n_components=3).fit_transform(data)
    noise = np.random.normal(loc=0, scale=0.5, size=data.shape)
    noisy_data = data + noise
    return noisy_data, targets


def evaluate_qkmeans(data, labels, centroids, aligned,dataset_name,ground_truth=None,kmeans_labels=None,iterations=None):
    """evaluate the qkmeans alg"""

    measurements = {}

    # ite
    measurements['N_ite'] = iterations

    # sse
    sse = float(np.sum([np.linalg.norm(data[i] - centroids[labels[i]]) ** 2 for i in range(len(data))]))
    measurements['sse'] = sse

    # sil
    sil = float(silhouette_score(data, labels))
    measurements['sil'] = sil

    # Davies-Bouldin Index (DBI)
    measurements['dbi'] = float(davies_bouldin_score(data, labels))

    if ground_truth is not None:
        # Adjusted Rand Index (ARI)
        measurements['ari'] = float(adjusted_rand_score(ground_truth, labels))

        # Normalized Mutual Information (NMI)
        measurements['nmi'] = float(normalized_mutual_info_score(ground_truth, labels))

        measurements['ami'] = float(adjusted_mutual_info_score(ground_truth, labels))

        measurements['hom'] = float(homogeneity_score(ground_truth, labels))

        measurements['comp'] = float(completeness_score(ground_truth, labels))

        measurements['fmi'] = float(fowlkes_mallows_score(ground_truth, labels))

        measurements['precision'] = precision_score(ground_truth, aligned, average='weighted', zero_division=0)
        measurements['recall'] = recall_score(ground_truth, aligned, average='weighted')
        measurements['f1'] = f1_score(ground_truth, aligned, average='weighted')
        measurements['accuracy'] = accuracy_score(ground_truth, aligned)

        cm = confusion_matrix(ground_truth, aligned)
        class_labels = [f"Cluster {i}" for i in range(len(cm))]

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm,
                    annot=True,
                    fmt='g',
                    cmap='Blues',
                    xticklabels=class_labels,
                    yticklabels=class_labels)
        plt.title(f"Confusion Matrix for {dataset_name}", fontsize=16)
        plt.xlabel("Predicted Labels", fontsize=14)
        plt.ylabel("True Labels", fontsize=14)
        plt.show()
    return measurements


def visualize_clusters(data, labels, centroids, dataset_name):
    """Visualize clustering results in 3D with each centroid matching its cluster color."""

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    unique_labels = np.unique(labels)
    colors = plt.cm.Set1.colors

    for cluster_idx in unique_labels:
        cluster_points = data[labels == cluster_idx]
        color = colors[cluster_idx%10]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1],cluster_points[:, 2],
                   color=color, alpha=1,s = 30, label=f"Cluster {cluster_idx}")

        ax.scatter(centroids[cluster_idx][0], centroids[cluster_idx][1], centroids[cluster_idx][2],
                   color=color, edgecolor='black', marker='X', s=200, label=f"Centroid {cluster_idx}")

    ax.set_title(f"QK-Means Clustering ({dataset_name})")

    plt.legend()
    plt.show()



def average_dicts(dict_list):
    """Calculate the mean and std of the metrics """
    if not dict_list:
        return {}

    n = len(dict_list)

    sums = defaultdict(float)
    sums_sq = defaultdict(float)


    for d in dict_list:
        for k, v in d.items():
            sums[k] += v
            sums_sq[k] += v * v


    results = {}
    for k in sums:
        mean = sums[k] / n
        variance = (sums_sq[k] / n) - (mean * mean)


        variance = max(0.0, variance)

        std = sqrt(variance)
        results[k] = {"mean": mean, "std": std}

    return results


def test_qk_means():

    """Test quantum k-means algorithm"""
    data_sets = {
        'breast cancer wisconsin':generate_breast_cancer(),
        'wine':generate_wine(),
        'blobs': generate_blobs(),
         'noisy iris':noisy_iris(),
        'moon': generate_moons(),
        'iris': generate_iris(),

    }
    scaler = StandardScaler()
    normalizer = Normalizer(norm='max')
    
    for dataset_name, (data_points, ground_truth) in data_sets.items():
        res = []
        data_points = scaler.fit_transform(data_points)
        data_points = normalizer.fit_transform(data_points)
        k = len(set(ground_truth))

        kmean_labels, kmean_centroids = k_means(data_points, k)
        for i in range(1):
            label_list, centroids, clusters, iterations = qk_means(data_points, k, sc_thresh=.0001,
                                                                        ground_truth=ground_truth, kmena_labels=kmean_labels,
                                                                        v_measure=True)

            print('evaluations for dataset ' + dataset_name)

            aligned_kmeans, km_map = hungarian_cluster_mapping(ground_truth, kmean_labels)
            aligned_qkmeans, qkm_map = hungarian_cluster_mapping(ground_truth, label_list)

            #print("KMeans accuracy (Hungarian):", accuracy_score(ground_truth, aligned_kmeans))
            #evaluations = evaluate_qkmeans(data_points, label_list,centroids,aligned_qkmeans ,dataset_name,ground_truth, kmean_labels, iterations)
            visualize_clusters(data_points, label_list, centroids, dataset_name)
            #print(evaluations)
            print(f"experiment {i}")

            #res.append(evaluations)
        print(average_dicts(res))
if __name__ == "__main__":
    start_time = time.time()
    test_qk_means()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"time taken:{execution_time}")



