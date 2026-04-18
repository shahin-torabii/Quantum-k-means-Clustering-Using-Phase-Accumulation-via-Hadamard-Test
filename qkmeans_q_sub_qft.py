import numpy as np
from qiskit import QuantumCircuit, transpile, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.visualization import circuit_drawer

from qiskit.circuit.library import MCMT, RYGate, QFT
from sklearn.datasets import make_blobs, make_moons, load_iris, load_wine, load_breast_cancer
from sklearn.preprocessing import StandardScaler, Normalizer
import matplotlib.pyplot as plt
from sklearn.metrics import v_measure_score, silhouette_score, adjusted_rand_score, normalized_mutual_info_score, \
    davies_bouldin_score, homogeneity_score, adjusted_mutual_info_score, completeness_score, accuracy_score
from sklearn.cluster import KMeans
import time
from scipy.stats import mode
from sklearn.decomposition import PCA
import math
import warnings
from scipy.optimize import linear_sum_assignment
import logging
logging.getLogger('qiskit_aer').setLevel(logging.ERROR)
logging.getLogger('qiskit').setLevel(logging.ERROR)
import random
from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    thermal_relaxation_error,
    ReadoutError,
)
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


def normalize(data):
    norm = np.linalg.norm(data)
    return data / norm if norm != 0 else data

# Function to create a controlled multi-qubit Ry rotation
def cnry(qc, theta, controls, target):

    n = len(controls)
    if n == 1:
        qc.cry(2 * theta, controls[0], target)
    else:
        mcmt_ry = MCMT(RYGate(2 * theta), num_ctrl_qubits=n, num_target_qubits=1)
        qc.append(mcmt_ry, controls + [target])

def qft(qc, qreg):
    n = len(qreg)
    for i in range(n):
        qc.h(qreg[i])
        for j in range(i + 1, n):
            theta = np.pi / (2 ** (j - i))
            qc.cp(theta, qreg[j], qreg[i])
    for i in range(n // 2):
        qc.swap(qreg[i], qreg[n - i - 1])

def inverse_qft(qc, qreg):
    n = len(qreg)
    for i in range(n // 2):
        qc.swap(qreg[i], qreg[n - i - 1])
    for i in reversed(range(n)):
        for j in reversed(range(i + 1, n)):
            theta = -np.pi / (2 ** (j - i))
            qc.cp(theta, qreg[j], qreg[i])
        qc.h(qreg[i])

# QFT-based multiplication
def qft_multiplication(qc, qreg_a, qreg_b, qreg_result):
    """
    Multiply qreg_a × qreg_b into qreg_result using Draper's adder.
    """
    n = len(qreg_a)
    m = len(qreg_result)
    if m < 2 * n:
        raise ValueError("qreg_result must be at least 2n")

    for i in range(n):  # For each bit a[i]
        # Shifted slice of qreg_result
        slice_qs = qreg_result[i:i+n]
        if len(slice_qs) < n:
            continue

        # Shifted b: we use qreg_b controlled on a[i]
        # Apply Draper's adder controlled on a[i]
        qft(qc, slice_qs)
        for k in range(n):
            for j in range(k + 1):
                theta = 2 * np.pi / (2 ** (k - j + 1))
                # Controlled on a[i] AND b[j]
                qc.mcp(theta, [qreg_a[i], qreg_b[j]], slice_qs[k])
        inverse_qft(qc, slice_qs)



def draper_qft_addition(qc, qreg_sum, qreg_add):
    n = len(qreg_sum)
    qft(qc, qreg_sum)
    for i in range(n):
        for j in range(i + 1):
            theta = 2 * np.pi / (2 ** (i - j + 1))
            qc.cp(theta, qreg_add[j], qreg_sum[i])
    inverse_qft(qc, qreg_sum)


# -----------------------------
# Controlled Draper QFT Adder
# -----------------------------
def controlled_draper_qft_addition(qc, control, qreg_sum, qreg_add):


    n = len(qreg_sum)

    # Step 1: QFT
    qft(qc, qreg_sum)

    # Step 2: Controlled phase rotations
    for i in range(n-1):
        for j in range(i + 1):
            theta = 2 * np.pi / (2 ** (i - j + 1))    
            # Apply phase only if (control AND qreg_add[j]) == 1
            qc.mcp(theta, [control, qreg_add[j]], qreg_sum[i])

    # Step 3: inverse QFT
    inverse_qft(qc, qreg_sum)


# -----------------------------
# Absolute Value (Two’s Complement)
# -----------------------------
def quantum_abs(qc, qreg, one):
    """
    Computes |x| in-place assuming two's complement encoding.

    qreg : input register (modified in-place)
    one  : auxiliary register encoding |1⟩ (same size as qreg)
    """

    n = len(qreg)
    sign = qreg[n - 1]  # MSB

    # Step 1: Conditional bit flip (invert all bits if negative)
    for i in range(n -  1):
        qc.cx(sign, qreg[i])

    # Step 2: Conditional +1 (two’s complement)
    controlled_draper_qft_addition(qc, sign, qreg, one)
def get_noisy_backend_for_kmeans():

    noise_model = NoiseModel()

    # ---- Depolarizing gate errors (REDUCED) ----
    # before: 0.001 / 0.01
    p1q = 0.0005      # mild single-qubit noise
    p2q = 0.005       # mild CX noise (MOST IMPORTANT)

    error_1q = depolarizing_error(p1q, 1)
    error_2q = depolarizing_error(p2q, 2)

    single_qubit = ['rz', 'sx', 'x', 'h', 'ry']
    two_qubit = ['cx']

    noise_model.add_all_qubit_quantum_error(error_1q, single_qubit)
    noise_model.add_all_qubit_quantum_error(error_2q, two_qubit)

    # ---- Thermal relaxation (WEAKER EFFECT) ----
    # increase coherence relative to gate duration
    t1, t2 = 200e3, 160e3   # longer coherence times
    gate_time_1q = 45        # faster gates
    gate_time_2q = 220

    relax_1q = thermal_relaxation_error(t1, t2, gate_time_1q)
    relax_single = thermal_relaxation_error(t1, t2, gate_time_2q)
    relax_2q = relax_single.tensor(relax_single)

    noise_model.add_all_qubit_quantum_error(relax_1q, single_qubit)
    noise_model.add_all_qubit_quantum_error(relax_2q, two_qubit)

    # ---- Readout error (REDUCED) ----
    # before: 2%
    p_flip = 0.008

    readout = ReadoutError([
        [1 - p_flip, p_flip],
        [p_flip, 1 - p_flip]
    ])

    noise_model.add_all_qubit_readout_error(readout)

    return AerSimulator(
        method='matrix_product_state',
        noise_model=noise_model,
        seed_simulator=42
    )

def distance_circuit_qft(record, centroid, f=2, shots=8192):
    """
    Compute Euclidean distance over all dimensions using
    repeated 1D QFT distance circuits.
    """

    total_expectation = 0

    for dim in range(len(record)):
        # --- SAME CODE AS BEFORE, BUT PER DIMENSION ---
        scale = 2 ** f
        r_int = round(record[dim] * scale)
        c_int = round(centroid[dim] * scale)

        n = max(len(bin(max(r_int, c_int))[2:]), f + 1)
        m = 2 * n

        qr = QuantumRegister(n, "record")
        qcen = QuantumRegister(n, "centroid")
        qd = QuantumRegister(n, "d_copy")
        qsq = QuantumRegister(m, "square")
        qs = QuantumRegister(m, "sum")
        one = QuantumRegister(n, "one")
        cr = ClassicalRegister(m, "c")

        qc = QuantumCircuit(qr, qd, qsq, qs, one, cr, qcen)

        # Encode record
        for i, bit in enumerate(format(r_int, f'0{n}b')[::-1]):
            if bit == '1':
                qc.x(qr[i])

        # Encode centroid
        for i, bit in enumerate(format(c_int, f'0{n}b')[::-1]):
            if bit == '1':
                qc.x(qcen[i])

        # --- QFT subtraction ---
        qft(qc, qr)
        for i in range(n):
            for j in range(i + 1):
                if (c_int >> j) & 1:
                    theta = -2 * np.pi / (2 ** (i - j + 1))
                    qc.cp(theta, qr[i], qcen[i])
        inverse_qft(qc, qr)

        # --- ABS ---
        qc.x(one[0])
        quantum_abs(qc, qr, one)

        # --- Copy ---
        for i in range(n):
            qc.cx(qr[i], qd[i])

        # --- Square ---
        qft_multiplication(qc, qr, qd, qsq)

        # --- Accumulate ---
        draper_qft_addition(qc, qs, qsq)

        qc.measure(qs[::-1], cr)

        # --- Run ---
        simulator = AerSimulator() # IMPORTANT
        transpiled_qc = transpile(qc, optimization_level=3)
        result = simulator.run(transpiled_qc, shots=shots).result()
        counts = result.get_counts()

        expectation = 0
        for key, count in counts.items():
            val = int(key, 2) / (scale ** 2)
            expectation += val * (count / shots)

        total_expectation += expectation

    distance = np.sqrt(total_expectation)
    print(f"Distance: {distance}")
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

                distance = distance_circuit_qft(record, centroid)
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

    metrics = evaluate_qkmeans(data, labels, centroids, ground_truth, kmena_labels, iteration_sims, iter_count, v_measure)
    return labels, centroids, clusters, metrics


def k_means(data, k):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(data)
    centroids = kmeans.cluster_centers_
    cluster_labels = np.zeros_like(labels)  # Create an array of zeros
    for i in range(k):  # 3 clusters for 3 classes
        mask = (labels == i)  # Find all samples in cluster i
        cluster_labels[mask] = mode(labels[mask])[0]  # Assign most frequent true label

    return labels, centroids


def hungarian_cluster_mapping(y_true, y_pred):
    """
    Uses Hungarian algorithm to find optimal 1-to-1 mapping
    between cluster labels and ground-truth labels.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # get unique labels
    true_labels = np.unique(y_true)
    pred_labels = np.unique(y_pred)

    # build cost matrix (negative overlap for maximization)
    cost_matrix = np.zeros((len(pred_labels), len(true_labels)))

    for i, p in enumerate(pred_labels):
        for j, t in enumerate(true_labels):
            cost_matrix[i, j] = -np.sum((y_pred == p) & (y_true == t))

    # Hungarian algorithm (minimize cost → maximize correct matches)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # create mapping
    mapping = {pred_labels[row]: true_labels[col] for row, col in zip(row_ind, col_ind)}

    # relabel predictions
    new_labels = np.array([mapping[label] for label in y_pred])

    return new_labels, mapping


def generate_blobs():
    """Generate synthetic blobs(blobs) data for clustering."""
    data, ground_cluster = make_blobs(n_samples=27, n_features=3, cluster_std=1.0, random_state=42)
    return data, ground_cluster


def generate_moons():
    """Generate synthetic data(noisy moon) for clustering."""
    points, ground_cluster = make_moons(n_samples=21, noise=.1, random_state=42)
    return points, ground_cluster


def generate_aniso():
    """Generate anisotropic data for clustering."""
    data, ground_cluster = make_blobs(n_samples=36, n_features=3, cluster_std=1.0, random_state=42)

    # Fix: Use a 3x3 transformation matrix for 3D data
    transformation_matrix = np.array([
        [0.6, -0.6, 0.2],
        [0.4, 0.8, -0.3],
        [0.1, -0.2, 0.9]
    ])

    data = np.dot(data, transformation_matrix.T)  # Transpose ensures proper dot product
    return data, ground_cluster


def generate_breast_cancer():
    """Generate Breast Cancer dataset for clustering."""

    # Load dataset
    dataset = load_breast_cancer()
    points = dataset.data  # shape (569, 30)
    targets = dataset.target  # 0 = malignant, 1 = benign

    reduced_points = PCA(n_components=4).fit_transform(points)

    return reduced_points, targets


def generate_iris():
    dataset = load_iris()
    data = dataset.data
    targets = dataset.target
    return data, targets

def generate_wine():
    dataset = load_wine()
    data = dataset.data
    labels = dataset.target
    data = PCA(n_components=4).fit_transform(data)
    return data, labels

def noisy_iris():
    dataset = load_iris()
    data = dataset.data
    targets = dataset.target
    data = PCA(n_components=3).fit_transform(data)
    noise = np.random.normal(loc=0, scale=0.5, size=data.shape)
    noisy_data = data +noise
    return noisy_data, targets

def evaluate_qkmeans(data, labels, centroids, ground_truth=None, kmeans_labels=None,
                     iteration_sims=None, iterations=None, v_measure=True):
    """evaluate the qkmeans alg with N_ite, avg_sim, sil, sse, v_measure """

    measurements = {}

    # ite
    measurements['N_ite'] = iterations

    # avg_sim
    if iteration_sims is not None:
        avg_sim = float(iteration_sims / iterations)
        measurements['avg_sim'] = avg_sim
    else:
        measurements['avg_sim'] = None

    # sse
    sse = float(np.sum([np.linalg.norm(data[i] - centroids[labels[i]]) ** 2 for i in range(len(data))]))
    measurements['sse'] = sse

    # sil
    sil = float(silhouette_score(data, labels))
    measurements['sil'] = sil

    if v_measure:
        # v_measure
        if ground_truth is not None:
            vm = float(v_measure_score(ground_truth, labels))
            measurements['vm'] = vm
        else:
            measurements['vm'] = None

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

        # V-measure

    return measurements


def visualize_clusters(data, labels, centroids, dataset_name):
    """Visualize clustering results in 3D with each centroid matching its cluster color."""

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')  # Create a 3D plot

    unique_labels = np.unique(labels)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))  # Generate colormap

    for cluster_idx, color in zip(unique_labels, colors):
        cluster_points = data[labels == cluster_idx]  # Get points of the current cluster
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1],cluster_points[:, 2],
                   color=color, alpha=0.5, label=f"Cluster {cluster_idx}")

        ax.scatter(centroids[cluster_idx][0], centroids[cluster_idx][1], centroids[cluster_idx][2],
                   color=color, edgecolor='black', marker='X', s=200, label=f"Centroid {cluster_idx}")

    ax.set_title(f"Q k-Means Clustering ({dataset_name})")

    plt.legend()
    plt.show()

def test_qk_means():
    from evaluation import evaluate_algorithms_
    """Test quantum k-means algorithm"""
    data_sets = {
        #'breast cancer': generate_breast_cancer(),
        # 'iris':generate_iris(),
        'wine':generate_wine(),
        #'blobs': generate_blobs(),
        # 'moon': generate_moons(),
        #  'aniso': generate_aniso(),
         
         #'iris': generate_iris(),
    }
    scaler = StandardScaler()
    normalizer = Normalizer(norm='max')
    for dataset_name, (data_points, ground_truth) in data_sets.items():
        data_points = scaler.fit_transform(data_points)
        data_points = normalizer.fit_transform(data_points)
        k = len(set(ground_truth))

        kmean_labels, kmean_centroids = k_means(data_points, k)
        label_list, centroids, clusters, evaluations= qk_means(data_points, k, sc_thresh=.0001,
                                                              ground_truth= ground_truth, kmena_labels=kmean_labels, v_measure=True)

        print('evaluations for dataset '+dataset_name)
        print(evaluations)
        aligned_kmeans, km_map = hungarian_cluster_mapping(ground_truth, kmean_labels)
        aligned_qkmeans, qkm_map = hungarian_cluster_mapping(ground_truth, label_list)

        print("KMeans accuracy (Hungarian):", accuracy_score(ground_truth, aligned_kmeans))
        print("QKMeans accuracy (Hungarian):", accuracy_score(ground_truth, aligned_qkmeans))

        evaluate_algorithms_(dataset_name, aligned_kmeans, aligned_qkmeans, ground_truth)
        visualize_clusters(data_points, label_list, centroids, dataset_name)


if __name__ == "__main__":
    start_time = time.time()
    test_qk_means()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"time taken:{execution_time}")



