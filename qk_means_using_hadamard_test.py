import numpy as np
from qiskit import QuantumCircuit, transpile, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.visualization import circuit_drawer

from qiskit.circuit.library import MCMT, RYGate
from sklearn.datasets import make_blobs, make_moons, load_iris, load_wine, load_diabetes, load_breast_cancer, load_digits
from sklearn.preprocessing import StandardScaler, Normalizer
import matplotlib.pyplot as plt
from sklearn.metrics import v_measure_score, silhouette_score, adjusted_rand_score, normalized_mutual_info_score, \
    davies_bouldin_score, homogeneity_score, adjusted_mutual_info_score, completeness_score
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

    qc.cry(-theta, ancilla_qubit, record_qubit)

    qc.h(ancilla_qubit)
   
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    thermal_relaxation_error,
    ReadoutError,
)

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
        method="density_matrix",
        noise_model=noise_model
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
        theta = (record_theta[i] + centroid_theta[i])/2
        hadamard_test(qc, record_qreg[i], centroid_qreg[i], ancilla_qreg[i], theta)

    qc.measure(ancilla_qreg, creg)
    


    simulator = get_noisy_backend_for_kmeans()
    #simulator = AerSimulator()
    transpiled_qc = transpile(qc, simulator, optimization_level=3)
    result = simulator.run(transpiled_qc, shots=shots).result()
    counts = result.get_counts()
    
    total_prob_0 = 0
    for outcome, count in counts.items():

        num_zeros = outcome.count('0')  # Count number of 0s in the outcome
        total_prob_0 += (num_zeros / n_data_qubits) * (count / shots)

    average_overlap = 2 * total_prob_0 - 1
    distance = np.sqrt(2 - 2 * average_overlap)
    print(f"Computed quantum distance: {distance}")
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

    metrics = evaluate_qkmeans(data, labels, centroids, ground_truth, kmena_labels, iteration_sims, iter_count,
                               v_measure)
    return labels, centroids, clusters, metrics


def k_means(data, k):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(data)
    centroids = kmeans.cluster_centers_
    cluster_labels = np.zeros_like(labels)
    for i in range(k):
        mask = (labels == i)
        cluster_labels[mask] = mode(labels[mask])[0]

    return labels, centroids


def generate_blobs():
    """Generate synthetic blobs(blobs) data for clustering."""
    data, ground_cluster = make_blobs(n_samples=800, n_features=4, cluster_std=1.0, random_state=42)
    return data, ground_cluster


def generate_moons():
    """Generate synthetic data(noisy moon) for clustering."""
    points, ground_cluster = make_moons(n_samples=267, noise=.1, random_state=42)
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

    data = np.dot(data, transformation_matrix.T)
    return data, ground_cluster

def generate_digits():
        
    dataset = load_digits()
    data, targets = dataset.data, dataset.target
    data = PCA(n_components=3).fit_transform(data)
    return data, targets


from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def generate_breast_cancer():
    """Generate Breast Cancer dataset for clustering."""
    
    # Load dataset
    dataset = load_breast_cancer()
    points = dataset.data          # shape (569, 30)
    targets = dataset.target       # 0 = malignant, 1 = benign

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



def generate_diabetes():
    """Generate Diabetes dataset for clustering."""
    dataset = load_diabetes()
    points = dataset.data
    targets = dataset.target
    reduced_points = PCA(n_components=3).fit_transform(points)
    return reduced_points, targets


def noisy_iris():
    dataset = load_iris()
    data = dataset.data
    targets = dataset.target
    data = PCA(n_components=3).fit_transform(data)
    noise = np.random.normal(loc=0, scale=0.5, size=data.shape)
    noisy_data = data + noise
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
        #'diabetes':generate_diabetes(),
        #'breast cancer wisconsin':generate_breast_cancer(),
         'wine':generate_wine(),
         #'blobs': generate_blobs(),
        #  'nisy':noisy_iris(),
        #  'moon': generate_moons(),
        # 'aniso': generate_aniso(),
        #'digits':generate_digits(),
        #'diabetes':generate_diabetes(),
        #'iris': generate_iris(),

    }
    scaler = StandardScaler()
    normalizer = Normalizer(norm='max')
    for dataset_name, (data_points, ground_truth) in data_sets.items():
        data_points = scaler.fit_transform(data_points)
        data_points = normalizer.fit_transform(data_points)
        k = len(set(ground_truth))
        kmean_labels, kmean_centroids = k_means(data_points, k)
        label_list, centroids, clusters, evaluations = qk_means(data_points, k, sc_thresh=.0001,
                                                                ground_truth=ground_truth, kmena_labels=kmean_labels,
                                                                v_measure=True)

        print('evaluations for dataset ' + dataset_name)
        print(evaluations)
        evaluate_algorithms_(dataset_name, kmean_labels, label_list, ground_truth)
        visualize_clusters(data_points, label_list, centroids, dataset_name)

if __name__ == "__main__":
    start_time = time.time()
    test_qk_means()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"time taken:{execution_time}")



