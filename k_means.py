import numpy as np
from sklearn.datasets import make_blobs, make_moons, load_iris, load_diabetes, load_wine, load_breast_cancer
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.cluster import KMeans  # Added
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,precision_score, recall_score, f1_score, accuracy_score,classification_report

from sklearn.metrics import (
    v_measure_score,
    silhouette_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    davies_bouldin_score,
    homogeneity_score,
    adjusted_mutual_info_score,
    completeness_score,
    fowlkes_mallows_score
)

from sklearn.decomposition import PCA
from scipy.optimize import linear_sum_assignment
import time
# from evaluation import evaluate_algorithms_ # Ensure this is in your path

# =========================================================
# WRAPPER FOR SKLEARN KMEANS
# =========================================================
def k_means_sklearn(data, k, max_iters=300, ground_truth=None):
    # Initialize sklearn KMeans
    # n_init=1 and random_state=42 to match your previous setup consistency
    kmeans = KMeans(n_clusters=k, max_iter=max_iters, n_init=1, random_state=42)
    
    # Fit the model
    kmeans.fit(data)
    
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    iters = kmeans.n_iter_
    
    return labels, centroids, iters

# =========================================================
# HUNGARIAN MAPPING
# =========================================================
def hungarian_cluster_mapping(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    true_labels = np.unique(y_true)
    pred_labels = np.unique(y_pred)
    cost = np.zeros((len(pred_labels), len(true_labels)))

    for i, p in enumerate(pred_labels):
        for j, t in enumerate(true_labels):
            cost[i, j] = -np.sum((y_pred == p) & (y_true == t))

    row_ind, col_ind = linear_sum_assignment(cost)
    mapping = {pred_labels[i]: true_labels[j] for i, j in zip(row_ind, col_ind)}
    new_labels = np.array([mapping[l] for l in y_pred])

    return new_labels, mapping

def reorder_centroids(centroids, mapping):
    new_centroids = np.zeros_like(centroids)
    for old_label, new_label in mapping.items():
        new_centroids[new_label] = centroids[old_label]
    return new_centroids

# =========================================================
# METRICS (Updated to remove avg_sim)
# =========================================================
def evaluate_kmeans(data, labels, centroids, aligned,ground_truth=None, iterations=None):
    m = {}

    # SSE (using sklearn's centers)
    m['sse'] = float(np.sum([
        np.linalg.norm(data[i] - centroids[labels[i]]) ** 2
        for i in range(len(data))
    ]))

    if len(np.unique(labels)) > 1:
        m['sil'] = float(silhouette_score(data, labels))
    else:
        m['sil'] = -1

    m['dbi'] = float(davies_bouldin_score(data, labels))

    if iterations:
        m['N_ite'] = iterations

    if ground_truth is not None:
        m['ari'] = float(adjusted_rand_score(ground_truth, labels))
        m['nmi'] = float(normalized_mutual_info_score(ground_truth, labels))
        m['ami'] = float(adjusted_mutual_info_score(ground_truth, labels))
        m['hom'] = float(homogeneity_score(ground_truth, labels))
        m['comp'] = float(completeness_score(ground_truth, labels))
        m['fmi'] = float(fowlkes_mallows_score(ground_truth, labels))
        
        m['precision'] = precision_score(ground_truth, aligned, average='weighted', zero_division=0)
        m['recall']= recall_score(ground_truth, aligned, average='weighted')
        m['f1'] = f1_score(ground_truth, aligned, average='weighted')
        m['accuracy'] = accuracy_score(ground_truth, aligned)

    return m

# =========================================================
# DATASET GENERATORS (Kept same)
# =========================================================
def generate_blobs():
    data, ground = make_blobs(n_samples=800, n_features=4, cluster_std=1.0, random_state=42)
    return data, ground

def generate_moons():
    data, ground = make_moons(n_samples=800, noise=0.2, random_state=42) # reduced noise for better Kmeans testing
    return data, ground

def generate_iris():
    d = load_iris()
    return d.data, d.target

def generate_breast_cancer():
    d = load_breast_cancer()
    return PCA(n_components=4).fit_transform(d.data), d.target

def generate_wine():
    d = load_wine()
    return PCA(n_components=4).fit_transform(d.data), d.target

def generate_diabetes():
    d = load_diabetes()
    return PCA(n_components=3).fit_transform(d.data), d.target

def noisy_iris():
    d = load_iris()
    noise = np.random.normal(0, 0.5, d.data.shape)
    return d.data + noise, d.target

# =========================================================
# TEST (Updated)
# =========================================================


from math import sqrt
from collections import defaultdict


def average_dicts(dict_list):
    if not dict_list:
        return {}

    n = len(dict_list)

    # For sums and sums of squares
    sums = defaultdict(float)
    sums_sq = defaultdict(float)

    # Accumulate
    for d in dict_list:
        for k, v in d.items():
            sums[k] += v
            sums_sq[k] += v * v

    # Compute mean and std
    results = {}
    for k in sums:
        mean = sums[k] / n
        variance = (sums_sq[k] / n) - (mean * mean)

        # numerical stability fix
        variance = max(0.0, variance)

        std = sqrt(variance)
        results[k] = {"mean": mean, "std": std}

    return results

def test_k_means():
    data_sets = {
        # 'breast cancer': generate_breast_cancer(),
        # 'blobs': generate_blobs(),
        # 'moon': generate_moons(),
        # 'wine': generate_wine(),
        'iris': generate_iris(),
         #'noisy iris': noisy_iris()
    }
    list_=[]
    scaler = StandardScaler()
    normalizer = Normalizer(norm='max')
    for i in range(1):
        for name, (X, y) in data_sets.items():
            X = scaler.fit_transform(X)
            X = normalizer.fit_transform(X)

            k = len(np.unique(y))

            # Using sklearn version
            labels, centroids, iters = k_means_sklearn(X, k, ground_truth=y)

            # Apply Hungarian mapping for centroid alignment
            aligned, mapping = hungarian_cluster_mapping(y, labels)
            centroids = reorder_centroids(centroids, mapping)

            print(f"\nDataset: {name}")
            #print(f"Final iterations: {iters}")

            metrics = evaluate_kmeans(
                X,
                labels,
                centroids,
                aligned,
                ground_truth=y,
                iterations=iters
            )
            print(metrics)
            list_.append(metrics)
    #print(average_dicts(list_))
            
            # evaluate_algorithms_(name, aligned, aligned, y)

if __name__ == "__main__":
    start = time.time()
    test_k_means()
    end = time.time()
    print("\nTotal Execution Time:", end - start)