import numpy as np
from sklearn.datasets import make_blobs, make_moons, load_iris, load_diabetes
from sklearn.preprocessing import StandardScaler, Normalizer
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, v_measure_score, adjusted_rand_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def init_centroids(data, k):
    """Initialize centroids using k-means++ strategy."""
    centroids = []
    first_centroid = data[np.random.choice(data.shape[0], replace=False)]
    centroids.append(first_centroid)
    for _ in range(k - 1):
        dists = np.min([np.linalg.norm(data - c, axis=1) for c in centroids], axis=0)
        probabilities = dists ** 2 / np.sum(dists ** 2)
        new_centroid = data[np.random.choice(data.shape[0], p=probabilities)]
        centroids.append(new_centroid)
    return centroids


def delta_k_means(data, k, delta, max_iter=10, sc_thresh=0.0001, ground_truth = None, kmean_lables = None, v_measure = True):
    centroids = init_centroids(data, k)
    labels = np.zeros(len(data), dtype=int)
    iter_count = 0
    iteration_sims= 0
    for _ in range(max_iter):
        new_clusters = [[] for _ in range(k)]

        for i, point in enumerate(data):
            distances = [np.linalg.norm(point - c) for c in centroids]
            nearest_centroid_idx = np.argmin(distances)

            possible_indices = [
                idx for idx, dist in enumerate(distances)
                if abs(dist ** 2 - distances[nearest_centroid_idx] ** 2) <= delta
            ]

            assigned_label = np.random.choice(possible_indices)
            labels[i] = assigned_label
            new_clusters[assigned_label].append(point)

        new_centroids = []
        for cluster in new_clusters:
            if cluster:
                new_centroids.append(np.mean(cluster, axis=0))
            else:

                new_centroids.append(init_centroids(data, 1)[0])

        if kmean_lables is not None:
            sim = np.mean(labels == kmean_lables) * 100
            iteration_sims+=adjusted_rand_score(labels, kmean_lables)

        if np.linalg.norm(np.array(new_centroids) - np.array(centroids), ord='fro') < sc_thresh:

            break

        centroids = new_centroids
        iter_count += 1
        metrics = evaluate_qkmeans(data, labels, centroids, ground_truth, kmean_lables, iteration_sims, iter_count, v_measure)
    print("iterations   =  ", iter)
    return labels, centroids, new_clusters, metrics


def k_means(data, k):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    return labels, centroids



def generate_blobs():
    """Generate synthetic blobs(blobs) data for clustering."""
    data, ground_cluster = make_blobs(n_samples=700, n_features=3, cluster_std=1.0, random_state=42)
    return data, ground_cluster


def generate_blobs2():
    """Generate synthetic data(blobs) for clustering."""
    data, ground_cluster = make_blobs(n_samples=700, n_features=2, cluster_std=2.0, random_state=28)
    return data, ground_cluster


def generate_moons():
    """Generate synthetic data(noisy moon) for clustering."""
    points, ground_cluster = make_moons(n_samples=700, noise=.05, random_state=42)
    return points, ground_cluster


def generate_aniso():
    """Generate anisotropic data for clustering."""
    data, ground_cluster = make_blobs(n_samples=700, n_features=2, cluster_std=1.0, random_state=42)
    transformation_matrix = [[.6, -.6], [.4, .8]]
    data = np.dot(data, np.array(transformation_matrix))
    return data, ground_cluster


def generate_iris():
    dataset = load_iris()
    points = dataset.data
    targets = dataset.target
    dataset_reduced_points = PCA(n_components=3).fit_transform(points)
    return dataset_reduced_points, targets


def generate_diabetes():
    dataset = load_diabetes()
    points = dataset.data
    targets = dataset.target
    reduced_points = PCA(n_components=3).fit_transform(points)
    return reduced_points, targets


def evaluate_qkmeans(data, labels, centroids, ground_truth=None, kmeans_labels=None,
                     iteration_sims=None, iterations=None, v_measure = True):
    """evaluate the qkmeans alg with N_ite, avg_sim, sil, sse, v_measure """

    measurements = {}

    #ite
    measurements['N_ite'] = iterations

    #avg_sim
    if iteration_sims is not None:
        avg_sim = float(iteration_sims/iterations)
        measurements['avg_sim'] = avg_sim
    else:
        measurements['sim'] = None

    #sse
    sse = float(np.sum([np.linalg.norm(data[i] - centroids[labels[i]]) ** 2 for i in range(len(data))]))
    measurements['sse'] = sse

    #sil
    sil = float(silhouette_score(data, labels))
    measurements['sil'] = sil

    #v_measure
    if v_measure:
        if ground_truth is not None:
            vm = float(v_measure_score(ground_truth, labels))
            measurements['vm'] = vm
        else:
            measurements['vm'] = None


    return measurements



def visualize_clusters(data, labels, centroids, dataset_name):
    """Visualize clustering results in 3D with each centroid matching its cluster color."""

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')  # Create a 3D plot

    unique_labels = np.unique(labels)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))  # Generate colormap

    for cluster_idx, color in zip(unique_labels, colors):
        cluster_points = data[labels == cluster_idx]  # Get points of the current cluster
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2],
                   color=color, alpha=0.5, label=f"Cluster {cluster_idx}")

        ax.scatter(centroids[cluster_idx][0], centroids[cluster_idx][1], centroids[cluster_idx][2],
                   color=color, edgecolor='black', marker='X', s=200, label=f"Centroid {cluster_idx}")

    ax.set_title(f"Q k-Means Clustering ({dataset_name})")

    plt.legend()
    plt.show()

def test_delta_k_means():
    """Test quantum k-means algorithm"""
    data_sets = {
         'blobs': generate_blobs(),
        'blobs2': generate_blobs2(),
        'moon': generate_moons(),
        'aniso': generate_aniso()
    }
    scaler = StandardScaler()
    normalizer = Normalizer(norm='max')
    for dataset_name, (data_points, cluster) in data_sets.items():
        data_points = scaler.fit_transform(data_points)
        data_points = normalizer.fit_transform(data_points)
        k = len(set(cluster))
        delta = .14
        kmean_labels, kmean_centroids = k_means(data_points, k)
        label_list, centroids, clusters, metrices = delta_k_means(data_points, k, delta,10, .0001, cluster, kmean_labels, True)
        print('evaluations for real dataset ' + dataset_name)
        print(metrices)
        visualize_clusters(data_points, label_list, centroids, dataset_name)

    real_data_sets = {
            'iris': generate_iris(),
        'diabetes': generate_diabetes()
    }

    for dataset_name, (data_points, cluster) in real_data_sets.items():
        k = 3
        delta = .195
        data_points = scaler.fit_transform(data_points)
        data_points = normalizer.fit_transform(data_points)
        kmean_labels, kmean_centroids = k_means(data_points, k)
        label_list, centroids, clusters, evaluations = delta_k_means(data_points, k, delta,sc_thresh=.0001,
                                                                    ground_truth=cluster,                                                                    kmean_lables=kmean_labels, v_measure=False)
        print('evaluations for real dataset ' + dataset_name)
        print(evaluations)
        visualize_clusters(data_points, label_list, centroids, dataset_name)


if __name__ == "__main__":
    test_delta_k_means()
