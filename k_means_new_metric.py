import numpy as np
from sklearn.datasets import make_blobs, make_moons, load_iris, load_diabetes, load_wine, load_digits,load_breast_cancer
from sklearn.preprocessing import Normalizer, StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import v_measure_score, silhouette_score, adjusted_rand_score, normalized_mutual_info_score, davies_bouldin_score, homogeneity_score,adjusted_mutual_info_score, completeness_score
import time
from sklearn.manifold import TSNE
from sklearn.metrics import fowlkes_mallows_score
import math

def euclidean_distance(a,b):
    d = np.sqrt(np.sum((a - b)**2))
    return d
#
# class Kmeans:
#
#     # construct method for hyperparameter initialization
#     def __init__(self, k=3, max_iter=100, tol=1e-06):
#         self.k = k
#         self.max_iter = max_iter
#         self.tol = tol
#
#     # randomly picks the initial centroids from the input data
#     def pick_centers(self, X):
#         centers_idxs = np.random.choice(self.n_samples, self.k)
#         return X[centers_idxs]

    # # finds the closest centroid for each data point
    # def get_closest_centroid(self, x, centroids):
    #     distances = [euclidean_distance(x, centroid) for centroid in centroids]
    #     return np.argmin(distances)
    #
    # # creates a list with lists containing the idxs of each cluster
    # # creates a list with lists containing the idxs of each cluster
    # def create_clusters(self, centroids, X):
    #     clusters = [[] for _ in range(self.k)]
    #     labels = np.empty(self.n_samples, dtype=int)  # Explicitly make it an integer array
    #
    #     for i, x in enumerate(X):
    #         centroid_idx = self.get_closest_centroid(x, centroids)
    #         clusters[centroid_idx].append(i)
    #         labels[i] = centroid_idx  # This will now store integer values
    #
    #     return clusters, labels
    #
    # # calculates the centroids for each cluster using the mean value
    # def compute_centroids(self, clusters, X):
    #     centroids = np.empty((self.k, self.n_features))
    #     for i, cluster in enumerate(clusters):
    #         centroids[i] = np.mean(X[cluster], axis=0)
    #
    #     return centroids
    #
    # # helper function to verify if the centroids changed significantly
    # def is_converged(self, old_centroids, new_centroids):
    #     distances = [euclidean_distance(old_centroids[i], new_centroids[i]) for i in range(self.k)]
    #     return (sum(distances) < self.tol)
    #
    # # method to train the data, find the optimized centroids and label each data point according to its cluster
    # def fit_predict(self, X):
    #     self.n_samples, self.n_features = X.shape
    #     self.centroids = self.pick_centers(X)
    #
    #     for i in range(self.max_iter):
    #         self.clusters, self.labels = self.create_clusters(self.centroids, X)
    #         new_centroids = self.compute_centroids(self.clusters, X)
    #         if self.is_converged(self.centroids, new_centroids):
    #             break
    #         self.centroids = new_centroids
    #
    # # method for evaluating the intracluster variance of the optimization
    # def clustering_errors(self, X):
    #     cluster_values = [X[cluster] for cluster in self.clusters]
    #     squared_distances = []
    #     # calculation of total squared Euclidean distance
    #     for i, cluster_array in enumerate(cluster_values):
    #         squared_distances.append(np.sum((cluster_array - self.centroids[i])**2))
    #
    #     total_error = np.sum(squared_distances)
    #     return total_error
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix


def euclidean_distance(a, b):
    return np.linalg.norm(a - b)


def k_means(data, k, max_iters=10, tol=1e-4, ground_truth = None):
    np.random.seed(42)
    iteration_sims = 0
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]
    prev_centroids = centroids.copy()
    labels = np.zeros(data.shape[0],dtype=int)
    iter = 0

    for ite in range(max_iters):
        iter += 1
        # Assign labels based on closest centroid
        for i in range(data.shape[0]):
            theta_record = 2 * np.arcsin(data[i])  # shape (n_features,)
            theta_centroids = 2 * np.arcsin(centroids)  # shape (k, n_features)
            
            # Compute average overlap per centroid
            avg_overlap = np.mean(np.cos((theta_record + theta_centroids)/4), axis=1)  # shape (k,)
            
            # Distance formula
            distances = np.sqrt(2 - 2 * avg_overlap)
            
            # Assign label
            labels[i] = np.argmin(distances)

        # Update centroids based on the mean of the points in each cluster
        for j in range(k):
            points = data[labels == j]
            if len(points) > 0:
                centroids[j] = points.mean(axis=0)

        # Check for convergence
        iteration_sims += adjusted_rand_score(labels, ground_truth)
        if np.linalg.norm(centroids - prev_centroids) < tol:
            break
        prev_centroids = centroids.copy()

    return labels.astype(int), centroids, iter, iteration_sims


def match_labels_with_hungarian(true_labels, predicted_labels, k):
    """
    Match the predicted labels with the true labels using the Hungarian Algorithm.
    """
    # Create confusion matrix: rows = predicted labels, columns = true labels
    cm = confusion_matrix(true_labels, predicted_labels, labels=np.arange(k))

    # Apply the Hungarian algorithm (maximize the matching)
    row_ind, col_ind = linear_sum_assignment(-cm)  # We negate the matrix to maximize

    # Create a new set of labels based on the optimal assignment
    matched_labels = np.zeros_like(predicted_labels)
    for i, j in zip(row_ind, col_ind):
        matched_labels[predicted_labels == i] = j  # Reassign predicted label to the true label

    return matched_labels


def evaluate_k_means_with_hungarian(data, ground_truth, k, max_iters=10, tol=1e-4):
    """
    Perform K-means clustering, then apply the Hungarian algorithm to match labels to true labels.
    """
    # Perform K-means
    predicted_labels, centroids, ite, iter_sims = k_means(data, k, max_iters, tol, ground_truth)

    # Use Hungarian algorithm to match predicted labels to true labels
    matched_labels = match_labels_with_hungarian(ground_truth, predicted_labels, k)

    # Evaluate performance
    confusion = confusion_matrix(ground_truth, matched_labels)
    print("Confusion Matrix:\n", confusion)

    # Calculate additional metrics (e.g., precision, recall, F1-score, etc.)
    # Here, you can compute other metrics as needed (you may use scikit-learn's metrics)

    return matched_labels, centroids, ite, confusion, iter_sims


# Example Usage (assuming ground_truth is available)


def default_kmeans(data, n_clusters = 3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(data)
    labels = kmeans.labels_
    return labels, kmeans.cluster_centers_


def generate_blobs():
    """Generate synthetic blobs(blobs) data for clustering."""
    data, ground_cluster = make_blobs(n_samples=800, n_features=4, cluster_std=1.0, random_state=42)
    return data, ground_cluster

def generate_blobs2():
    """Generate synthetic data(blobs) for clustering."""
    data, ground_cluster = make_blobs(n_samples=800, n_features=2, cluster_std=2.0, random_state=28)
    return data, ground_cluster

def generate_moons():
    """Generate synthetic data(noisy moon) for clustering."""
    points, ground_cluster = make_moons(n_samples=800, noise=.05, random_state=42)
    return points, ground_cluster

def generate_aniso():
    """Generate anisotropic data for clustering."""
    data, ground_cluster = make_blobs(n_samples=700, n_features=2, cluster_std=1.0, random_state=42)
    transformation_matrix = [[.6, -.6], [.4, .8]]
    data = np.dot(data, np.array(transformation_matrix))
    return data, ground_cluster

def generate_iris():
    """Generate Iris dataset for clustering."""
    dataset = load_iris()
    points = dataset.data
    targets = dataset.target
    return points, targets

def generate_breast_cancer():
    """Generate Breast Cancer dataset for clustering."""
    
    # Load dataset
    dataset = load_breast_cancer()
    points = dataset.data          # shape (569, 30)
    targets = dataset.target       # 0 = malignant, 1 = benign

    reduced_points = PCA(n_components=4).fit_transform(points)

    return reduced_points, targets



def generate_wine():

    dataset = load_wine()
    data = dataset.data
    targets = dataset.target
    data = PCA(n_components=4).fit_transform(data)
    return data, targets

def generate_diabetes():
    """Generate Diabetes dataset for clustering."""
    dataset = load_diabetes()
    points = dataset.data
    targets = dataset.target
    reduced_points = PCA(n_components=3).fit_transform(points)
    return reduced_points, targets


def generate_digits():
    dataset = load_digits()
    data, targets = dataset.data, dataset.target

    data = PCA(n_components=3).fit_transform(data)
    return data, targets



def noisy_iris():
    dataset = load_iris()
    data = dataset.data
    targets = dataset.target
    noise = np.random.normal(loc=0, scale=0.5, size=data.shape)
    noisy_data = data +noise
    return noisy_data, targets



def evaluate_kmeans(data, labels, centroids, ground_truth=None):
    """Evaluate K-Means clustering with multiple metrics."""

    measurements = {}


    #sse
    sse = float(np.sum([np.linalg.norm(data[i] - centroids[labels[i]]) ** 2 for i in range(len(data))]))
    measurements['sse'] = sse

    #sil
    sil = float(silhouette_score(data, labels))
    measurements['sil'] = sil

    if ground_truth is not None:
        vm = float(v_measure_score(ground_truth, labels))
        measurements['vm'] = vm
    else:
        measurements['vm'] = None


    measurements['dbi'] = float(davies_bouldin_score(data, labels))

    if ground_truth is not None:

        measurements['ari'] = float(adjusted_rand_score(ground_truth, labels))


        measurements['nmi'] = float(normalized_mutual_info_score(ground_truth, labels))

        measurements['ami'] = float(adjusted_mutual_info_score(ground_truth, labels))

        measurements['hom'] = float(homogeneity_score(ground_truth, labels))

        measurements['fmi'] = float(fowlkes_mallows_score(ground_truth, labels))

        measurements['comp'] = float(completeness_score(ground_truth, labels))




    return measurements


# def evaluate_avg_sim(data, labels, centroids):
#     """Evaluate average similarity for real datasets."""
#     avg_sim = 0.0
#     for i in range(len(data)):
#         avg_sim += np.dot(data[i], centroids[labels[i]]) / (np.linalg.norm(data[i]) * np.linalg.norm(centroids[labels[i]]))
#     avg_sim /= len(data)
#     return float(avg_sim)
def evaluate_avg_sim(ground_truth, labels):
    """Evaluate average similarity using Adjusted Rand Index (ARI)."""
    return float(adjusted_rand_score(ground_truth, labels))


def visualize_clusters(data, labels, centroids, dataset_name):
    """Visualize clustering results in 3D with each centroid matching its cluster color."""

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')  # Create a 3D plot

    unique_labels = np.unique(labels)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))  # Generate colormap

    for cluster_idx, color in zip(unique_labels, colors):
        cluster_points = data[labels == cluster_idx]  # Get points of the current cluster
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1],
                   color=color, alpha=0.5, label=f"Cluster {cluster_idx}")

        ax.scatter(centroids[cluster_idx][0], centroids[cluster_idx][1],
                   color=color, edgecolor='black', marker='X', s=200, label=f"Centroid {cluster_idx}")

    ax.set_title(f"k-Means Clustering ({dataset_name})")

    plt.legend()
    plt.show()


def test_k_means():
    """Test k-means algorithm with evaluation metrics."""
    from evaluation import evaluate_algorithms_
    data_sets = {
        'breast cancer':generate_breast_cancer(),
        'blobs': generate_blobs(),
        #'blobs2': generate_blobs2(),
        'moon': generate_moons(),
        #'aniso': generate_aniso(),
        'wine':generate_wine(),
        'iris': generate_iris(),
        #'digits':generate_digits(),
        #'diabetes': generate_diabetes()
       'noisy iris': noisy_iris()
    }
    scaler = StandardScaler()
    normalizer = Normalizer(norm='max')
    for dataset_name, (data_points, cluster) in data_sets.items():
        data_points = scaler.fit_transform(data_points)
        data_points = normalizer.fit_transform(data_points)
        k = len(set(cluster))

        labels, centroids, ite, iter_sims = k_means(
            data_points, k, ground_truth=cluster
        )
        # k_means.fit_predict(data_points)
        # labels = k_means.labels  
        # centroids = k_means.centroids
        print(f"Dataset: {dataset_name}")
        # labels, centroids, ite,_ , iter_sims= evaluate_k_means_with_hungarian(data_points,cluster, k, max_iters=10)
        # print("Iterations:", ite)
        # print("avg_Sim:", iter_sims/ite)
        print("avg_Sim is :", evaluate_avg_sim(labels, cluster))
        evaluate_algorithms_(dataset_name, labels, labels, cluster)
        print(evaluate_kmeans(data_points, labels, centroids,cluster))
        visualize_clusters(data_points, labels, centroids, dataset_name)

    real_data_sets = {
        'iris': generate_iris(),
        'diabetes': generate_diabetes()
    }
    #
    # for dataset_name, (data_points, cluster) in data_sets.items():
    #     data_points = scaler.fit_transform(data_points)
    #     data_points = normalizer.fit_transform(data_points)
    #     k = set(len(cluster))
    #     labels, centroids, ite = k_means(data_points, k, max_iters=10)
    #
    #     metrics = evaluate_k_means(data_points, labels, centroids)
    #     avg_sim = evaluate_avg_sim(data_points, labels, centroids)
    #
    #     print(f"Dataset: {dataset_name}")
    #     print("Iterations:", ite)
    #     print("Evaluation Metrics:", metrics)
    #     print("Average Similarity:", avg_sim)
    #     visualize_clusters(data_points, labels, centroids, dataset_name)



def visualize_embed(X, labels, title):
    tsne = TSNE(n_components=3, init="random")
    X = tsne.fit_transform(X)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection = "3d")
    scatter = ax.scatter(X[:,0], X[:, 1], X[:,2], c= labels, cmap="viridis")

    # cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
    # cbar.set_label('Class ID')

    ax.set_xlabel('Dim 1')
    ax.set_ylabel('Dim 2')
    ax.set_zlabel('Dim 3')
    ax.set_title(title)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    start = time.time()
    test_k_means()
    end = time.time()
    print("time taken:", start - end)


