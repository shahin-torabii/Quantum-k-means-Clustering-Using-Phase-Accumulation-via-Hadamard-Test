from qk_means import qk_means, generate_iris
from k_means import k_means
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, Normalizer
import numpy as np

def elbow_method(data_points, method='classical', max_k=10):
    """
    Implement the elbow method to determine the optimal number of clusters for k-Means or qk-Means.
    """
    sse = []
    range_of_k = range(2, max_k + 1)

    for k in range_of_k:
        if method == 'classical':
            labels, centroids = k_means(data_points, k)
            sse.append(np.sum([np.linalg.norm(data_points[i] - centroids[labels[i]])**2 for i in range(len(data_points))]))
        elif method == 'quantum':
            _, centroids, _, evaluations = qk_means(data_points, k, max_iter=10)
            sse.append(evaluations['sse'])
        else:
            raise ValueError("Method must be 'classical' or 'quantum'.")

    plt.figure(figsize=(8, 5))
    plt.plot(range_of_k, sse, marker='o', label=method + ' SSE')
    plt.title(f'Elbow Method ({method} k-Means)')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('SSE')
    plt.grid()
    plt.legend()
    plt.show()


def test_elbow_method():
    scaler = StandardScaler()
    normalizer = Normalizer(norm='max')

    iris_points, targets = generate_iris()
    iris_points = scaler.fit_transform(iris_points)
    iris_points = normalizer.fit_transform(iris_points)

    elbow_method(iris_points, method='classical', max_k=8)
    elbow_method(iris_points, method='quantum', max_k=8)



if __name__ == "__main__":
    test_elbow_method()
