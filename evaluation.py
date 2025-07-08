import numpy as np
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.metrics import confusion_matrix,precision_score, recall_score, f1_score, accuracy_score,classification_report
from q_kmeans_using_swap import qk_means, generate_blobs, generate_aniso, generate_moons
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns


def k_means(data, k):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    return labels, centroids


def evaluate_algorithms():
    """
    Evaluate k-Means vs qk-Means clustering on multiple datasets.
    Reports confusion matrix and performance metrics for each dataset.
    """
    data_sets = {
        'moon': generate_moons(),
        'aniso': generate_aniso()
    }
    scaler = StandardScaler()
    normalizer = Normalizer(norm='max')

    for dataset_name, (data_points, ground_truth) in data_sets.items():

        data_points = scaler.fit_transform(data_points)
        data_points = normalizer.fit_transform(data_points)
        k = len(set(ground_truth))  # Number of clusters


        kmeans_labels, _, _ = k_means(data_points, k)

        qkmeans_labels, _, _, _ = qk_means(data_points, k, ground_truth=ground_truth, kmena_lables=kmeans_labels)

        cm = confusion_matrix(ground_truth, qkmeans_labels)
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

        precision = precision_score(ground_truth, qkmeans_labels, average='weighted', zero_division=0)
        recall = recall_score(ground_truth, qkmeans_labels, average='weighted')
        f1 = f1_score(ground_truth, qkmeans_labels, average='weighted')
        accuracy = accuracy_score(ground_truth, qkmeans_labels)
        report = classification_report(ground_truth,qkmeans_labels)


        print(f"\nDataset: {dataset_name}")
        print(f"Confusion Matrix:\n{cm}")
       # print(f"TP: {total_tp}, FP: {total_fp}, FN: {total_fn}, TN: {total_tn}")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")
        print("\nClassification Report (Precision, Recall, F1):")
        print(report)



def evaluate_algorithms_(dataset_name, kmeans_labels, qkmeans_labels, ground_truth):
    """
    Evaluate k-Means vs qk-Means clustering on multiple datasets.
    Reports confusion matrix and performance metrics for each dataset.
    """



    cm = confusion_matrix(ground_truth,qkmeans_labels)
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

    precision = precision_score(ground_truth, qkmeans_labels, average='weighted', zero_division=0)
    recall = recall_score(ground_truth, qkmeans_labels, average='weighted')
    f1 = f1_score(ground_truth, qkmeans_labels, average='weighted')
    accuracy = accuracy_score(ground_truth, qkmeans_labels)
    report = classification_report(ground_truth,qkmeans_labels)

    print(f"\nDataset: {dataset_name}")
    print(f"Confusion Matrix:\n{cm}")
    # print(f"TP: {total_tp}, FP: {total_fp}, FN: {total_fn}, TN: {total_tn}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")
    print("\nClassification Report (Precision, Recall, F1):")
    print(report)


if __name__ == "__main__":
    evaluate_algorithms()
