# Quantum k-Means Clustering Using Fidelity and Control Rotation

This repository contains the Python source code accompanying the research paper:

**Title**: *Quantum k-Means Clustering Using Fidelity and Control Rotation*  
**Authors**: Shahin Torabi, MohammadHadi Allayean  
**Date**: 2025

## 🧠 Project Description

This work explores quantum-enhanced versions of the k-Means clustering algorithm by leveraging fidelity-based distance metrics and controlled rotation circuits. It includes several implementations and comparative evaluations, combining both classical and quantum subroutines. The goal is to demonstrate how quantum techniques—such as the Hadamard test and Quantum Fourier Transform (QFT)—can be used to improve or alter cluster assignment performance in high-dimensional datasets.

---

## 📂 Repository Structure

| File Name                          | Description |
|-----------------------------------|-------------|
| `delta_k_means.py`                | Implements the delta-k-means clustering algorithm—a variant of classical k-means that uses an alternative distance function or update rule. |
| `elbow_method.py`                 | Script to compute the optimal number of clusters using the elbow method. Useful for validating clustering output. |
| `evaluation.py`                   | Evaluation metrics and visualization tools to compare classical and quantum clustering performance. Includes metrics like Silhouette Score, accuracy, and inertia. |
| `qk_means_using_hadamard_test.py`| Quantum k-means implementation that uses the **Hadamard test** to compute phase accumulation to find similarity between quantum states for cluster assignment. |
| `qkmeans_q_sub_qft.py`           | Quantum k-means variant that uses **quantum subroutines and QFT (Quantum Fourier Transform)** for calculating distances and updating centroids. |
| `k-means-new-metric.py`           |A variant of k-means that uses angle and phase inofrmation instead of Eculdian distance for distance calculation. |

---

## 🧪 Requirements

- Python 3.8+
- Qiskit
- NumPy
- Matplotlib
- Scikit-learn (for dataset generation and evaluation)
