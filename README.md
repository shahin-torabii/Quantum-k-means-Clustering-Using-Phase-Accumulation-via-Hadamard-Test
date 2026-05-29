# Quantum k-Means Clustering Using Hadamard-Test-Based Similarity Estimation

This repository contains the Python source code accompanying the research paper:

**Title**: *Quantum k-Means Clustering Using Hadamard-Test-Based Similarity Estimation*  
**Authors**: MohammadHadi Allayean, Shahin Torabi, Mehdi Allayean  
**Date**: 2026  

---

## 🧠 Project Description

This work presents a **hybrid quantum-classical k-means clustering algorithm** designed to address limitations of prior quantum k-means approaches, particularly in the **NISQ (Noisy Intermediate-Scale Quantum)** era.

Instead of fully quantumizing the algorithm, this research focuses on quantum-enhancing the **most computationally expensive step of k-means: distance computation**. By keeping centroid updates classical and only quantumizing the assignment phase, the approach avoids deep circuits and improves near-term feasibility.

Two quantum strategies for distance/similarity estimation are introduced:

### 1️⃣ QFT-Based Quantum Subtraction

- Implements Euclidean distance computation using **quantum subtraction**
- Uses **Quantum Fourier Transform**-based quantum addition
- Simulates classical subtraction in a quantum setting
- Provides a principled quantum arithmetic approach
- Experimental results show:
  - High circuit depth
  - Weak clustering performance
  - Limited practicality on NISQ devices

### 2️⃣ Hadamard-Test-Based Similarity Estimation

- Uses the **Hadamard test** to estimate a rotation-based similarity score
- Converts similarity into a heuristic dissimilarity metric
- Requires shallower circuits
- Demonstrates clustering performance comparable to classical k-means
- More suitable for near-term quantum hardware

---

## 🚀 Main Contributions

- ✅ A **hybrid quantum-classical clustering framework** inspired by k-means  
- ✅ A **Hadamard-test-based similarity estimation method** for cluster assignment  
- ✅ A **quantum Euclidean distance estimation circuit** based on QFT subtraction  


## 📂 Repository Structure

| File Name | Description |
|------------|-------------|
| `elbow_method.py` | Computes the optimal number of clusters using the elbow method. |
| `qk_means_using_hadamard_test.py` | Implements the **Hadamard-test-based quantum k-means (QK-H)** for similarity estimation and cluster assignment. |
| `qkmeans_q_sub_qft.py` | Implements the **QFT-based quantum subtraction (QK-QFT)** for Euclidean distance estimation. |
| `k-means-new-metric.py` | Classical-inspired variant using the metric introduced in **QK-H** instead of Euclidean distance. |
| `k-means.py` | Implementation of classical k-means. |
---

## 🧪 Requirements

- Python 3.8+
- Qiskit
- NumPy
- Matplotlib
- Scikit-learn

Install dependencies:
```bash
pip install qiskit numpy matplotlib scikit-learn
```
---

## 📊 Experimental Findings
Experiments were conducted on six benchmark datasets and evaluated using multiple clustering quality metrics. The proposed quantum approaches were systematically compared against:

- Classical k-means (Euclidean distance)
- Classical k-means using the proposed angle-based metric


Key findings from the experimental evaluation include:
- **QK-QFT** suffers from large circuit depth and reduced clustering quality.
- **QK-H** achieves clustering results comparable to classical k-means.
- The Hadamard-test-based approach provides a practical path toward implementing clustering on near-term quantum hardware.

---

## 🎯 Research Impact

This work demonstrates that:

- Fully quantum k-means may not yet be practical for NISQ devices.
- Hybrid approaches that selectively quantumize costly subroutines are more feasible.
- Hadamard-test-based similarity estimation is a promising direction for near-term quantum machine learning.
