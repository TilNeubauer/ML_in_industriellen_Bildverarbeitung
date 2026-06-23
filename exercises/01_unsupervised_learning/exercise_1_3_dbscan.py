"""Exercise 1.3: DBSCAN auf Ellipsen und dem Iris-Datensatz.

Aufruf: pdm run python exercises/01_unsupervised_learning/exercise_1_3_dbscan.py
"""

import os
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"
os.environ["MPLCONFIGDIR"] = str(Path("results/.matplotlib").resolve())

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

RESULTS = Path("results/exercise_1_3")
SEED = 6020


def ellipses():
    rng = np.random.default_rng(SEED)
    first = rng.normal(size=(200, 2)) * (1, 0.5)
    second = rng.normal(size=(200, 2)) * (1, 0.2) + (1, -2)
    angle = np.pi / 4
    second = second @ [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    return np.vstack((first, second)), np.r_[np.zeros(200), np.ones(200)]


def save_plot(data, labels, title, name):
    plt.figure(figsize=(6, 5))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap="tab10", s=24)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(RESULTS / name, dpi=150)
    plt.close()


def best_dbscan(data, truth):
    scaled = StandardScaler().fit_transform(data)
    choices = [(eps, metric) for eps in np.linspace(0.2, 1.0, 17) for metric in ("euclidean", "manhattan")]
    scores = []
    for eps, metric in choices:
        labels = DBSCAN(eps=eps, min_samples=5, metric=metric).fit_predict(scaled)
        scores.append((adjusted_rand_score(truth, labels), eps, metric, labels))
    return max(scores, key=lambda item: item[0]), scaled


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    data, truth = ellipses()
    (score, eps, metric, labels), scaled = best_dbscan(data, truth)
    print(f"Ellipsen: ARI = {score:.3f}, eps = {eps:.2f}, Metrik = {metric}")
    save_plot(scaled, labels, "Ellipsen: DBSCAN", "ellipses.png")

    iris = load_iris()
    (score, eps, metric, labels), scaled = best_dbscan(iris.data, iris.target)
    print(f"Iris: ARI = {score:.3f}, eps = {eps:.2f}, Metrik = {metric}")
    save_plot(PCA(n_components=2, random_state=SEED).fit_transform(scaled), labels, "Iris: DBSCAN", "iris.png")


if __name__ == "__main__":
    main()
