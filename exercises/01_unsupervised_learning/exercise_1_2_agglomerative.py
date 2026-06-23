"""Exercise 1.2: Agglomeratives Clustering auf Iris und Moons.

Aufruf: pdm run python exercises/01_unsupervised_learning/exercise_1_2_agglomerative.py
"""

import os
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"
os.environ["MPLCONFIGDIR"] = str(Path("results/.matplotlib").resolve())

import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import load_iris, make_moons
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

RESULTS = Path("results/exercise_1_2")
SEED = 6020


def plot(data, labels, title, name):
    points = PCA(n_components=2, random_state=SEED).fit_transform(data)
    plt.figure(figsize=(6, 5))
    plt.scatter(points[:, 0], points[:, 1], c=labels, cmap="tab10", s=24)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(RESULTS / name, dpi=150)
    plt.close()


def cluster(data, truth, n_clusters, title, name):
    scaled = StandardScaler().fit_transform(data)
    labels = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward").fit_predict(scaled)
    print(f"{title}: ARI = {adjusted_rand_score(truth, labels):.3f}")
    plot(scaled, labels, title, name)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    iris = load_iris()
    cluster(iris.data[:, :2], iris.target, 3, "Iris (2 Merkmale)", "iris_2d.png")
    cluster(iris.data, iris.target, 3, "Iris (4 Merkmale)", "iris_4d.png")
    moons, truth = make_moons(n_samples=1_000, noise=0.05, random_state=SEED)
    cluster(moons, truth, 2, "Moons", "moons.png")


if __name__ == "__main__":
    main()
