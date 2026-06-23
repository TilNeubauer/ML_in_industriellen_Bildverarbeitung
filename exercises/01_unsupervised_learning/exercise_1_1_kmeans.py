"""Exercise 1.1: K-means auf Iris, Moons und Cats-vs-Dogs-Wavelets.

Aufruf: pdm run python exercises/01_unsupervised_learning/exercise_1_1_kmeans.py
Die Abbildungen und Kennzahlen werden unter results/exercise_1_1/ gespeichert.
"""

import os
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen

os.environ["MPLBACKEND"] = "Agg"
os.environ["MPLCONFIGDIR"] = str(Path("results/.matplotlib").resolve())

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris, make_moons
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RESULTS = Path("results/exercise_1_1")
SEED = 6020


def save_scatter(data, labels, title, filename, centers=None):
    """Speichert eine zweidimensionale Cluster-Darstellung."""
    plt.figure(figsize=(6, 5))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap="tab10", s=25)
    if centers is not None:
        plt.scatter(centers[:, 0], centers[:, 1], c="black", marker="*", s=180)
    plt.title(title)
    plt.xlabel("Komponente 1")
    plt.ylabel("Komponente 2")
    plt.tight_layout()
    plt.savefig(RESULTS / filename, dpi=150)
    plt.close()


def fit_kmeans(data, clusters):
    """Standardisiert Merkmale und führt K-means aus."""
    model = make_pipeline(StandardScaler(), KMeans(clusters, n_init=20, random_state=SEED))
    labels = model.fit_predict(data)
    return model, labels


def iris_experiment():
    iris = load_iris()
    for dimensions in (2, 4):
        features = iris.data[:, :dimensions]
        _, labels = fit_kmeans(features, clusters=3)
        score = adjusted_rand_score(iris.target, labels)
        print(f"Iris ({dimensions} Merkmale): ARI = {score:.3f}")

        plot_data = PCA(n_components=2, random_state=SEED).fit_transform(features)
        save_scatter(plot_data, labels, f"Iris: K-means mit {dimensions} Merkmalen", f"iris_{dimensions}d.png")


def moons_experiment():
    data, truth = make_moons(n_samples=1_000, noise=0.05, random_state=SEED)
    _, labels = fit_kmeans(data, clusters=2)
    score = adjusted_rand_score(truth, labels)
    print(f"Moons: ARI = {score:.3f} (K-means trennt nicht-konvexe Monde nur eingeschränkt)")
    save_scatter(data, labels, "Moons: K-means", "moons.png")


def download_mat(url, key):
    """Lädt eine Wavelet-Matrix direkt aus den Kursdaten."""
    with urlopen(url, timeout=30) as response:
        return loadmat(BytesIO(response.read()))[key]


def cats_dogs_experiment():
    base = "https://github.com/dynamicslab/databook_python/raw/refs/heads/master/DATA/"
    cats = download_mat(base + "catData_w.mat", "cat_wave")
    dogs = download_mat(base + "dogData_w.mat", "dog_wave")
    data = np.concatenate((dogs, cats), axis=1).T
    truth = np.r_[np.zeros(dogs.shape[1], dtype=int), np.ones(cats.shape[1], dtype=int)]

    for components in (2, 4, 10):
        embedding = PCA(n_components=components, random_state=SEED).fit_transform(data)
        _, labels = fit_kmeans(embedding, clusters=2)
        score = adjusted_rand_score(truth, labels)
        print(f"Cats vs. Dogs ({components} PCA-Komponenten): ARI = {score:.3f}")
        save_scatter(
            embedding[:, :2], labels,
            f"Cats vs. Dogs: K-means nach PCA ({components} Komponenten)",
            f"cats_dogs_pca_{components}.png",
        )


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    iris_experiment()
    moons_experiment()
    cats_dogs_experiment()


if __name__ == "__main__":
    main()
