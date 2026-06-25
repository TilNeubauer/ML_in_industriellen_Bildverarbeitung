"""Exercise 1.1 (Apply the -means algorithm to other datasets) As an exercise,
     to get some practice for using -means, apply the algorithm to some other datasets to see how it performs.

    We already looked at the Fisher Iris dataset (see the introduction this part of the notes)
    and discussed some basic features. Try with only two dimensional data, as well as four dimensional 
    data to find the three clusters.

    For Figure 1.7 we will look at the moons dataset. Try working out the clusters in this case.

    Apply the algorithm to the cats and dogs (in wavelet basis, see the introduction this part of the notes)
    for various principal components and find the two clusters.


Aufruf: pdm run python exercises/01_unsupervised_learning/exercise_1_1_kmeans.py
Ergebnisse: exeecise_1_1
"""

import os
from pathlib import Path

# oeffene der Plots wird unterdrückt 
os.environ["MPLBACKEND"] = "Agg"
os.environ["MPLCONFIGDIR"] = str(Path("results/.matplotlib").resolve())

import matplotlib.pyplot as plt
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
    print(f"Moons: ARI = {score:.3f}")
    save_scatter(data, labels, "Moons: K-means", "moons.png")


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    iris_experiment()
    moons_experiment()
    


if __name__ == "__main__":
    main()
