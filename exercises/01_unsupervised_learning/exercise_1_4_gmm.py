"""Exercise 1.4: Gaussian-Mixture-Modelle auf Ellipsen, Iris und Moons.

Aufruf: pdm run python exercises/01_unsupervised_learning/exercise_1_4_gmm.py
"""

import os
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"
os.environ["MPLCONFIGDIR"] = str(Path("results/.matplotlib").resolve())

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris, make_moons
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.preprocessing import StandardScaler

RESULTS = Path("results/exercise_1_4")
SEED = 6020


def plot(data, labels, title, name):
    points = PCA(n_components=2, random_state=SEED).fit_transform(data)
    plt.figure(figsize=(6, 5))
    plt.scatter(points[:, 0], points[:, 1], c=labels, cmap="tab10", s=24)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(RESULTS / name, dpi=150)
    plt.close()


def run(data, truth, clusters, title, name):
    data = StandardScaler().fit_transform(data)
    models = {
        "GMM": GaussianMixture(n_components=clusters, n_init=10, random_state=SEED),
        "Bayesian GMM": BayesianGaussianMixture(n_components=6, n_init=3, random_state=SEED),
    }
    for model_name, model in models.items():
        labels = model.fit_predict(data)
        print(f"{title}, {model_name}: ARI = {adjusted_rand_score(truth, labels):.3f}")
        plot(data, labels, f"{title}: {model_name}", f"{name}_{model_name.replace(' ', '_')}.png")


def ellipses():
    rng = np.random.default_rng(SEED)
    first = rng.normal(size=(200, 2)) * (1, 0.5)
    second = rng.normal(size=(200, 2)) * (1, 0.2) + (1, -2)
    angle = np.pi / 4
    second = second @ [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    return np.vstack((first, second)), np.r_[np.zeros(200), np.ones(200)]


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    data, truth = ellipses()
    run(data, truth, 2, "Ellipsen", "ellipses")
    iris = load_iris()
    run(iris.data, iris.target, 3, "Iris", "iris")
    moons, truth = make_moons(n_samples=1_000, noise=0.05, random_state=SEED)
    run(moons, truth, 2, "Moons", "moons")


if __name__ == "__main__":
    main()
