"""Exercise 2.1: LDA für die Ellipsen und Iris (Versicolor/Virginica).

Aufgabenstellung:
1. Wende LDA auf das Beispiel mit zwei Ellipsen an.
2. Trenne im Fisher-Iris-Datensatz Versicolor und Virginica mittels LDA.

Aufruf: pdm run python exercises/02_supervised_learning/exercise_2_1_lda.py
"""

import os
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"
os.environ["MPLCONFIGDIR"] = str(Path("results/.matplotlib").resolve())

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

RESULTS = Path("results/exercise_2_1")
SEED = 6020


def ellipses():
    rng = np.random.default_rng(SEED)
    first = rng.normal(size=(200, 2)) * (1, 0.5)
    second = rng.normal(size=(200, 2)) * (1, 0.2) + (1, -2)
    angle = np.pi / 4
    second = second @ [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    return np.vstack((first, second)), np.r_[np.zeros(200), np.ones(200)]


def evaluate(data, target, title, filename):
    train_x, test_x, train_y, test_y = train_test_split(
        data, target, test_size=0.25, stratify=target, random_state=SEED
    )
    model = LinearDiscriminantAnalysis().fit(train_x, train_y)
    print(f"{title}: Test-Accuracy = {accuracy_score(test_y, model.predict(test_x)):.3f}")
    labels = model.predict(data)
    plt.figure(figsize=(6, 5))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap="tab10", s=24)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(RESULTS / filename, dpi=150)
    plt.close()


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    evaluate(*ellipses(), "LDA: Ellipsen", "ellipses.png")
    iris = load_iris()
    mask = iris.target > 0
    evaluate(iris.data[mask, :2], iris.target[mask] - 1, "LDA: Versicolor vs. Virginica", "iris.png")


if __name__ == "__main__":
    main()
