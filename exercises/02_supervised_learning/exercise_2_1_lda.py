"""Exercise 2.1: L
    Apply the LDA algorithm to the toy example (see Figure 1.1) 
    to recover the two clusters as good as possible.

    Additionally, for a higher dimensional problem, using LDA split the Fisher Iris dataset 
    (see the introduction to part of the notes) into two clusters. 
    Try for the harder split between versicolor and virginica types of flowers.


Aufruf: pdm run python exercises/02_supervised_learning/exercise_2_1_lda.py
"""

import os
from pathlib import Path

# oeffene der Plots wird unterdrückt 
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

    data_el, target_el = ellipses()
    evaluate(data_el, target_el, "LDA: Ellipsen", "ellipses.png")
    
    iris = load_iris()
    mask = iris.target > 0
    evaluate(iris.data[mask, :2], iris.target[mask] - 1, "LDA: Versicolor vs. Virginica", "iris.png")


if __name__ == "__main__":
    main()
