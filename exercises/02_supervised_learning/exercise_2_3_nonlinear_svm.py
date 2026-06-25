"""Exercise 2.3:
    Extend the above findings to an example in 2D with a circular classification line. 
    Create tests data of your classification by changing to np.linspace(-1, 1, 12).

    Recall the moons example from Section 1.3 and use a degree PolynomialFeatures for classification.
    In both cases, plot the classification line in a projection onto the original 2D space.


Aufruf: pdm run python exercises/02_supervised_learning/exercise_2_3_nonlinear_svm.py
Ergebnisse: exercise_2_3
"""

import os
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"
os.environ["MPLCONFIGDIR"] = str(Path("results/.matplotlib").resolve())
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import LinearSVC

RESULTS = Path("results/exercise_2_3")
SEED = 6020


def plot(model, data, labels, title, name):
    grid = np.linspace(-1.5, 1.5, 300)
    xx, yy = np.meshgrid(grid, grid)
    decision = model.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.figure(figsize=(6, 5))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap="tab10", s=18)
    plt.contour(xx, yy, decision, levels=[0], colors="black")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(RESULTS / name, dpi=150)
    plt.close()


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)

    axis = np.linspace(-1, 1, 30)
    xx, yy = np.meshgrid(axis, axis)
    circle = np.c_[xx.ravel(), yy.ravel()]
    circle_y = (np.hypot(*circle.T) < 0.5).astype(int)

    moons, moons_y = make_moons(n_samples=500, noise=0.15, random_state=SEED)

    datasets = (
        (circle, circle_y, "Kreis", "circle.png"),
        (moons, moons_y, "Moons", "moons.png"),
    )

    for data, labels, title, name in datasets:
        model = make_pipeline(
            PolynomialFeatures(3),
            StandardScaler(),
            LinearSVC(C=10, random_state=SEED, dual="auto"),
        )
        model.fit(data, labels)

        accuracy = model.score(data, labels)
        print(f"{title}: Accuracy = {accuracy:.3f}")

        plot(model, data, labels, title, name)


if __name__ == "__main__":
    main()
