"""Exercise 2.4: 
    Recall the moons example from Section 1.3 and use an SVC classification to distinguish 
    the clusters. Look at four different results for gamma in {0.1, 5} and C in {0.001, 1000}, 
    compare Geron (2022).

    In all of the four images plot the classification line in a projection
    onto the original 2D space.


Aufruf: pdm run python exercises/02_supervised_learning/exercise_2_4_rbf_svm.py#
ergebnisse: exercise_2_4
"""

import os
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"
os.environ["MPLCONFIGDIR"] = str(Path("results/.matplotlib").resolve())

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

RESULTS = Path("results/exercise_2_4")
SEED = 6020


def plot_decision_line(axis, xx, yy, decision):
    if decision.min() <= 0 <= decision.max():
        axis.contour(xx, yy, decision, levels=[0], colors="black")
    else:
        axis.text(
            0.5,
            0.04,
            "0-Kontur nicht im Plotbereich",
            ha="center",
            va="bottom",
            transform=axis.transAxes,
        )


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)

    data, labels = make_moons(n_samples=500, noise=0.30, random_state=SEED)
    train_x, test_x, train_y, test_y = train_test_split(
        data,
        labels,
        random_state=SEED,
        stratify=labels,
    )

    grid = np.linspace(-2, 3, 300)
    xx, yy = np.meshgrid(grid, grid)
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    parameter_sets = (
        (0.1, 0.001),
        (0.1, 1000),
        (5, 0.001),
        (5, 1000),
    )

    fig, axes = plt.subplots(2, 2, figsize=(9, 7))

    for axis, (gamma, penalty) in zip(axes.ravel(), parameter_sets):
        model = SVC(C=penalty, gamma=gamma)
        model.fit(train_x, train_y)

        score = model.score(test_x, test_y)
        decision = model.decision_function(grid_points).reshape(xx.shape)

        axis.scatter(data[:, 0], data[:, 1], c=labels, cmap="tab10", s=12)
        plot_decision_line(axis, xx, yy, decision)

        axis.set_title(f"gamma={gamma}, C={penalty}, Test={score:.3f}")
        print(f"gamma={gamma}, C={penalty}: {score:.3f}")

    fig.tight_layout()
    fig.savefig(RESULTS / "rbf_grid.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
