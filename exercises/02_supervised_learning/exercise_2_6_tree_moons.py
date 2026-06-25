"""Exercise 2.6: 
    Recall the moons example from Section 1.3 and use a DecisionTreeClassifier 
    classification to distinguish the clusters and plot the decision splits.

    Play around with the parameters, e.g. min_samples_leaf = 5 and see how 
    this influences the score for a test set.


Aufruf: pdm run python exercises/02_supervised_learning/exercise_2_6_tree_moons.py
Ergebnisse: exercise 2_6
"""

import os
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"
os.environ["MPLCONFIGDIR"] = str(Path("results/.matplotlib").resolve())

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

RESULTS = Path("results/exercise_2_6")
SEED = 6020


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)

    data, labels = make_moons(n_samples=500, noise=0.30, random_state=SEED)
    train_x, test_x, train_y, test_y = train_test_split(
        data,
        labels,
        stratify=labels,
        random_state=SEED,
    )

    grid = np.linspace(-2, 3, 300)
    xx, yy = np.meshgrid(grid, grid)
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    fig, axes = plt.subplots(1, 4, figsize=(10, 4))
    #print(len(axes))

    for axis, leaf in zip(axes, (1, 5, 10, 20)):
        model = DecisionTreeClassifier(min_samples_leaf=leaf, random_state=SEED)
        model.fit(train_x, train_y)

        score = model.score(test_x, test_y)
        print(f"min_samples_leaf={leaf}: Test-Accuracy = {score:.3f}")

        predictions = model.predict(grid_points).reshape(xx.shape)

        axis.contourf(xx, yy, predictions, alpha=0.25, cmap="tab10")
        axis.scatter(data[:, 0], data[:, 1], c=labels, cmap="tab10", s=14)
        axis.set_title(f"leaf={leaf}, Test={score:.3f}")

    fig.tight_layout()
    fig.savefig(RESULTS / "trees.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
