"""Exercise 2.6: Entscheidungsbaum auf Moons.

Aufgabenstellung: Klassifiziere die Moons mit DecisionTreeClassifier, plotte die
Entscheidungsregionen und untersuche den Einfluss von min_samples_leaf = 5.

Aufruf: pdm run python exercises/02_supervised_learning/exercise_2_6_tree_moons.py
"""

import os
from pathlib import Path
os.environ["MPLBACKEND"] = "Agg"; os.environ["MPLCONFIGDIR"] = str(Path("results/.matplotlib").resolve())
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

RESULTS = Path("results/exercise_2_6"); SEED = 6020


def main():
    RESULTS.mkdir(parents=True, exist_ok=True); data, labels = make_moons(n_samples=500, noise=0.30, random_state=SEED)
    train_x, test_x, train_y, test_y = train_test_split(data, labels, stratify=labels, random_state=SEED)
    grid = np.linspace(-2, 3, 300); xx, yy = np.meshgrid(grid, grid); fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, leaf in zip(axes, (1, 5)):
        model = DecisionTreeClassifier(min_samples_leaf=leaf, random_state=SEED).fit(train_x, train_y)
        score = model.score(test_x, test_y); print(f"min_samples_leaf={leaf}: Test-Accuracy = {score:.3f}")
        axis.contourf(xx, yy, model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape), alpha=.25, cmap="tab10")
        axis.scatter(data[:, 0], data[:, 1], c=labels, cmap="tab10", s=14); axis.set_title(f"leaf={leaf}, Test={score:.3f}")
    fig.tight_layout(); fig.savefig(RESULTS / "trees.png", dpi=150); plt.close(fig)


if __name__ == "__main__": main()
