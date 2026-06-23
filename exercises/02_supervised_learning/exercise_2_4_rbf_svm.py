"""Exercise 2.4: RBF-SVM auf dem Moons-Datensatz.

Aufgabenstellung: Trainiere SVC für gamma in {0.1, 5} und C in {0.001, 1000};
plotte für alle vier Kombinationen die Trennlinie im ursprünglichen 2D-Raum.

Aufruf: pdm run python exercises/02_supervised_learning/exercise_2_4_rbf_svm.py
"""

import os
from pathlib import Path
os.environ["MPLBACKEND"] = "Agg"; os.environ["MPLCONFIGDIR"] = str(Path("results/.matplotlib").resolve())
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

RESULTS = Path("results/exercise_2_4"); SEED = 6020


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    data, labels = make_moons(n_samples=500, noise=0.30, random_state=SEED)
    train_x, test_x, train_y, test_y = train_test_split(data, labels, random_state=SEED, stratify=labels)
    grid = np.linspace(-2, 3, 300); xx, yy = np.meshgrid(grid, grid)
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    for axis, gamma, penalty in zip(axes.ravel(), (0.1, 0.1, 5, 5), (0.001, 1000, 0.001, 1000)):
        model = SVC(C=penalty, gamma=gamma).fit(train_x, train_y); score = model.score(test_x, test_y)
        axis.scatter(data[:, 0], data[:, 1], c=labels, cmap="tab10", s=12)
        axis.contour(xx, yy, model.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape), levels=[0], colors="black")
        axis.set_title(f"gamma={gamma}, C={penalty}, Test={score:.3f}"); print(f"gamma={gamma}, C={penalty}: {score:.3f}")
    fig.tight_layout(); fig.savefig(RESULTS / "rbf_grid.png", dpi=150); plt.close(fig)


if __name__ == "__main__": main()
