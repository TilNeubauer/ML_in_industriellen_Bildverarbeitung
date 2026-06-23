"""Exercise 2.3: Nichtlineare SVM.

Aufgabenstellung:
1. Erweitere die lineare SVM auf eine kreisförmige 2D-Trennlinie und teste mit
   np.linspace(-1, 1, 12).
2. Klassifiziere die Moons mit PolynomialFeatures vom Grad 3.
3. Plotte in beiden Fällen die Trennlinie im ursprünglichen 2D-Raum.

Aufruf: pdm run python exercises/02_supervised_learning/exercise_2_3_nonlinear_svm.py
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
    plt.figure(figsize=(6, 5)); plt.scatter(data[:, 0], data[:, 1], c=labels, cmap="tab10", s=18)
    plt.contour(xx, yy, decision, levels=[0], colors="black")
    plt.title(title); plt.tight_layout(); plt.savefig(RESULTS / name, dpi=150); plt.close()


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    axis = np.linspace(-1, 1, 11); xx, yy = np.meshgrid(axis, axis)
    circle = np.c_[xx.ravel(), yy.ravel()]; circle_y = (np.hypot(*circle.T) < 0.5).astype(int)
    moons, moons_y = make_moons(n_samples=500, noise=0.15, random_state=SEED)
    for data, labels, title, name in ((circle, circle_y, "Kreis", "circle.png"), (moons, moons_y, "Moons", "moons.png")):
        model = make_pipeline(PolynomialFeatures(3), StandardScaler(), LinearSVC(C=10, random_state=SEED, dual="auto"))
        model.fit(data, labels); print(f"{title}: Accuracy = {model.score(data, labels):.3f}"); plot(model, data, labels, title, name)


if __name__ == "__main__": main()
