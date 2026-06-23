"""Exercise 2.7: Baumregression.

Aufgabenstellung: Fitte die Beobachtungen aus Exercise 2.5 mit verschiedenen
max_depth-Werten sowie ohne max_depth und min_samples_leaf = 10.

Aufruf: pdm run python exercises/02_supervised_learning/exercise_2_7_tree_regression.py
"""

import os
from pathlib import Path
os.environ["MPLBACKEND"] = "Agg"; os.environ["MPLCONFIGDIR"] = str(Path("results/.matplotlib").resolve())
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor

RESULTS = Path("results/exercise_2_7"); SEED = 6020


def main():
    RESULTS.mkdir(parents=True, exist_ok=True); rng = np.random.default_rng(SEED)
    x = 6 * rng.random(100) - 3; y = .5 * x**2 + x + 2 + rng.normal(size=100); grid = np.linspace(-3, 3, 500)
    plt.figure(figsize=(7, 5)); plt.scatter(x, y, c="black", s=14, label="Beobachtungen")
    for depth in (2, 5, None):
        model = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=10, random_state=SEED).fit(x[:, None], y)
        print(f"max_depth={depth}: MSE = {mean_squared_error(y, model.predict(x[:, None])):.3f}")
        plt.plot(grid, model.predict(grid[:, None]), label=f"depth={depth}")
    plt.legend(); plt.tight_layout(); plt.savefig(RESULTS / "tree_regression.png", dpi=150); plt.close()


if __name__ == "__main__": main()
