"""Exercise 2.7: 
    We can use a decision tree for regression. Have a look at docs and use the findings to fit the 
    observations of Exercise 2.5 with various max_depth values and no value here but limiting 
    min_samples_leaf=10.


Aufruf: pdm run python exercises/02_supervised_learning/exercise_2_7_tree_regression.py
ergebnisse: exercise_2_7
"""

import os
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"
os.environ["MPLCONFIGDIR"] = str(Path("results/.matplotlib").resolve())

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor

RESULTS = Path("results/exercise_2_7")
SEED = 6020


def create_observations():
    rng = np.random.default_rng(SEED)

    number_of_observations = 100
    x = 6 * rng.random(number_of_observations) - 3
    y = 0.5 * x**2 + x + 2 + rng.normal(size=number_of_observations)

    return x, y


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)

    x, y = create_observations()
    grid = np.linspace(-3, 3, 500)

    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, c="black", s=14, label="Beobachtungen")

    #for depth in (2, 5, None):
    for depth in (1, 2, 5, 10, None):
        model = DecisionTreeRegressor(
            max_depth=depth,
            min_samples_leaf=10,
            random_state=SEED,
        )
        model.fit(x[:, None], y)

        predictions = model.predict(x[:, None])
        mse = mean_squared_error(y, predictions)
        print(f"max_depth={depth}: MSE = {mse:.3f}")

        grid_predictions = model.predict(grid[:, None])
        plt.plot(grid, grid_predictions, label=f"depth={depth}")

    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS / "tree_regression.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
