"""Exercise 2.5: Nichtlineare SVM-Regression.

    Aufgabenstellung: Passe die vorgegebenen verrauschten quadratischen Beobachtungen
    mit verschiedenen Kernel-Funktionen und Polynomialgraden an.

Aufruf: pdm run python exercises/02_supervised_learning/exercise_2_5_svr.py
Ergebnisse: exercise_2_5
"""

import os
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"
os.environ["MPLCONFIGDIR"] = str(Path("results/.matplotlib").resolve())

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR

RESULTS = Path("results/exercise_2_5")
SEED = 6020


def create_observations():
    # Aus aufgbenstellung
    np.random.seed(SEED)

    number_of_observations = 100
    x = 6 * np.random.rand(number_of_observations) - 3
    y = 0.5 * x**2 + x + 2 + np.random.randn(number_of_observations)

    return x, y


def create_models():
    return {
        "linear": SVR(kernel="linear", C=10),
        "poly_2": SVR(kernel="poly", degree=2, C=10, epsilon=0.1),
        "rbf": SVR(kernel="rbf", C=10, gamma="scale"),
    }


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)

    x, y = create_observations()
    grid = np.linspace(-3, 3, 500)
    models = create_models()

    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, c="black", s=14, label="Beobachtungen")

    for name, model in models.items():
        model.fit(x[:, None], y)

        predictions = model.predict(x[:, None])
        mse = mean_squared_error(y, predictions)
        print(f"{name}: MSE = {mse:.3f}")

        grid_predictions = model.predict(grid[:, None])
        plt.plot(grid, grid_predictions, label=name)

    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS / "svr.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
