"""Exercise 2.9: Multiklassen-Klassifikation.

Aufgabenstellung:
1. Teste SVM auf Fisher Iris.
2. Vergleiche sie mit einem Random Forest.
3. Erstelle und interpretiere Konfusionsmatrizen für mehr als zwei Klassen.

Aufruf: pdm run python exercises/02_supervised_learning/exercise_2_9_multiclass.py
"""

import os
from pathlib import Path
os.environ["MPLBACKEND"] = "Agg"; os.environ["MPLCONFIGDIR"] = str(Path("results/.matplotlib").resolve())
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

RESULTS = Path("results/exercise_2_9"); SEED = 6020


def main():
    RESULTS.mkdir(parents=True, exist_ok=True); iris = load_iris()
    train_x, test_x, train_y, test_y = train_test_split(iris.data, iris.target, stratify=iris.target, random_state=SEED)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for axis, (name, model) in zip(axes, {"SVM": SVC(C=2), "Random Forest": RandomForestClassifier(n_estimators=500, random_state=SEED)}.items()):
        model.fit(train_x, train_y); print(f"{name}: Test-Accuracy = {model.score(test_x, test_y):.3f}")
        ConfusionMatrixDisplay.from_estimator(model, test_x, test_y, display_labels=iris.target_names, ax=axis, colorbar=False); axis.set_title(name)
    fig.tight_layout(); fig.savefig(RESULTS / "confusion_matrices.png", dpi=150); plt.close(fig)


if __name__ == "__main__": main()
