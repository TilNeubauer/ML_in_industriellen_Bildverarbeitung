"""Exercise 4.3: Weitere Untersuchungen mit skops.io.

Aufgabenstellung: Speichere ein trainiertes Scikit-Learn-Modell mit skops.io,
prüfe unbekannte Typen, lade es wieder und vergleiche die Vorhersagen.

Aufruf: pdm run python exercises/04_data_management/exercise_4_3_skops.py
"""

from pathlib import Path

import skops.io as sio
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RESULTS = Path("results/exercise_4_3")
SEED = 6020


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    iris = load_iris()
    train_x, test_x, train_y, test_y = train_test_split(iris.data, iris.target, stratify=iris.target, random_state=SEED)
    model = make_pipeline(StandardScaler(), SVC(C=2)).fit(train_x, train_y)
    path = RESULTS / "iris_svc.skops"
    sio.dump(model, path)
    unknown = sio.get_untrusted_types(file=path)
    assert not unknown, f"Unbekannte Typen: {unknown}"
    restored = sio.load(path, trusted=unknown)
    assert (model.predict(test_x) == restored.predict(test_x)).all()
    print(f"Test-Accuracy: {restored.score(test_x, test_y):.3f}")
    print("Keine unbekannten Typen; Vorhersagen identisch: True")


if __name__ == "__main__":
    main()
