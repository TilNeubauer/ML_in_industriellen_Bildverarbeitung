"""Exercise 4.2: Weitere Untersuchungen mit pickle.

Aufgabenstellung: Speichere ein trainiertes Modell mit pickle, stelle es wieder
her und vergleiche Accuracy und Vorhersagen. Beachte: Pickle-Dateien aus
unbekannten Quellen dürfen wegen möglicher Codeausführung nie geladen werden.

Aufruf: pdm run python exercises/04_data_management/exercise_4_2_pickle.py
"""

from pathlib import Path
import pickle

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RESULTS = Path("results/exercise_4_2")
SEED = 6020


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    iris = load_iris()
    train_x, test_x, train_y, test_y = train_test_split(iris.data, iris.target, stratify=iris.target, random_state=SEED)
    model = make_pipeline(StandardScaler(), SVC(C=2)).fit(train_x, train_y)
    path = RESULTS / "iris_svc.pkl"
    with path.open("wb") as file:
        pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)
    with path.open("rb") as file:
        restored = pickle.load(file)  # Nur vertrauenswürdige lokale Dateien laden.
    assert (model.predict(test_x) == restored.predict(test_x)).all()
    print(f"Test-Accuracy: {restored.score(test_x, test_y):.3f}")
    print(f"Dateigröße: {path.stat().st_size} Bytes; Vorhersagen identisch: True")


if __name__ == "__main__":
    main()
