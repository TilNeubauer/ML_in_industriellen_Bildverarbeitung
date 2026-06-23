"""Exercise 4.1: SVC-Modellpersistenz testen.

Aufgabenstellung: Speichere einen SVC, lade ihn wieder und prüfe, ob die
Vorhersagen nach der Wiederherstellung identisch sind.

Aufruf: pdm run python exercises/04_data_management/exercise_4_1_svc_persistence.py
"""

from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RESULTS = Path("results/exercise_4_1")
SEED = 6020


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    iris = load_iris()
    train_x, test_x, train_y, test_y = train_test_split(iris.data, iris.target, stratify=iris.target, random_state=SEED)
    model = make_pipeline(StandardScaler(), SVC(C=2)).fit(train_x, train_y)
    path = RESULTS / "iris_svc.joblib"
    joblib.dump(model, path)
    restored = joblib.load(path)
    identical = (model.predict(test_x) == restored.predict(test_x)).all()
    print(f"Test-Accuracy: {restored.score(test_x, test_y):.3f}")
    print(f"Vorhersagen nach Wiederherstellung identisch: {identical}")
    assert identical


if __name__ == "__main__":
    main()
