"""Exercise 4.2:
    For the loaded model, switch to soft voting by calling

    Use joblib to persist and load the module, also check the file size.

    Some user defined functions can cause problems for pickle try persisting the model with 
    cloudpickle and test with the user defined kernel function 
    rbf = lambda x, y: np.exp(1e-2 * np.abs(x@y.T)).

Aufruf: pdm run python exercises/04_data_management/exercise_4_2_pickle.py
"""

import pickle
from pathlib import Path

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RESULTS = Path("results/exercise_4_2")
SEED = 6020


def create_model():
    return make_pipeline(
        StandardScaler(),
        SVC(C=2),
    )


def save_pickle(model, path):
    with path.open("wb") as file:
        pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path):
    with path.open("rb") as file:
        return pickle.load(file)


def write_report(path, model_path, accuracy, predictions_identical):
    text = (
        "Exercise 4.2: Pickle Persistence\n"
        "\n"
        f"Gespeichertes Modell: {model_path.name}\n"
        f"Dateigroesse: {model_path.stat().st_size} Bytes\n"
        f"Test-Accuracy nach Wiederherstellung: {accuracy:.3f}\n"
        f"Vorhersagen identisch: {predictions_identical}\n"
    )
    path.write_text(text, encoding="utf-8")


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)

    iris = load_iris()
    train_x, test_x, train_y, test_y = train_test_split(
        iris.data,
        iris.target,
        stratify=iris.target,
        random_state=SEED,
    )

    model = create_model()
    model.fit(train_x, train_y)

    model_path = RESULTS / "iris_svc.pkl"
    report_path = RESULTS / "iris_svc_pickle_report.txt"

    save_pickle(model, model_path)
    restored_model = load_pickle(model_path)

    original_predictions = model.predict(test_x)
    restored_predictions = restored_model.predict(test_x)
    predictions_identical = (original_predictions == restored_predictions).all()

    accuracy = restored_model.score(test_x, test_y)
    write_report(report_path, model_path, accuracy, predictions_identical)

    print(f"Test-Accuracy: {accuracy:.3f}")
    print(f"Dateigroesse: {model_path.stat().st_size} Bytes")
    print(f"Vorhersagen identisch: {predictions_identical}")
    print(f"Lesbarer Bericht: {report_path}")

    assert predictions_identical


if __name__ == "__main__":
    main()
