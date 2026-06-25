"""Exercise 4.1:
    Try to rewrite the model and check the resulting score after recovery vs. 
    the original score for the following modifications.

    1. Remove the probability=True for SVC.
    2. Replace SVC by LinearSVC.
    3. Remove the SVC all together and replace it with a LogisticRegression classifier.

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


def create_model():
    return make_pipeline(
        StandardScaler(),
        SVC(C=2),
    )


def write_report(path, model_path, accuracy, predictions_identical):
    text = (
        "Exercise 4.1: SVC Persistence\n"
        "\n"
        f"Gespeichertes Modell: {model_path.name}\n"
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

    model_path = RESULTS / "iris_svc.joblib"
    report_path = RESULTS / "iris_svc_report.txt"

    joblib.dump(model, model_path)
    restored_model = joblib.load(model_path)

    original_predictions = model.predict(test_x)
    restored_predictions = restored_model.predict(test_x)
    predictions_identical = (original_predictions == restored_predictions).all()

    accuracy = restored_model.score(test_x, test_y)
    write_report(report_path, model_path, accuracy, predictions_identical)

    print(f"Test-Accuracy: {accuracy:.3f}")
    print(f"Vorhersagen nach Wiederherstellung identisch: {predictions_identical}")
    print(f"Lesbarer Bericht: {report_path}")

    assert predictions_identical


if __name__ == "__main__":
    main()
