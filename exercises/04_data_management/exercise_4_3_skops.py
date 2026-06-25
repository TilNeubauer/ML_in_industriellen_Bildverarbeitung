"""Exercise 4.3: 

Aufgabenstellung:
1. The included PCA works with float64, this is not necessary, 
can we reduce the file size by switching to float16? Hint: look at voting_clf[0].components_.

2. TApply the self defined kernel from Exercise 4.2 and test the load/recover cycle.

Aufruf: pdm run python exercises/04_data_management/exercise_4_3_skops.py
"""

from copy import deepcopy
from pathlib import Path

import numpy as np
import skops.io as sio
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RESULTS = Path("results/exercise_4_3")
SEED = 6020


def custom_rbf_kernel(x, y):
    return np.exp(1e-2 * np.abs(x @ y.T))


def create_voting_model():
    voting = VotingClassifier(
        [
            ("logreg", LogisticRegression(max_iter=2_000, random_state=SEED)),
            ("svc", SVC(C=2, random_state=SEED)),
            ("tree", DecisionTreeClassifier(random_state=SEED)),
        ]
    )
    return make_pipeline(PCA(n_components=2, random_state=SEED), voting)


def save_and_size(model, path):
    sio.dump(model, path)
    return path.stat().st_size


def trusted_load(path):
    unknown_types = sio.get_untrusted_types(file=path)
    return sio.load(path, trusted=unknown_types), unknown_types


def write_report(path, lines):
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compare_pca_file_sizes(train_x, train_y):
    voting_clf = create_voting_model()
    voting_clf.fit(train_x, train_y)

    float64_path = RESULTS / "voting_pca_float64.skops"
    float64_size = save_and_size(voting_clf, float64_path)

    float16_clf = deepcopy(voting_clf)
    float16_clf[0].components_ = float16_clf[0].components_.astype(np.float16)

    float16_path = RESULTS / "voting_pca_float16.skops"
    float16_size = save_and_size(float16_clf, float16_path)

    reduction = 100 * (1 - float16_size / float64_size)
    return float64_size, float16_size, reduction


def test_custom_kernel(train_x, test_x, train_y, test_y):
    model = make_pipeline(SVC(C=2, kernel=custom_rbf_kernel))
    model.fit(train_x, train_y)

    model_path = RESULTS / "svc_custom_kernel.skops"
    save_and_size(model, model_path)

    restored_model, unknown_types = trusted_load(model_path)

    original_predictions = model.predict(test_x)
    restored_predictions = restored_model.predict(test_x)
    predictions_identical = (original_predictions == restored_predictions).all()
    accuracy = restored_model.score(test_x, test_y)

    return model_path, unknown_types, accuracy, predictions_identical


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)

    iris = load_iris()
    train_x, test_x, train_y, test_y = train_test_split(
        iris.data,
        iris.target,
        stratify=iris.target,
        random_state=SEED,
    )

    float64_size, float16_size, reduction = compare_pca_file_sizes(train_x, train_y)
    model_path, unknown_types, accuracy, predictions_identical = test_custom_kernel(
        train_x,
        test_x,
        train_y,
        test_y,
    )

    report_lines = [
        "Exercise 4.3: skops.io Untersuchungen",
        "",
        f"PCA float64 Datei: {float64_size} Bytes",
        f"PCA float16 Datei: {float16_size} Bytes",
        f"Redution: {reduction:.1f} %",
        "",
        f"Custom-Kernel Modell: {model_path.name}",
        f"Unbekante Typen: {unknown_types}",
        f"Test-Accuracy nach Wiederherstellung: {accuracy:.3f}",
        f"Vorhersagen identisch: {predictions_identical}",
    ]
    write_report(RESULTS / "skops_investigation_report.txt", report_lines)

    for line in report_lines:
        print(line)

    assert predictions_identical


if __name__ == "__main__":
    main()
