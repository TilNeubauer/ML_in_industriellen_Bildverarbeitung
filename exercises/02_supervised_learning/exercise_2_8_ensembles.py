"""Exercise 2.8: Ensemble-Methoden.

Aufgabenstellung: Vergleiche auf Moons Hard-/Soft-Voting (LDA, SVM, Baum),
Bagging, Random Forest und Stacking. Bestimme außerdem Feature-Importances
für Iris sowie Dogs-vs-Cats in Rohpixel- und Wavelet-Darstellung.

Aufruf: pdm run python exercises/02_supervised_learning/exercise_2_8_ensembles.py
"""

import os
from urllib.request import urlopen
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"
os.environ["MPLCONFIGDIR"] = str(Path("results/.matplotlib").resolve())

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import load_iris, make_moons
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from scipy.io import loadmat

RESULTS = Path("results/exercise_2_8")
DOGS_CATS_DATA = RESULTS / "dogs_cats_data"
DOGS_CATS_URL = "https://github.com/dynamicslab/databook_python/raw/refs/heads/master/DATA"
SEED = 6020


def create_base_models():
    return [
        ("lda", LinearDiscriminantAnalysis()),
        ("svm", CalibratedClassifierCV(SVC(random_state=SEED), ensemble=False)),
        ("tree", DecisionTreeClassifier(random_state=SEED)),
    ]


def print_score(name, model, test_x, test_y):
    test_accuracy = model.score(test_x, test_y)
    print(f"{name}: Test-Accuracy = {test_accuracy:.3f}")


def print_voting_scores(voting, test_x, test_y):
    print_score("Voting gesamt", voting, test_x, test_y)

    for name, classifier in voting.named_estimators_.items():
        print_score(f"  {name}", classifier, test_x, test_y)


def run_voting_classifiers(train_x, test_x, train_y, test_y):
    base_models = create_base_models()

    hard_voting = VotingClassifier(base_models, voting="hard")
    hard_voting.fit(train_x, train_y)
    print("\nHard Voting")
    print_voting_scores(hard_voting, test_x, test_y)

    soft_voting = VotingClassifier(base_models, voting="soft")
    soft_voting.fit(train_x, train_y)
    print("\nSoft Voting")
    print_voting_scores(soft_voting, test_x, test_y)

    return base_models


def run_bagging_classifier(train_x, test_x, train_y, test_y):
    bagging = BaggingClassifier(
        DecisionTreeClassifier(random_state=SEED),
        n_estimators=500,
        max_samples=0.8,
        oob_score=True,
        random_state=SEED,
    )
    bagging.fit(train_x, train_y)

    print("\nBagging")
    print_score("bagging", bagging, test_x, test_y)
    print(f"  OOB-Score = {bagging.oob_score_:.3f}")


def run_random_forest_classifier(train_x, test_x, train_y, test_y):
    forest = RandomForestClassifier(
        n_estimators=500,
        max_leaf_nodes=16,
        random_state=SEED,
    )
    forest.fit(train_x, train_y)

    print("\nRandom Forest")
    print_score("random_forest", forest, test_x, test_y)


def run_stacking_classifier(base_models, train_x, test_x, train_y, test_y):
    final_estimator = RandomForestClassifier(n_estimators=200, random_state=SEED)
    stacking = StackingClassifier(base_models, final_estimator=final_estimator)
    stacking.fit(train_x, train_y)

    print("\nStacking")
    print_score("stacking", stacking, test_x, test_y)


def run_moons_ensembles():
    data, labels = make_moons(n_samples=500, noise=0.30, random_state=SEED)
    train_x, test_x, train_y, test_y = train_test_split(
        data,
        labels,
        random_state=SEED,
    )

    base_models = run_voting_classifiers(train_x, test_x, train_y, test_y)
    run_bagging_classifier(train_x, test_x, train_y, test_y)
    run_random_forest_classifier(train_x, test_x, train_y, test_y)
    run_stacking_classifier(base_models, train_x, test_x, train_y, test_y)


def plot_feature_importances(title, names, importances, filename):
    plt.figure(figsize=(7, 4))
    plt.bar(names, importances)
    plt.title(title)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(RESULTS / filename, dpi=150)
    plt.close()


def plot_iris_feature_importances():
    iris = load_iris()
    forest = RandomForestClassifier(n_estimators=500, random_state=SEED)
    forest.fit(iris.data, iris.target)

    print("\nIris Feature-Importances")
    print("Iris-Importances:", forest.feature_importances_.round(3))

    plot_feature_importances(
        "Iris: Feature-Importances",
        iris.feature_names,
        forest.feature_importances_,
        "iris_importance.png",
    )


def download_dogs_cats_dataset():
    DOGS_CATS_DATA.mkdir(parents=True, exist_ok=True)

    for animal in ("dog", "cat"):
        output = DOGS_CATS_DATA / f"{animal}Data.mat"

        if output.exists():
            continue

        url = f"{DOGS_CATS_URL}/{animal}Data.mat"
        print(f"Lade {url} herunter ...")

        with urlopen(url, timeout=60) as response:
            output.write_bytes(response.read())


def load_dogs_cats_images():
    download_dogs_cats_dataset()

    dog_images = loadmat(DOGS_CATS_DATA / "dogData.mat")["dog"]
    cat_images = loadmat(DOGS_CATS_DATA / "catData.mat")["cat"]

    dog_images = dog_images.T.reshape(-1, 64, 64) / 255.0
    cat_images = cat_images.T.reshape(-1, 64, 64) / 255.0

    images = np.vstack((dog_images, cat_images))
    labels = np.r_[np.zeros(len(dog_images)), np.ones(len(cat_images))]

    return images, labels


def haar_wavelet_features(images):
    features = []

    for image in images:
        low_columns = (image[:, 0::2] + image[:, 1::2]) / 2
        high_columns = (image[:, 0::2] - image[:, 1::2]) / 2

        approximation = (low_columns[0::2, :] + low_columns[1::2, :]) / 2
        horizontal = (low_columns[0::2, :] - low_columns[1::2, :]) / 2
        vertical = (high_columns[0::2, :] + high_columns[1::2, :]) / 2
        diagonal = (high_columns[0::2, :] - high_columns[1::2, :]) / 2

        image_features = np.concatenate(
            (
                approximation.ravel(),
                horizontal.ravel(),
                vertical.ravel(),
                diagonal.ravel(),
            )
        )
        features.append(image_features)

    return np.array(features)


def evaluate_dogs_cats_features(name, features, labels, filename):
    component_count = min(10, features.shape[0], features.shape[1])
    pca = PCA(n_components=component_count, random_state=SEED)
    pca_features = pca.fit_transform(features)

    train_x, test_x, train_y, test_y = train_test_split(
        pca_features,
        labels,
        random_state=SEED,
        stratify=labels,
    )

    forest = RandomForestClassifier(n_estimators=500, random_state=SEED)
    forest.fit(train_x, train_y)

    component_names = [f"PC {index + 1}" for index in range(component_count)]

    print_score(f"Dogs/Cats {name}", forest, test_x, test_y)
    print(f"Dogs/Cats {name} PCA-Importances:", forest.feature_importances_.round(3))

    plot_feature_importances(
        f"Dogs/Cats {name}: PCA-Feature-Importances",
        component_names,
        forest.feature_importances_,
        filename,
    )


def evaluate_dogs_cats_feature_importances():
    images, labels = load_dogs_cats_images()
    raw_features = images.reshape(len(images), -1)
    wavelet_features = haar_wavelet_features(images)

    print("\nDogs/Cats Feature-Importances")
    evaluate_dogs_cats_features("raw", raw_features, labels, "dogs_cats_raw_importance.png")
    evaluate_dogs_cats_features("wavelet", wavelet_features, labels, "dogs_cats_wavelet_importance.png")


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)

    run_moons_ensembles()
    plot_iris_feature_importances()
    evaluate_dogs_cats_feature_importances()


if __name__ == "__main__":
    main()
