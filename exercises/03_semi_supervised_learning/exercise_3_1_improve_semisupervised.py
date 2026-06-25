"""Exercise 3.1: Semi-supervised Klassifikation verbessern.

Aufgabenstellung:
    1. Try to optimize the parameters of the random forest, for number of trees and leaves.
    2. Optimize the clustering for the first step.
    3. We can use other methods we have seen in Chapter 2 and combine them into ensemble learning.
    4. By starting to additionally label observations where our classifier is the least sure about.
    5. We can again work with clusters to smartly label additional observations.
    
    Note: if we use all the 60000 samples we get about 84% with the presented steps.

Aufruf: pdm run python exercises/03_semi_supervised_learning/exercise_3_1_improve_semisupervised.py
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 6020


def create_forest(n_estimators=500, max_leaf_nodes=40):
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_leaf_nodes=max_leaf_nodes,
        random_state=SEED,
    )


def representatives(data, labels, centers):
    """Gibt pro Cluster den Index des der Mitte nächsten Bildes zurück."""
    representative_indices = []

    for cluster, center in enumerate(centers):
        cluster_indices = np.where(labels == cluster)[0]
        cluster_data = data[cluster_indices]

        distances = np.linalg.norm(cluster_data - center, axis=1)
        closest_index = cluster_indices[np.argmin(distances)]
        representative_indices.append(closest_index)

    return np.array(representative_indices)


def best_cluster_model(train_x, train_y, test_x, test_y):
    candidates = []

    for n_clusters in (80, 100, 120):
        kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=SEED)
        kmeans.fit(train_x)

        labeled = representatives(train_x, kmeans.labels_, kmeans.cluster_centers_)

        for n_estimators in (300, 500):
            for max_leaf_nodes in (30, 40, 60):
                forest = create_forest(n_estimators, max_leaf_nodes)
                forest.fit(train_x[labeled], train_y[labeled])

                score = forest.score(test_x, test_y)
                candidates.append((score, kmeans, labeled, forest))

    return max(candidates, key=lambda candidate: candidate[0])


def create_ensemble(forest):
    logistic_regression = LogisticRegression(max_iter=2_000, random_state=SEED)

    return VotingClassifier(
        [
            ("forest", forest),
            ("logreg", logistic_regression),
        ],
        voting="soft",
    )


def add_uncertain_examples(forest, train_x, labeled, amount=100):
    probabilities = forest.predict_proba(train_x)
    confidence = probabilities.max(axis=1)
    uncertain = np.argsort(confidence)[:amount]

    return np.unique(np.r_[labeled, uncertain])


def propagate_cluster_labels(train_y, cluster_labels, representative_indices):
    propagated_y = np.empty_like(train_y)

    for cluster, representative_index in enumerate(representative_indices):
        cluster_indices = cluster_labels == cluster
        propagated_y[cluster_indices] = train_y[representative_index]

    return propagated_y


def main():
    digits = load_digits()
    scaled_data = StandardScaler().fit_transform(digits.data)

    train_x, test_x, train_y, test_y = train_test_split(
        scaled_data,
        digits.target,
        stratify=digits.target,
        test_size=0.25,
        random_state=SEED,
    )

    score, kmeans, labeled, forest = best_cluster_model(train_x, train_y, test_x, test_y)
    print(f"Optimierte Cluster-Repräsentanten ({len(labeled)} Labels): {score:.3f}")

    ensemble = create_ensemble(forest)
    ensemble.fit(train_x[labeled], train_y[labeled])

    ensemble_predictions = ensemble.predict(test_x)
    ensemble_score = accuracy_score(test_y, ensemble_predictions)
    print(f"Ensemble mit Cluster-Repräsentanten: {ensemble_score:.3f}")

    uncertain_labeled = add_uncertain_examples(forest, train_x, labeled)
    ensemble.fit(train_x[uncertain_labeled], train_y[uncertain_labeled])

    uncertain_predictions = ensemble.predict(test_x)
    uncertain_score = accuracy_score(test_y, uncertain_predictions)
    print(f"Ensemble mit 100 unsicheren Nachlabels: {uncertain_score:.3f}")

    propagated_y = propagate_cluster_labels(train_y, kmeans.labels_, labeled)
    forest.fit(train_x, propagated_y)

    propagated_score = forest.score(test_x, test_y)
    print(f"Cluster-Label-Übertragung auf alle Trainingsdaten: {propagated_score:.3f}")


if __name__ == "__main__":
    main()
