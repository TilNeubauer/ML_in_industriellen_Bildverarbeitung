"""Exercise 3.1: Semi-supervised Klassifikation verbessern.

Aufgabenstellung:
1. Optimiere Random-Forest- und Clustering-Parameter.
2. Kombiniere Klassifikatoren und label zusätzliche, besonders unsichere Beispiele.
3. Nutze Cluster, um weitere Labels gezielt zu übertragen.

Hier dient sklearns Zifferndatensatz als schnell lokal testbare MNIST-Alternative.
Aufruf: pdm run python exercises/03_semi_supervised_learning/exercise_3_1_improve_semisupervised.py
"""

from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 6020


def representatives(data, labels, centers):
    """Gibt pro Cluster den Index des der Mitte nächsten Bildes zurück."""
    return np.array([np.where(labels == cluster)[0][np.argmin(np.linalg.norm(data[labels == cluster] - center, axis=1))]
                     for cluster, center in enumerate(centers)])


def main():
    digits = load_digits()
    train_x, test_x, train_y, test_y = train_test_split(
        StandardScaler().fit_transform(digits.data), digits.target, stratify=digits.target, test_size=.25, random_state=SEED
    )
    kmeans = KMeans(n_clusters=100, n_init=20, random_state=SEED).fit(train_x)
    labeled = representatives(train_x, kmeans.labels_, kmeans.cluster_centers_)
    forest = RandomForestClassifier(n_estimators=500, max_leaf_nodes=40, random_state=SEED)
    forest.fit(train_x[labeled], train_y[labeled])
    print(f"Nur {len(labeled)} Cluster-Repräsentanten: {forest.score(test_x, test_y):.3f}")

    uncertain = np.argsort(forest.predict_proba(train_x).max(axis=1))[:100]
    labeled = np.unique(np.r_[labeled, uncertain])
    ensemble = VotingClassifier([("forest", forest), ("logreg", LogisticRegression(max_iter=2_000, random_state=SEED))], voting="soft")
    ensemble.fit(train_x[labeled], train_y[labeled])
    print(f"Mit 100 unsicheren Nachlabels: {accuracy_score(test_y, ensemble.predict(test_x)):.3f}")


if __name__ == "__main__":
    main()
