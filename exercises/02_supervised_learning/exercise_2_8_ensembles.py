"""Exercise 2.8: Ensemble-Methoden.

Aufgabenstellung: Vergleiche auf Moons Hard-/Soft-Voting (LDA, SVM, Baum),
Bagging, Random Forest und Stacking. Bestimme außerdem Iris-Feature-Importances.

Aufruf: pdm run python exercises/02_supervised_learning/exercise_2_8_ensembles.py
"""

import os
from pathlib import Path
os.environ["MPLBACKEND"] = "Agg"; os.environ["MPLCONFIGDIR"] = str(Path("results/.matplotlib").resolve())
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, make_moons
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RESULTS = Path("results/exercise_2_8"); SEED = 6020


def main():
    RESULTS.mkdir(parents=True, exist_ok=True); data, labels = make_moons(n_samples=500, noise=.30, random_state=SEED)
    train_x, test_x, train_y, test_y = train_test_split(data, labels, random_state=SEED, stratify=labels)
    base = [("lda", LinearDiscriminantAnalysis()), ("svm", SVC(probability=True, random_state=SEED)), ("tree", DecisionTreeClassifier(random_state=SEED))]
    models = {
        "hard_voting": VotingClassifier(base, voting="hard"), "soft_voting": VotingClassifier(base, voting="soft"),
        "bagging": BaggingClassifier(DecisionTreeClassifier(), n_estimators=500, max_samples=.8, oob_score=True, random_state=SEED),
        "random_forest": RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, random_state=SEED),
        "stacking": StackingClassifier(base, final_estimator=RandomForestClassifier(n_estimators=200, random_state=SEED)),
    }
    for name, model in models.items():
        model.fit(train_x, train_y); extra = f", OOB={model.oob_score_:.3f}" if name == "bagging" else ""
        print(f"{name}: Test-Accuracy = {model.score(test_x, test_y):.3f}{extra}")
    iris = load_iris(); forest = RandomForestClassifier(n_estimators=500, random_state=SEED).fit(iris.data, iris.target)
    plt.figure(figsize=(7, 4)); plt.bar(iris.feature_names, forest.feature_importances_); plt.xticks(rotation=20, ha="right"); plt.tight_layout(); plt.savefig(RESULTS / "iris_importance.png", dpi=150); plt.close()
    print("Iris-Importances:", forest.feature_importances_.round(3))


if __name__ == "__main__": main()
