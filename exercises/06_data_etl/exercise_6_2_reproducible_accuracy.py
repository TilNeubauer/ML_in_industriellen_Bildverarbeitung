"""Exercise 6.2: Optionale Untersuchung veränderter Accuracy.

Aufgabenstellung: Ermittle, warum sich Accuracy nach erneutem Training ändert.
Die Lösung demonstriert: zufällige Splits/Initialisierungen ändern Ergebnisse;
feste random_state-Werte machen den gesamten Ablauf reproduzierbar.
"""

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def score(seed=None):
    iris = load_iris()
    train_x, test_x, train_y, test_y = train_test_split(iris.data, iris.target, test_size=.25, stratify=iris.target, random_state=seed)
    return RandomForestClassifier(n_estimators=100, random_state=seed).fit(train_x, train_y).score(test_x, test_y)


def main():
    fixed = (score(6020), score(6020))
    varying = (score(), score())
    print(f"Fester Seed: {fixed}; ohne Seed: {varying}")
    assert fixed[0] == fixed[1]


if __name__ == "__main__": main()
