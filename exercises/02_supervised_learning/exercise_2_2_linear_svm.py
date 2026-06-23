"""Exercise 2.2: Normale SVM-Vektoren und Klassifikation zweier Punkte.

Aufgabenstellung:
1. Berechne den zur Trennlinie normalen Vektor w für die zwei vorgegebenen Linien.
2. Klassifiziere a = [1.4, 5.1]^T und b = [4.7, 7.0]^T.

Aufruf: pdm run python exercises/02_supervised_learning/exercise_2_2_linear_svm.py
"""

import numpy as np


def normal_vector(first, second):
    """Gibt einen Normalenvektor der durch zwei Punkte bestimmten Geraden zurück."""
    direction = np.asarray(second) - np.asarray(first)
    return np.array([-direction[1], direction[0]]) / np.linalg.norm(direction)


def classify(point, line_point, normal):
    """Klassifiziert nach der Seite der Trennlinie (+1 oder -1)."""
    return int(np.sign(np.dot(np.asarray(point) - line_point, normal)))


def main():
    lines = {
        "Fall 1": (np.array([1.25, 4.1]), np.array([5.0, 7.4])),
        "Fall 2": (np.array([2.6, 4.25]), np.array([2.0, 7.0])),
    }
    for name, (first, second) in lines.items():
        normal = normal_vector(first, second)
        print(f"{name}: w = {normal.round(3)}")
        for point_name, point in {"a": [1.4, 5.1], "b": [4.7, 7.0]}.items():
            print(f"  Punkt {point_name}: Klasse {classify(point, first, normal):+d}")


if __name__ == "__main__":
    main()
