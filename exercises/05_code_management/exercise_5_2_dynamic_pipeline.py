"""Exercise 5.2: Pipeline und Estimators dynamisch aus params.yaml erzeugen.

Aufgabenstellung:
1. Lade die Estimators statt eines festen estimators=[] dynamisch.
2. Erzeuge auch die Pipeline dynamisch mittels import_module.
"""

from importlib import import_module
from pathlib import Path

import yaml
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def create_object(specification):
    """Erzeugt eine Klasse aus vollständigem Importpfad und init_args."""
    module_name, class_name = specification["type"].rsplit(".", 1)
    return getattr(import_module(module_name), class_name)(**specification.get("init_args", {}))


def main():
    params = yaml.safe_load(Path(__file__).with_name("params.yaml").read_text())
    steps = [(step["name"], create_object(step)) for step in params["pipeline"]["steps"]]
    pipeline = create_object({**params["pipeline"], "init_args": {"steps": steps}})
    iris = load_iris(); train_x, test_x, train_y, test_y = train_test_split(iris.data, iris.target, random_state=6020, stratify=iris.target)
    pipeline.fit(train_x, train_y)
    print(f"Dynamische Pipeline: Test-Accuracy = {pipeline.score(test_x, test_y):.3f}")


if __name__ == "__main__": main()
