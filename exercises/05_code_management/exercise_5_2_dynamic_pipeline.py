"""Exercise 5.2: Externalize params.

Use this to make the model creation more and more dynamic.
Replace the array estimators=[] by dynamically loading the different estimators.
Use the same for the pipeline.

Note: If you see an advantage in rewriting the config structure to make your code
easier, feel free to do so.

Aufruf: pdm run python exercises/05_code_management/exercise_5_2_dynamic_pipeline.py
"""

from importlib import import_module
from pathlib import Path

import yaml
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

SEED = 6020


def create_object(specification):
    module_name, class_name = specification["type"].rsplit(".", 1)
    class_ = getattr(import_module(module_name), class_name)

    init_args = dict(specification.get("init_args", {}))
    for nested_key in ("steps", "estimators"):
        if nested_key in specification:
            init_args[nested_key] = [
                (item["name"], create_object(item))
                for item in specification[nested_key]
            ]

    return class_(**init_args)


def main():
    params = yaml.safe_load(Path(__file__).with_name("params.yaml").read_text())
    pipeline = create_object(params["pipeline"])

    iris = load_iris()
    train_x, test_x, train_y, test_y = train_test_split(
        iris.data,
        iris.target,
        random_state=SEED,
        stratify=iris.target,
    )

    pipeline.fit(train_x, train_y)
    score = pipeline.score(test_x, test_y)
    print(f"Dynamische Pipeline: Test-Accuracy = {score:.3f}")


if __name__ == "__main__":
    main()
