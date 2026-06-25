"""Exercise 6.2: 

Optional: If we rerun our training, we can see that the results change slightly. 
Find out what has changed.

Note: This is not optimal, but the upside is we control the ETL so we can actually 
make sure that a new image is processed in the same fashion and we do not need to ask 
the authors of Brunton and Kutz (2022) to help us out.


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
