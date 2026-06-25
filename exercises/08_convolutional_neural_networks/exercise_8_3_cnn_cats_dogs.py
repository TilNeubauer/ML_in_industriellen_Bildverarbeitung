"""Exercise 8.3: CNN for original cats and dogs images.


Aufruf: pdm run python exercises/08_convolutional_neural_networks/exercise_8_3_cnn_cats_dogs.py
"""

from pathlib import Path
from urllib.request import urlopen

import numpy as np
import torch
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

SEED = 6020
DATA_DIR = Path("results/exercise_2_8/dogs_cats_data")
BASE_URL = "https://github.com/dynamicslab/databook_python/raw/refs/heads/master/DATA"


def ensure_mat_file(animal):
    path = DATA_DIR / f"{animal}Data.mat"
    if path.exists():
        return path

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with urlopen(f"{BASE_URL}/{path.name}", timeout=60) as response:
        path.write_bytes(response.read())
    return path


def load_images(animal):
    matrix = loadmat(ensure_mat_file(animal))[animal]
    return matrix.T.reshape(-1, 1, 64, 64) / 255.0


def load_data():
    images = np.vstack((load_images("dog"), load_images("cat"))).astype("float32")
    labels = np.r_[np.zeros(80, dtype="int64"), np.ones(80, dtype="int64")]

    train_x, test_x, train_y, test_y = train_test_split(
        images,
        labels,
        test_size=0.25,
        stratify=labels,
        random_state=SEED,
    )
    return (
        torch.tensor(train_x),
        torch.tensor(test_x),
        torch.tensor(train_y),
        torch.tensor(test_y),
    )


class CatsDogsCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(8, 16, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(32 * 8 * 8, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 2),
        )

    def forward(self, images):
        return self.classifier(self.features(images))


def train(model, train_x, train_y, epochs=80):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = torch.nn.CrossEntropyLoss()

    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_function(model(train_x), train_y)
        loss.backward()
        optimizer.step()


def accuracy(model, test_x, test_y):
    predictions = model(test_x).argmax(dim=1)
    return (predictions == test_y).float().mean().item()


def main():
    torch.manual_seed(SEED)
    train_x, test_x, train_y, test_y = load_data()

    model = CatsDogsCNN()
    train(model, train_x, train_y)

    score = accuracy(model, test_x, test_y)
    print(f"CNN-Test-Accuracy auf originalen 64x64-Bildern: {score:.3f}")


if __name__ == "__main__":
    main()
