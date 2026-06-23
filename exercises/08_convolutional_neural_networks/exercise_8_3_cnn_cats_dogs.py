"""Exercise 8.3: CNN für originale Cats-vs-Dogs-Bilder.

Aufgabenstellung: Implementiere das CNN mit den originalen 64×64-Pixel-Bildern
statt der Wavelet-Darstellung. Teste eine Architektur mit mehreren Convolution-
Stufen und bewerte die Test-Accuracy.

Aufruf: pdm run python exercises/08_convolutional_neural_networks/exercise_8_3_cnn_cats_dogs.py
"""

import io

import numpy as np
import requests
import torch
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

SEED = 6020
BASE_URL = "https://github.com/dynamicslab/databook_python/raw/refs/heads/master/DATA/"


def load_images(animal):
    """Lädt die 80 64×64-Rohbilder einer Klasse aus den Kursdaten."""
    response = requests.get(f"{BASE_URL}{animal}Data.mat", timeout=60)
    response.raise_for_status()
    matrix = loadmat(io.BytesIO(response.content))[animal]
    return matrix.T.reshape(-1, 1, 64, 64) / 255.0


class CatsDogsCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, kernel_size=3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(8, 16, kernel_size=3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
        )
        self.classifier = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(16 * 16 * 16, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2))

    def forward(self, images):
        return self.classifier(self.features(images))


def main():
    torch.manual_seed(SEED)
    images = np.vstack((load_images("dog"), load_images("cat"))).astype("float32")
    labels = np.r_[np.zeros(80, dtype="int64"), np.ones(80, dtype="int64")]
    train_x, test_x, train_y, test_y = train_test_split(images, labels, test_size=.25, stratify=labels, random_state=SEED)
    train_x, test_x = torch.tensor(train_x), torch.tensor(test_x)
    train_y, test_y = torch.tensor(train_y), torch.tensor(test_y)
    model = CatsDogsCNN(); optimizer = torch.optim.Adam(model.parameters(), lr=.001); loss_function = torch.nn.CrossEntropyLoss()
    for _ in range(60):
        optimizer.zero_grad(); loss_function(model(train_x), train_y).backward(); optimizer.step()
    accuracy = (model(test_x).argmax(1) == test_y).float().mean().item()
    print(f"CNN-Test-Accuracy auf originalen 64×64-Bildern: {accuracy:.3f}")


if __name__ == "__main__": main()
