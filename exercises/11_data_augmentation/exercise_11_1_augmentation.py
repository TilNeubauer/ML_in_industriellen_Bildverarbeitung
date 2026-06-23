"""Exercise 11.1: Datenaugmentation für Bildklassifikation.

Aufgabenstellung: Erstelle eine Pipeline mit Datenaugmentierungen und trainiere
die bereits erstellten Modelle erneut. Untersuche, ob sich Generalisierung bzw.
Test-Accuracy verbessern.

Aufruf: pdm run python exercises/11_data_augmentation/exercise_11_1_augmentation.py
"""

import os
from pathlib import Path
os.environ["MPLBACKEND"] = "Agg"; os.environ["MPLCONFIGDIR"] = str(Path("results/.matplotlib").resolve())

import matplotlib.pyplot as plt
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torchvision.transforms import RandomAffine

RESULTS = Path("results/exercise_11_1"); SEED = 6020


class DigitCNN(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.layers = torch.nn.Sequential(torch.nn.Conv2d(1, 8, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2), torch.nn.Flatten(), torch.nn.Linear(8 * 4 * 4, 10))
    def forward(self, data): return self.layers(data)


def train(train_x, train_y, test_x, test_y, augmentation=None):
    torch.manual_seed(SEED); model = DigitCNN(); optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    for _ in range(80):
        data = augmentation(train_x) if augmentation else train_x
        optimizer.zero_grad(); torch.nn.functional.cross_entropy(model(data), train_y).backward(); optimizer.step()
    return (model(test_x).argmax(1) == test_y).float().mean().item()


def main():
    RESULTS.mkdir(parents=True, exist_ok=True); digits = load_digits()
    train_x, test_x, train_y, test_y = train_test_split(digits.images / 16, digits.target, stratify=digits.target, test_size=.25, random_state=SEED)
    train_x, test_x = torch.tensor(train_x[:, None], dtype=torch.float32), torch.tensor(test_x[:, None], dtype=torch.float32)
    train_y, test_y = torch.tensor(train_y), torch.tensor(test_y)
    augmentation = RandomAffine(degrees=12, translate=(.1, .1))
    baseline, augmented = train(train_x, train_y, test_x, test_y), train(train_x, train_y, test_x, test_y, augmentation)
    print(f"Ohne Augmentation: {baseline:.3f}; mit Augmentation: {augmented:.3f}")
    fig, axes = plt.subplots(1, 2, figsize=(4, 2)); axes[0].imshow(train_x[0, 0], cmap="gray"); axes[0].set_title("Original"); axes[1].imshow(augmentation(train_x[:1])[0, 0], cmap="gray"); axes[1].set_title("Augmentiert")
    for axis in axes: axis.axis("off")
    fig.tight_layout(); fig.savefig(RESULTS / "augmentation.png", dpi=150); plt.close(fig)


if __name__ == "__main__": main()
