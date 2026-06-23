"""Exercise 9.1: Zusätzliche Transformationen für den Autoencoder.

Aufgabenstellung:
1. Shuffle Pixel mit einer festen Permutation.
2. Rotiere, erweitere und beschneide Bilder wieder.
3. Spiegele Bilder.
4. Trainiere den Autoencoder mit diesen Datensätzen und bewerte das Ergebnis.

Aufruf: pdm run python exercises/09_autoencoders/exercise_9_1_transformations.py
"""

import os
from pathlib import Path
os.environ["MPLBACKEND"] = "Agg"; os.environ["MPLCONFIGDIR"] = str(Path("results/.matplotlib").resolve())

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import rotate
from sklearn.datasets import load_digits

RESULTS = Path("results/exercise_9_1"); SEED = 6020


class Autoencoder(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.encoder = torch.nn.Sequential(torch.nn.Linear(64, 16), torch.nn.ReLU(), torch.nn.Linear(16, 2)); self.decoder = torch.nn.Sequential(torch.nn.Linear(2, 16), torch.nn.ReLU(), torch.nn.Linear(16, 64), torch.nn.Sigmoid())
    def forward(self, data): return self.decoder(self.encoder(data))


def transform(images):
    rng = np.random.default_rng(SEED); shuffled = images.reshape(-1, 64)[:, rng.permutation(64)].reshape(-1, 8, 8)
    rotated = np.array([rotate(image, 20, reshape=False, mode="constant") for image in images])
    return {"Original": images, "Shuffle": shuffled, "Rotation": rotated, "Spiegelung": images[:, :, ::-1]}


def train(data):
    torch.manual_seed(SEED); model = Autoencoder(); optimizer = torch.optim.Adam(model.parameters(), lr=.01); values = torch.tensor(data.reshape(-1, 64), dtype=torch.float32)
    for _ in range(100): optimizer.zero_grad(); loss = torch.nn.functional.mse_loss(model(values), values); loss.backward(); optimizer.step()
    return loss.item()


def main():
    RESULTS.mkdir(parents=True, exist_ok=True); images = load_digits().images / 16
    transformed = transform(images); print({name: round(train(data), 4) for name, data in transformed.items()})
    fig, axes = plt.subplots(1, 4, figsize=(10, 3))
    for axis, (name, data) in zip(axes, transformed.items()): axis.imshow(data[0], cmap="gray"); axis.set_title(name); axis.axis("off")
    fig.tight_layout(); fig.savefig(RESULTS / "transformations.png", dpi=150); plt.close(fig)


if __name__ == "__main__": main()
