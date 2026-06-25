"""Exercise 9.1: Add more transformations.

Aufruf: pdm run python exercises/09_autoencoders/exercise_9_1_transformations.py
"""

import os
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"
os.environ["MPLCONFIGDIR"] = str(Path("results/.matplotlib").resolve())

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import rotate
from sklearn.datasets import load_digits

RESULTS = Path("results/exercise_9_1")
SEED = 6020


class Autoencoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(64, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 2),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(2, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 64),
            torch.nn.Sigmoid(),
        )

    def forward(self, data):
        return self.decoder(self.encoder(data))


def rotate_with_padding(image, angle=20, padding=2):
    padded = np.pad(image, padding, mode="constant")
    rotated = rotate(padded, angle, reshape=False, mode="constant")
    return rotated[padding:-padding, padding:-padding]


def transform(images):
    rng = np.random.default_rng(SEED)
    permutation = rng.permutation(64)

    shuffled = images.reshape(-1, 64)[:, permutation].reshape(-1, 8, 8)
    rotated = np.array([rotate_with_padding(image) for image in images])
    flipped = images[:, :, ::-1]

    return {
        "original": images,
        "shuffle": shuffled,
        "rotation": rotated,
        "flip": flipped,
    }


def train(images, epochs=120):
    torch.manual_seed(SEED)
    model = Autoencoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    values = torch.tensor(images.reshape(-1, 64), dtype=torch.float32)

    for _ in range(epochs):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(values), values)
        loss.backward()
        optimizer.step()

    return loss.item()


def save_examples(transformed):
    fig, axes = plt.subplots(1, len(transformed), figsize=(10, 3))

    for axis, (name, images) in zip(axes, transformed.items()):
        axis.imshow(images[0], cmap="gray")
        axis.set_title(name)
        axis.axis("off")

    fig.tight_layout()
    fig.savefig(RESULTS / "transformations.png", dpi=150)
    plt.close(fig)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    images = load_digits().images / 16
    transformed = transform(images)

    losses = {name: train(data) for name, data in transformed.items()}
    all_images = np.vstack(list(transformed.values()))
    losses["all"] = train(all_images)

    for name, loss in losses.items():
        print(f"{name}: reconstruction_loss={loss:.4f}")

    save_examples(transformed)


if __name__ == "__main__":
    main()
