"""Exercise 9.2: Explore the latent space and generate numbers.

Aufruf: pdm run python exercises/09_autoencoders/exercise_9_2_latent_space.py
"""

import os
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"
os.environ["MPLCONFIGDIR"] = str(Path("results/.matplotlib").resolve())

import matplotlib.pyplot as plt
import torch
from sklearn.datasets import load_digits

RESULTS = Path("results/exercise_9_2")
SEED = 6020


class Autoencoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 2),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(2, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 64),
            torch.nn.Sigmoid(),
        )

    def forward(self, data):
        return self.decoder(self.encoder(data))


def train(model, values, epochs=250):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for _ in range(epochs):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(values), values)
        loss.backward()
        optimizer.step()

    return loss.item()


def save_latent_space(latent, targets):
    plt.figure(figsize=(6, 5))
    plt.scatter(latent[:, 0], latent[:, 1], c=targets, cmap="tab10", s=8)
    plt.colorbar(label="digit")
    plt.tight_layout()
    plt.savefig(RESULTS / "latent_space.png", dpi=150)
    plt.close()


def save_generated_digits(model):
    points = torch.tensor(
        [
            [-2.0, -2.0],
            [-2.0, 0.0],
            [-2.0, 2.0],
            [0.0, -2.0],
            [0.0, 0.0],
            [0.0, 2.0],
            [2.0, -2.0],
            [2.0, 0.0],
            [2.0, 2.0],
        ]
    )

    with torch.no_grad():
        generated = model.decoder(points).numpy()

    fig, axes = plt.subplots(3, 3, figsize=(5, 5))
    for axis, image in zip(axes.ravel(), generated):
        axis.imshow(image.reshape(8, 8), cmap="gray")
        axis.axis("off")

    fig.tight_layout()
    fig.savefig(RESULTS / "generated_digits.png", dpi=150)
    plt.close(fig)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)

    digits = load_digits()
    values = torch.tensor(digits.data / 16, dtype=torch.float32)

    model = Autoencoder()
    loss = train(model, values)

    with torch.no_grad():
        latent = model.encoder(values).numpy()

    save_latent_space(latent, digits.target)
    save_generated_digits(model)

    print(f"Reconstruction loss: {loss:.4f}")
    print("Plots: latent_space.png, generated_digits.png")


if __name__ == "__main__":
    main()
