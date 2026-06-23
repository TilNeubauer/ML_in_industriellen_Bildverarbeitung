"""Exercise 9.2: Latenten Raum untersuchen und Ziffern generieren.

Aufgabenstellung:
1. Mache Encoder und Decoder getrennt zugreifbar und visualisiere die Cluster.
2. Erzeuge Ziffern durch selbst gewählte Punkte im latenten Raum.

Aufruf: pdm run python exercises/09_autoencoders/exercise_9_2_latent_space.py
"""

import os
from pathlib import Path
os.environ["MPLBACKEND"] = "Agg"; os.environ["MPLCONFIGDIR"] = str(Path("results/.matplotlib").resolve())

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import load_digits

RESULTS = Path("results/exercise_9_2"); SEED = 6020


class Autoencoder(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.encoder = torch.nn.Sequential(torch.nn.Linear(64, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2)); self.decoder = torch.nn.Sequential(torch.nn.Linear(2, 32), torch.nn.ReLU(), torch.nn.Linear(32, 64), torch.nn.Sigmoid())
    def forward(self, data): return self.decoder(self.encoder(data))


def main():
    RESULTS.mkdir(parents=True, exist_ok=True); torch.manual_seed(SEED); digits = load_digits(); values = torch.tensor(digits.data / 16, dtype=torch.float32)
    model = Autoencoder(); optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    for _ in range(200): optimizer.zero_grad(); loss = torch.nn.functional.mse_loss(model(values), values); loss.backward(); optimizer.step()
    with torch.no_grad(): latent = model.encoder(values).numpy(); generated = model.decoder(torch.tensor([[-2., -2.], [-2., 2.], [2., -2.], [2., 2.]])).numpy()
    plt.figure(figsize=(6, 5)); plt.scatter(latent[:, 0], latent[:, 1], c=digits.target, cmap="tab10", s=8); plt.colorbar(label="Ziffer"); plt.tight_layout(); plt.savefig(RESULTS / "latent_space.png", dpi=150); plt.close()
    fig, axes = plt.subplots(1, 4, figsize=(8, 2));
    for axis, image in zip(axes, generated): axis.imshow(image.reshape(8, 8), cmap="gray"); axis.axis("off")
    fig.tight_layout(); fig.savefig(RESULTS / "generated_digits.png", dpi=150); plt.close(fig)
    print(f"Rekonstruktionsverlust: {loss.item():.4f}")


if __name__ == "__main__": main()
