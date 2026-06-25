"""Exercise 8.2: Applying pooling to images.


Max-Pooling bewahrt starke lokale Aktivierungen, zum Beispiel helle Punkte,
Kanten und markante Strukturen. Schwächere Details innerhalb eines Pooling-
Fensters gehen dabei verloren.

Mean-Pooling bewahrt die durchschnittliche lokale Helligkeit und glättet das
Bild. Einzelne starke Kanten oder kleine Details werden dadurch abgeschwächt.

Beide Pooling-Arten reduzieren die räumliche Auflösung. Die genaue Position
kleiner Details geht also teilweise verloren.

Aufruf: pdm run python exercises/08_convolutional_neural_networks/exercise_8_2_pooling.py
"""

import os
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"
os.environ["MPLCONFIGDIR"] = str(Path("results/.matplotlib").resolve())

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image

RESULTS = Path("results/exercise_8_2")


def load_cat():
    cat = Image.open(Path(__file__).with_name("cat.png")).convert("L")
    return torch.tensor(np.asarray(cat), dtype=torch.float32)[None, None] / 255


def save_plot(original, maximum, average):
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    images = (original, maximum, average)
    titles = ("Original", "Max-Pooling", "Mean-Pooling")

    for axis, image, title in zip(axes, images, titles):
        axis.imshow(image.squeeze(), cmap="gray")
        axis.set_title(title)
        axis.axis("off")

    fig.tight_layout()
    fig.savefig(RESULTS / "pooling.png", dpi=150)
    plt.close(fig)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)

    image = load_cat()
    maximum = functional.max_pool2d(image, kernel_size=2)
    average = functional.avg_pool2d(image, kernel_size=2)

    save_plot(image, maximum, average)

    print(f"Original: {tuple(image.shape)}")
    print(f"Pooling:  {tuple(maximum.shape)}")
    

if __name__ == "__main__":
    main()
