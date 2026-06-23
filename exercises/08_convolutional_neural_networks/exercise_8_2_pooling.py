"""Exercise 8.2: Max- und Mean-Pooling auf einem Bild.

Aufgabenstellung: Implementiere Max- und Mean-Pooling (oder nutze eine passende
Bibliotheksfunktion), wende beides auf ein Bild an und erkläre erhaltene bzw.
verlorene Merkmale.

Max-Pooling behält starke Kanten/Aktivierungen; Mean-Pooling bewahrt lokale
Helligkeit, glättet aber Kanten stärker. Beide reduzieren Ortsauflösung.
"""

import base64
import os
import re
from io import BytesIO
from pathlib import Path
os.environ["MPLBACKEND"] = "Agg"; os.environ["MPLCONFIGDIR"] = str(Path("results/.matplotlib").resolve())

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image, ImageOps

RESULTS = Path("results/exercise_8_2")


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)

    svg = Path(__file__).with_name("cat.svg").read_text()
    encoded = re.search(r"data:image/png;base64,\s*([^\"]+)", svg).group(1)

    cat = ImageOps.flip(Image.open(BytesIO(base64.b64decode(encoded))).convert("L"))
    cat_zero = np.asarray(cat.resize((64, 64), Image.Resampling.NEAREST))
    image = torch.tensor(cat_zero, dtype=torch.float32)[None, None]

    maximum,  = functional.max_pool2d(image, 2)
    average = functional.avg_pool2d(image, 2)
    
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    for axis, data, title in zip(axes, (image, maximum, average), ("Original", "Max-Pooling", "Mean-Pooling")):
        axis.imshow(data.squeeze(), cmap="gray"); axis.set_title(title); axis.axis("off")
    fig.tight_layout(); fig.savefig(RESULTS / "pooling.png", dpi=150); plt.close(fig)
    print(f"cat0: {tuple(image.shape)}, Max/Mean: {tuple(maximum.shape)}")


if __name__ == "__main__": main()
