"""Exercise 10.3: Padding-Varianten für EfficientNet-Eingaben.

Aufgabenstellung: Vergleiche symmetrisches schwarzes/weißes sowie asymmetrisches
Padding statt Resize, um die Zielgröße 224×224 zu erreichen.
"""

import torch
from torchvision.transforms import Pad


def main():
    image = torch.rand(3, 64, 64)
    variants = {"schwarz_symmetrisch": Pad(80, fill=0), "weiss_symmetrisch": Pad(80, fill=255), "asymmetrisch": Pad((0, 0, 160, 160), fill=0)}
    for name, transform in variants.items():
        padded = transform(image)
        assert padded.shape == (3, 224, 224)
        print(f"{name}: {tuple(padded.shape)}")


if __name__ == "__main__": main()
