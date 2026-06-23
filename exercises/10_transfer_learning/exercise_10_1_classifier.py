"""Exercise 10.1: EfficientNet um einen Zwei-Klassen-Kopf ergänzen.

Aufgabenstellung: Ergänze statt des finalen Klassifikators eine zusätzliche
lineare Schicht mit zwei Ausgängen für Cats-vs-Dogs.
"""

import torch
from torchvision.models import efficientnet_b0


def main():
    backbone = efficientnet_b0(weights=None)
    features = backbone.classifier[1].in_features
    backbone.classifier[1] = torch.nn.Linear(features, 2)
    assert backbone(torch.rand(2, 3, 224, 224)).shape == (2, 2)
    print(f"EfficientNet-B0-Klassifikator: {features} -> 2")


if __name__ == "__main__": main()
