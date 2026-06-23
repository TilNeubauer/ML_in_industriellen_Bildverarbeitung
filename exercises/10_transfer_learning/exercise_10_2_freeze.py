"""Exercise 10.2: Vortrainiertes Netzwerk einfrieren.

Aufgabenstellung: Friere die ursprünglichen EfficientNet-Schichten ein und
trainiere nur den finalen Zwei-Klassen-Klassifikator.
"""

import torch
from torchvision.models import efficientnet_b0


def main():
    model = efficientnet_b0(weights=None)
    for parameter in model.features.parameters(): parameter.requires_grad = False
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    frozen = sum(parameter.numel() for parameter in model.parameters() if not parameter.requires_grad)
    assert trainable > 0 and frozen > 0
    print(f"Trainierbar: {trainable}; eingefroren: {frozen}")


if __name__ == "__main__": main()
