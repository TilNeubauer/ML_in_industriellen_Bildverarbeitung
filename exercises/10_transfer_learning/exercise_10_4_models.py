"""Exercise 10.4: Anderes EfficientNet-Modell verwenden.

Aufgabenstellung: Teste mindestens ein weiteres EfficientNet-Modell und prüfe,
ob eine vergleichbare Zwei-Klassen-Architektur möglich ist.
"""

import torch
from torchvision.models import efficientnet_b0, efficientnet_v2_s


def classifier(model):
    model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, 2)
    return model


def main():
    for name, model in (("EfficientNet-B0", classifier(efficientnet_b0(weights=None))), ("EfficientNet-V2-S", classifier(efficientnet_v2_s(weights=None)))):
        assert model(torch.rand(1, 3, 224, 224)).shape == (1, 2)
        print(f"{name}: Zwei-Klassen-Ausgabe erfolgreich")


if __name__ == "__main__": main()
