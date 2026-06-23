"""Exercise 7.3: Lernrate, Momentum und Optimizer vergleichen.

Aufgabenstellung: Teste SGD für Lernraten 10^-1 bis 10^-4 und verschiedene
Momentumwerte. Vergleiche Adam und Adagrad über die finale Accuracy.
"""

import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

SEED = 6020


def train(optimizer_factory):
    torch.manual_seed(SEED); digits = load_digits()
    x_train, x_test, y_train, y_test = train_test_split(digits.data / 16, digits.target, test_size=.25, stratify=digits.target, random_state=SEED)
    train_x, test_x = torch.tensor(x_train, dtype=torch.float32), torch.tensor(x_test, dtype=torch.float32)
    train_y, test_y = torch.tensor(y_train), torch.tensor(y_test)
    model = torch.nn.Sequential(torch.nn.Linear(64, 32), torch.nn.ReLU(), torch.nn.Linear(32, 10))
    optimizer = optimizer_factory(model.parameters()); loss = torch.nn.CrossEntropyLoss()
    for _ in range(100):
        optimizer.zero_grad(); loss(model(train_x), train_y).backward(); optimizer.step()
    return (model(test_x).argmax(1) == test_y).float().mean().item()


def main():
    settings = {"SGD lr=0.1, momentum=0.9": lambda p: torch.optim.SGD(p, lr=.1, momentum=.9), "SGD lr=0.001": lambda p: torch.optim.SGD(p, lr=.001), "Adam": lambda p: torch.optim.Adam(p, lr=.01), "Adagrad": lambda p: torch.optim.Adagrad(p, lr=.1)}
    for name, factory in settings.items(): print(f"{name}: Accuracy = {train(factory):.3f}")


if __name__ == "__main__": main()
