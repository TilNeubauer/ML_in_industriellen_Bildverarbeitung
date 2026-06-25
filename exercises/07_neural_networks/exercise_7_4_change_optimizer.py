"""Exercise 7.4: Change the optimizer.



Aufruf: pdm run python exercises/07_neural_networks/exercise_7_4_change_optimizer.py
"""

import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

SEED = 6020


def load_data():
    digits = load_digits()
    train_x, test_x, train_y, test_y = train_test_split(
        digits.data / 16,
        digits.target,
        test_size=0.25,
        stratify=digits.target,
        random_state=SEED,
    )
    return (
        torch.tensor(train_x, dtype=torch.float32),
        torch.tensor(test_x, dtype=torch.float32),
        torch.tensor(train_y),
        torch.tensor(test_y),
    )


def create_model():
    return torch.nn.Sequential(
        torch.nn.Linear(64, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 10),
    )


def train(optimizer_factory, epochs=200):
    train_x, test_x, train_y, test_y = load_data()

    torch.manual_seed(SEED)
    model = create_model()
    optimizer = optimizer_factory(model.parameters())
    loss_function = torch.nn.CrossEntropyLoss()

    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_function(model(train_x), train_y)
        loss.backward()
        optimizer.step()

    predictions = model(test_x).argmax(dim=1)
    return (predictions == test_y).float().mean().item()


def main():
    optimizers = {
        "SGD": lambda params: torch.optim.SGD(params, lr=0.1, momentum=0.9),
        "Adam": lambda params: torch.optim.Adam(params, lr=0.01),
        "AdamW": lambda params: torch.optim.AdamW(params, lr=0.01),
        "Adamax": lambda params: torch.optim.Adamax(params, lr=0.01),
        "Adadelta": lambda params: torch.optim.Adadelta(params, lr=1.0),
        "Adagrad": lambda params: torch.optim.Adagrad(params, lr=0.1),
    }

    for name, optimizer_factory in optimizers.items():
        accuracy = train(optimizer_factory)
        print(f"{name}: accuracy={accuracy:.3f}")


if __name__ == "__main__":
    main()
