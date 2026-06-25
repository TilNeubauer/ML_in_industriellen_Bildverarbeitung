"""Exercise 7.5: Train and validation split.



Aufruf: pdm run python exercises/07_neural_networks/exercise_7_5_train_validation_split.py
"""

import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

SEED = 6020


def tensors(data, labels):
    return torch.tensor(data, dtype=torch.float32), torch.tensor(labels)


def create_model():
    return torch.nn.Sequential(
        torch.nn.Linear(64, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 10),
    )


def accuracy(model, test_x, test_y):
    predictions = model(test_x).argmax(dim=1)
    return (predictions == test_y).float().mean().item()


def train_fixed_split(optimizer_factory, epochs=120):
    digits = load_digits()
    train_x, test_x, train_y, test_y = train_test_split(
        digits.data / 16,
        digits.target,
        test_size=0.25,
        stratify=digits.target,
        random_state=SEED,
    )
    train_x, train_y = tensors(train_x, train_y)
    test_x, test_y = tensors(test_x, test_y)

    torch.manual_seed(SEED)
    model = create_model()
    optimizer = optimizer_factory(model.parameters())
    loss_function = torch.nn.CrossEntropyLoss()

    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_function(model(train_x), train_y)
        loss.backward()
        optimizer.step()

    return accuracy(model, test_x, test_y)


def train_resplit_each_epoch(optimizer_factory, epochs=120):
    digits = load_digits()

    torch.manual_seed(SEED)
    model = create_model()
    optimizer = optimizer_factory(model.parameters())
    loss_function = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        train_x, _, train_y, _ = train_test_split(
            digits.data / 16,
            digits.target,
            test_size=0.25,
            stratify=digits.target,
            random_state=SEED + epoch,
        )
        train_x, train_y = tensors(train_x, train_y)

        optimizer.zero_grad()
        loss = loss_function(model(train_x), train_y)
        loss.backward()
        optimizer.step()

    _, test_x, _, test_y = train_test_split(
        digits.data / 16,
        digits.target,
        test_size=0.25,
        stratify=digits.target,
        random_state=SEED,
    )
    test_x, test_y = tensors(test_x, test_y)
    return accuracy(model, test_x, test_y)


def main():
    optimizers = {
        "SGD": lambda params: torch.optim.SGD(params, lr=0.1, momentum=0.9),
        "Adam": lambda params: torch.optim.Adam(params, lr=0.01),
        "Adagrad": lambda params: torch.optim.Adagrad(params, lr=0.1),
    }

    for name, optimizer_factory in optimizers.items():
        fixed = train_fixed_split(optimizer_factory)
        resplit = train_resplit_each_epoch(optimizer_factory)
        print(f"{name}: fixed={fixed:.3f}, resplit_each_epoch={resplit:.3f}")


if __name__ == "__main__":
    main()
