"""Exercise 7.6: Train with softmax.


Aufruf: pdm run python exercises/07_neural_networks/exercise_7_6_train_with_softmax.py
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


def create_model(use_softmax):
    layers = [
        torch.nn.Linear(64, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 10),
    ]
    if use_softmax:
        layers.append(torch.nn.Softmax(dim=1))
    return torch.nn.Sequential(*layers)


def train(use_softmax, epochs=200):
    train_x, test_x, train_y, test_y = load_data()

    torch.manual_seed(SEED)
    model = create_model(use_softmax)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    loss_function = torch.nn.CrossEntropyLoss()

    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_function(model(train_x), train_y)
        loss.backward()
        optimizer.step()

    predictions = model(test_x).argmax(dim=1)
    return (predictions == test_y).float().mean().item()


def main():
    print(f"Without Softmax in model: accuracy={train(use_softmax=False):.3f}")
    print(f"With Softmax in model:    accuracy={train(use_softmax=True):.3f}")
    print("CrossEntropyLoss already applies log-softmax internally.")


if __name__ == "__main__":
    main()
