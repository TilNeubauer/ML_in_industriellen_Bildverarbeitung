"""Exercise 7.3: 

Aufruf: pdm run python exercises/07_neural_networks/exercise_7_3_learning_rate_optimizer.py
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


def train(train_x, test_x, train_y, test_y, learning_rate, momentum, epochs=200):
    torch.manual_seed(SEED)
    model = torch.nn.Sequential(
        torch.nn.Linear(64, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 10),
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    loss_function = torch.nn.CrossEntropyLoss()

    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_function(model(train_x), train_y)
        loss.backward()
        optimizer.step()

    probabilities = torch.softmax(model(test_x), dim=1)
    predictions = probabilities.argmax(dim=1)
    accuracy = (predictions == test_y).float().mean().item()
    confidence = probabilities.max(dim=1).values.mean().item()

    return accuracy, confidence


def main():
    train_x, test_x, train_y, test_y = load_data()

    for learning_rate in (1e-1, 1e-2, 1e-3, 1e-4):
        for momentum in (0.0, 0.5, 0.9):
            accuracy, confidence = train(
                train_x,
                test_x,
                train_y,
                test_y,
                learning_rate,
                momentum,
            )
            print(
                f"lr={learning_rate:.0e}, momentum={momentum:.1f}: "
                f"accuracy={accuracy:.3f}, confidence={confidence:.3f}"
            )


if __name__ == "__main__":
    main()
