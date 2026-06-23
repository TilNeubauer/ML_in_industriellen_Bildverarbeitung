"""Exercise 7.1: Trainierbare Parameter eines vollständig verbundenen NN.

Aufgabenstellung: Bestimme Formen von A1..A4 und b1..b4, Sparsity, die
ausmultiplizierte lineare Komposition und bewerte diese Darstellung.

Für Schichtgrößen n0..n4 gilt allgemein: A_k hat Form (n_k, n_(k-1)), b_k
hat Form (n_k,). Vollverbundene A_k sind nicht sparse; y = A4(A3(A2(A1x+b1)
+b2)+b3)+b4. Die schrittweise Berechnung ist speicherschonender und erlaubt
Aktivierungen zwischen den Schichten.
"""

import torch


def main():
    sizes = (64, 32, 16, 8, 10)
    layers = [torch.nn.Linear(sizes[i], sizes[i + 1]) for i in range(4)]
    for index, layer in enumerate(layers, 1):
        print(f"A{index}: {tuple(layer.weight.shape)}, b{index}: {tuple(layer.bias.shape)}")


if __name__ == "__main__": main()
