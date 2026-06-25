# Exercise 7.1: Neural-Network-Parameter

Für die Abbildung gilt:

```text
x  hat 8 Einträge
x1 hat 4 Einträge
x2 hat 3 Einträge
x3 hat 4 Einträge
y  hat 2 Einträge
```

## Matrixformen

```text
A1: (4, 8)
A2: (3, 4)
A3: (4, 3)
A4: (2, 4)
```

Allgemein gilt:

```text
Ak: (Anzahl Neuronen in Schicht k, Anzahl Neuronen in Schicht k-1)
```

## Sparse Matrizen

Wenn die Schichten als vollständig verbunden interpretiert werden, ist keine der Matrizen sparse.


## Biasformen

```text
b1: (4,)
b2: (3,)
b3: (4,)
b4: (2,)
```

Jeder Biasvektor hat so viele Einträge wie die Zielschicht Neuronen hat.

## Formel

Für lineare Aktivierungen `f(x) = x` gilt schichtweise:

```text
x1 = A1 x + b1
x2 = A2 x1 + b2
x3 = A3 x2 + b3
y  = A4 x3 + b4
```

Ausmultipliziert:

```text
y = A4 A3 A2 A1 x
    + A4 A3 A2 b1
    + A4 A3 b2
    + A4 b3
    + b4
```

## Ist das eine gute Rechenform?

Nein, für die praktische Berechnung ist die Form nicht ideal.

Schichtweise zu rechnen ist besser, weil es übersichtlicher ist und weniger Zwischenspeicher braucht.