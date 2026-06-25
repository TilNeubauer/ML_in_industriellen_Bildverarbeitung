# Exercise 7.2: Explain `.summary()`

## Param #

Nur `Linear`-Layer haben trainierbare Parameter. `Tanh` hat keine Gewichte und keinen Bias.

Für einen `Linear`-Layer gilt:

```text
Parameter = n_out * n_in + n_out
            Gewichte       Bias
```

Im Modell:

```text
Linear 1024 -> 2: 2 * 1024 + 2 = 2050
Tanh:             0
Linear 2 -> 2:    2 * 2 + 2    = 6

Gesamt: 2050 + 6 = 2056
```

## Datentyp

Die Parameter brauchen `8.03125 KB` Speicher:

```text
2056 * 4 Byte = 8224 Byte
8224 / 1024   = 8.03125 KB
```

Ein Parameter braucht also `4 Byte`.

Das entspricht `float32`, also 32 Bit pro Parameter.
