# Exercise 7.2 – Explain the output of `.summary()`

## Aufgabenstellung

Erkläre die Spalte `Param #` der erweiterten Ausgabe von `torchinfo.summary()`.
Leite die Zahlen her. Was sagt eine Parametergröße von `8.03125 KB` über den
verwendeten Datentyp aus?

## Lösung

Das Modell besteht aus zwei trainierbaren `Linear`-Schichten. `Tanh` besitzt
keine trainierbaren Parameter und erhält deshalb `--` in der Spalte `Param #`.


| Layer | Rechnung | Parameter |
| --- | --- | ---: |
| Erste lineare Schicht (`1024 → 2`) | Gewichte: `2 × 1024`; Bias: `2` | `2 × 1024 + 2 = 2,050` |
| `Tanh` | keine Gewichte, kein Bias | `0` |
| Zweite lineare Schicht (`2 → 2`) | Gewichte: `2 × 2`; Bias: `2` | `2 × 2 + 2 = 6` |
| **Gesamt** | `2,050 + 6` | **2,056** |

Für eine lineare Schicht mit `n_in` Eingängen und `n_out` Ausgängen gilt also:

```text
Param # = n_out × n_in + n_out
          Gewichte       Bias
```

Die Parameter belegen `8.03125 KB`:

```text
2,056 Parameter × 4 Byte = 8,224 Byte
8,224 Byte / 1,024 = 8.03125 KB
```

Jeder Parameter braucht damit **4 Byte**. Das entspricht dem üblichen
PyTorch-Standarddatentyp **`torch.float32`** (32 Bit pro Parameter).
