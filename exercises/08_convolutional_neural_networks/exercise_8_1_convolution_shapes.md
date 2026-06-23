# Exercise 8.1 – Output shape per input and kernel shape

## Aufgabenstellung

Für ein Eingabebild der Größe `(a × b)` und einen Kernel der Größe `(c × d)`:

1. Gib die Ausgabegröße ohne Padding und mit Stride `s=1` an.
2. Erweitere sie um Padding bei `s=1`.
3. Erweitere sie um Stride ohne Padding.
4. Gib die allgemeine Formel für Padding und Stride an.
5. Bestimme das Padding für gleiche Ein- und Ausgabegröße bei `s=1`.

## Lösung


Die folgenden Formeln gelten für jede Raumrichtung getrennt. Die 2D-Ausgabe
entsteht, indem die Formel einmal für `a, c` und einmal für `b, d` verwendet
wird.

| Fall | Höhe der Ausgabe |
| --- | --- |
| Kein Padding, `s=1` | `a - c + 1` |
| Padding `p`, `s=1` | `a + 2p - c + 1` |
| Kein Padding, Stride `s` | `floor((a - c) / s) + 1` |
| Padding `p`, Stride `s` | `floor((a + 2p - c) / s) + 1` |

Damit bei `s=1` die Ausgabe genauso hoch wie die Eingabe wird, muss gelten:

```text
a + 2p - c + 1 = a
p = (c - 1) / 2
```

Das ist bei ungeraden Kernelgrößen ganzzahlig, etwa `p=1` für `c=3`.
Bei geraden Kernelgrößen ist symmetrisches „same padding“ ohne zusätzliche
asymmetrische Randbehandlung nicht möglich.
