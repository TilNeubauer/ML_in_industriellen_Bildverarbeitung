# Exercise 8.1: Convolution Output Shape

Input:

```text
height = a
width  = b
kernel height = c
kernel width  = d
padding = p
stride  = s
```

## No Padding, Stride 1

```text
output height = a - c + 1
output width  = b - d + 1
```

## Padding, Stride 1

```text
output height = a + 2p - c + 1
output width  = b + 2p - d + 1
```

## No Padding, Stride s

```text
output height = floor((a - c) / s) + 1
output width  = floor((b - d) / s) + 1
```

## Padding and Stride s

```text
output height = floor((a + 2p - c) / s) + 1
output width  = floor((b + 2p - d) / s) + 1
```

## Same Output Size

For `s = 1`, same output height as input height means:

```text
a + 2p - c + 1 = a
2p = c - 1
p = (c - 1) / 2
```

For the width:

```text
p = (d - 1) / 2
```

So for a `3 x 3` kernel, use:

```text
p = 1
```

This works cleanly for odd kernel sizes. Even kernel sizes need asymmetric padding if the output should have exactly the same size.
