# PairShift

## Building regression model for symmetric features with TensorFlow 2.0

### Basic version


There are N items in a pool. Some items are identical (with the same item name). A person comes to the pool and randomly picks 2 items and measures the "difference" between the two items, where the difference can be measured in a multi-dimensional feature space. The person records the names of both items (they could have the same name), and the distance, then throws the 2 items back to the pool. This is repeated until there are M measurements. With this data, we build a model to predict the difference between any pair of item names.

The prediction should have the following constraints:

1. If item1 == item2, different is 0.
2. If we swap item1 and item2 in an input, the difference should become -difference.

We use TensorFlow to build a linear model with no hidden layer. In order to satisty the above two constraints, each item name should have just one weight with opposite signs being in item1 and item2, instead of two freely adjustable weights being in item1 and item2. Moreover, the bias is set to 0.

### Advanced version

Assume each item, when picked up each time, generates a tag value `u` to be seen, which can be different from time to time. Assume the true value `v` of the item has a linear correction to the tag value, i.e. `v = c + (1 + d) * u`. If the correction is small, both `c` and `d` are small. The user uses the true values to match two items in a pair, which means

```
c_i + (1 + d_i) * u_i = c_j + (1 + d_j) * u_j
```

or

```
(c_i + d_i * u_i) - (c_j + d_j * u_j) = u_j - u_i
```

which forms the basic equation to solve for `c`s and `d`s. `u` values are known tag values so we use `u_j - u_i` as response in the label column. `c` and `d` values are trainable variables.
