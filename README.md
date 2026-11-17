# Neural Network from Scratch

A single-hidden-layer neural network written in NumPy, with forward propagation and backpropagation implemented by hand. No PyTorch, no TensorFlow, no autograd, nothing that computes a gradient for you.

The demo at the bottom of the file trains the network on XOR, the problem that made single-layer perceptrons famous for what they could not do.

## Why bother

Because "I know how backpropagation works" is easy to say and hard to mean. Writing it once, with the chain rule laid out explicitly over two weight matrices, is the difference between using a framework and understanding what the framework is doing when it calls `.backward()`.

## The implementation

```text
inputs ──► weights1 ──► sigmoid ──► hidden ──► weights2 ──► sigmoid ──► output
                                          ◄── error propagates back ◄──
```

| Piece | Where |
|---|---|
| Weight initialisation, sigmoid and its derivative | `__init__`, `sigmoid`, `sigmoid_derivative` |
| Forward pass through both layers | `forward` |
| Error at the output, backpropagation into the hidden layer, weight updates | `backward` |
| The training loop and the XOR demo | `train`, and the calls at the bottom of the file |

The sigmoid derivative is computed as `x * (1 - x)` on the **activation**, which is the correct form: if `x = sigmoid(z)` then `sigmoid'(z) = x(1 - x)`.

## Running it

```bash
pip install numpy
python NeuralNetwork.py
```

It prints the error every 1000 epochs, then the expected and actual outputs for the four XOR inputs.

## Changing the experiment

The architecture is set by `NeuralNetwork(2, 3, 1)`, the training pairs are `inputs` and `expected_output`, and the epoch count is the third argument to `train`. Swapping in a different truth table is a four-line change.

## What it deliberately leaves out

- **No bias terms.** The network can only fit functions that pass through the origin of each layer's input space.
- **No learning rate.** Every update applies the full delta, so a larger dataset would need one to stay stable.
- **No mini-batches.** All four examples go through in one step.
- **No regularisation, no validation split, no metrics beyond mean absolute error.**

These are the pieces that separate a teaching implementation from a usable one, and each is a natural next exercise.

## License

MIT. See `LICENSE`.
