# Postman-Task

A feedforward neural network built from scratch in NumPy — no autograd, no `.backward()` — trained on MNIST to classify handwritten digits, with backpropagation implemented and verified manually via numerical gradient checking.

**Architecture:** 784 (input) → 128 (hidden, ReLU) → 10 (output, softmax), trained with mini-batch gradient descent and cross-entropy loss.

**Result:** ~97.9% test accuracy, with all gradients verified against numerical estimates within acceptable tolerance.

## Requirements

```bash
pip install numpy scikit-learn
```

## Files (run in this order)

This project was originally written as a single Jupyter notebook and later split into separate files, one per notebook cell. Because later files depend on variables and functions defined earlier, **they must be run in the order below, in the same session** (e.g. as cells in a notebook, or via `%run` / `exec` if run as scripts) — running them out of order or independently will cause `NameError`s or shape-mismatch errors.

| Order | File | What it does |
|---|---|---|
| 1 | `import mnsit` | Loads the MNIST dataset via `fetch_openml` and extracts `X` (pixel data) and `y` (labels). |
| 2 | `weights and biases` | Normalizes pixel values, initializes `w1`, `b1`, `w2`, `b2`, transposes `X`, one-hot encodes labels into `Y`, and creates the train/test split. |
| 3 | `functions` | Defines `softmax`, `cross_entropy`, `accuracy`, `forward`, `backward`, and `compute_loss`. |
| 4 | `test and train loop` | Runs mini-batch gradient descent training for 50 epochs, printing loss and train/test accuracy every 10 epochs. |
| 5 | `gradient check` | Defines `gradient_check`, which numerically verifies the manually-computed gradients for any of `w1`, `b1`, `w2`, `b2` against a finite-difference approximation. |
| 6 | `Run Gradient check` | Calls `gradient_check` for each of `w1`, `b1`, `w2`, `b2` and prints the comparison results. |

## Usage

If running as a notebook, execute each file's cell top to bottom in the order above.

If running as plain `.py` scripts in one shared session, either:
- Paste/run their contents in order in a single interactive Python or IPython session, or
- Use `%run "import mnsit.py"`, then `%run "weights and biases.py"`, etc., sequentially in IPython/Jupyter.

Running any file 3–6 on its own, without having run the prior files first in the same session, will fail since they depend on variables (`X`, `Y`, `w1`, etc.) and functions defined earlier.

## Notes

- `forward()` returns `(z1, a1, z2, a2)` — the intermediate values are needed for backpropagation, not just the final prediction.
- Gradient checking uses `epsilon=1e-5` and samples 10 random parameter positions per array (checking every parameter individually would require ~200,000+ forward passes and isn't necessary to verify correctness).
- A full write-up covering the backpropagation derivation, gradient-checking methodology, training results, and a debugging log is included separately (see `neural_network_writeup.md`, if present in this repo).
