# Rust Autodiff Tutorial

[![CI](https://github.com/hmyuuu/ad-rs-tutorial/actions/workflows/ci.yml/badge.svg)](https://github.com/hmyuuu/ad-rs-tutorial/actions/workflows/ci.yml)
[![Docs](https://github.com/hmyuuu/ad-rs-tutorial/actions/workflows/docs.yml/badge.svg)](https://hmyuuu.github.io/ad-rs-tutorial/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive tutorial for learning Rust's experimental `std::autodiff` feature (automatic differentiation).

## Overview

This tutorial teaches automatic differentiation from the ground up, assuming no prior AD knowledge. You'll learn:

- What automatic differentiation is and how it differs from numerical/symbolic differentiation
- Forward mode vs reverse mode AD
- How to use Rust's `#[autodiff]` attribute
- Activity annotations (`Active`, `Const`, `Duplicated`)
- Practical examples from `neural network layers` to `quantum optimal control`

## Project Structure

```
ad-rs-tutorial/
├── docs/                    # mdBook tutorial
├── examples/
│   ├── 01_scalar_square/    # f(x) = x²
│   ├── 02_scalar_sin/       # f(x) = sin(x)
│   ├── 03_multi_variable/   # f(x,y) = x² + xy + y²
│   ├── 04_rosenbrock/       # Gradient descent optimization
│   ├── 05_vector_dot/       # Dot product gradient
│   ├── 06_vector_norm/      # L2 norm gradient
│   ├── 07_mse_loss/         # MSE loss function
│   ├── 08_cross_entropy/    # Cross-entropy loss
│   ├── 09_linear_layer/     # Neural network layer
│   ├── 10_forward_mode/     # Forward mode AD
│   ├── 11_activity_demo/    # All activity annotations
│   ├── 12_control_flow/     # if/else, loops
│   ├── 13_complex_function/ # Complex number operations
│   └── 14_quantum_control/  # Quantum optimal control
└── Cargo.toml               # Workspace configuration
```

## Prerequisites

- Rust nightly toolchain with Enzyme backend
- Basic Rust knowledge

## Setup

### Option 1: Use Pre-built Enzyme Toolchain (Recommended)

If you have access to a pre-built Enzyme toolchain:

```bash
# Add the enzyme toolchain
rustup toolchain link enzyme /path/to/enzyme-toolchain

# Verify it works
RUSTFLAGS="-Z autodiff=Enable" cargo +enzyme build
```

### Option 2: Build from Source

See [rust-lang/rust#124509](https://github.com/rust-lang/rust/issues/124509) for instructions on building rustc with Enzyme support.

## Running Examples

Each example can be run individually:

```bash
# Run a specific example (requires enzyme toolchain)
RUSTFLAGS="-Z autodiff=Enable" cargo +enzyme run -p scalar_square
RUSTFLAGS="-Z autodiff=Enable" cargo +enzyme run -p rosenbrock

# Build all examples
RUSTFLAGS="-Z autodiff=Enable" cargo +enzyme build --workspace

# Run all examples
make run-all
```

### Example List

| Example | Description |
|---------|-------------|
| `scalar_square` | Basic f(x) = x² derivative |
| `scalar_sin` | Taylor series sin(x) |
| `multi_variable` | Multi-variable gradients |
| `rosenbrock` | Gradient descent optimization |
| `vector_dot` | Dot product gradient |
| `vector_norm` | L2 norm gradient |
| `mse_loss` | Mean squared error loss |
| `cross_entropy` | Binary cross-entropy loss |
| `linear_layer` | Neural network layer |
| `forward_mode` | Forward mode AD |
| `activity_demo` | Activity annotations demo |
| `control_flow` | Control flow (if/else, loops) |
| `complex_function` | Complex number differentiation |
| `quantum_control` | Quantum optimal control (>99.99% fidelity) |

## Important: Enzyme Limitations

When writing code for Enzyme autodiff, be aware of these limitations:

### Use `while` loops instead of `for` iterators

Enzyme cannot deduce types for Rust's `Range` iterators. Use explicit `while` loops:

```rust
// DON'T - Enzyme can't handle Range iterators
for i in 0..n {
    sum += x[i];
}

// DO - Use while loops instead
let mut i = 0;
while i < n {
    sum += x[i];
    i += 1;
}
```

### Avoid unsupported std library functions

Some std functions like `abs()`, `sqrt()`, `sin()`, `cos()`, `ln()` may not have Enzyme support. Use manual implementations:

```rust
// Manual abs
let abs_x = if x < 0.0 { -x } else { x };

// Newton-Raphson sqrt
let mut guess = x / 2.0;
let mut i = 0;
while i < 10 {
    guess = (guess + x / guess) / 2.0;
    i += 1;
}
```

## Documentation

Read the full tutorial online: [https://hmyuuu.github.io/ad-rs-tutorial/](https://hmyuuu.github.io/ad-rs-tutorial/)

Or build locally:

```bash
# Install mdbook if needed
cargo install mdbook

# Build and serve documentation
make docs-serve
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## References

- [Rust autodiff RFC](https://github.com/rust-lang/rfcs/pull/3453)
- [Enzyme AD](https://enzyme.mit.edu/)
- [Automatic Differentiation in Machine Learning: a Survey](https://arxiv.org/abs/1502.05767)
