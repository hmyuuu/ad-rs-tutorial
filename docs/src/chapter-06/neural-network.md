# Neural Network Layers

Neural networks are compositions of differentiable layers. Let's implement a linear layer and compute its gradients.

## Linear Layer: y = Wx + b

A linear layer transforms input x using weights W and bias b:

```rust
#![feature(autodiff)]
use std::autodiff::autodiff;

/// Linear layer with MSE loss
/// Returns loss = ||Wx + b - target||²
#[autodiff(d_linear, Reverse, Const, Duplicated, Duplicated, Const, Active)]
fn linear_loss(
    x: &[f64],        // Input (constant)
    weights: &[f64],  // Weights (need gradients)
    bias: &[f64],     // Bias (need gradients)
    target: &[f64],   // Target (constant)
) -> f64 {
    // y = Wx + b (2x2 matrix × 2-vector)
    let y0 = weights[0] * x[0] + weights[1] * x[1] + bias[0];
    let y1 = weights[2] * x[0] + weights[3] * x[1] + bias[1];

    // MSE loss
    let d0 = y0 - target[0];
    let d1 = y1 - target[1];
    (d0 * d0 + d1 * d1) / 2.0
}
```

## Computing Gradients

```rust
fn main() {
    let x = [1.0, 2.0];
    let weights = [0.5, 0.5, 0.5, 0.5];  // 2x2 matrix
    let bias = [0.1, 0.1];
    let target = [1.0, 2.0];

    let mut grad_w = [0.0; 4];
    let mut grad_b = [0.0; 2];

    let loss = d_linear(
        &x,
        &weights, &mut grad_w,
        &bias, &mut grad_b,
        &target,
        1.0
    );

    println!("Loss: {loss}");
    println!("∂L/∂W: {:?}", grad_w);
    println!("∂L/∂b: {:?}", grad_b);
}
```

Run: `RUSTFLAGS="-Z autodiff=Enable" cargo run -p linear_layer`

## Gradient Descent Update

```rust
let lr = 0.1;

// Update weights
for (w, g) in weights.iter_mut().zip(grad_w.iter()) {
    *w -= lr * g;
}

// Update bias
for (b, g) in bias.iter_mut().zip(grad_b.iter()) {
    *b -= lr * g;
}
```

## Understanding the Gradients

For y = Wx + b with loss L = ||y - t||²:

- ∂L/∂W = (y - t) × xᵀ
- ∂L/∂b = (y - t)

The gradients tell us:
- How each weight affects the loss
- How to adjust weights to reduce loss

## Building Deeper Networks

Stack multiple layers:

```rust
fn two_layer_network(x: &[f64], w1: &[f64], b1: &[f64],
                     w2: &[f64], b2: &[f64]) -> f64 {
    // Layer 1
    let h = linear(x, w1, b1);

    // Activation (e.g., ReLU)
    let h_act: Vec<f64> = h.iter().map(|&v| v.max(0.0)).collect();

    // Layer 2
    let y = linear(&h_act, w2, b2);

    // Loss
    mse(&y, &target)
}
```

Autodiff handles the entire computation graph!

## Annotation Pattern for Layers

| Parameter | Annotation | Reason |
|-----------|------------|--------|
| Input x | `Const` | Fixed during backward pass |
| Weights W | `Duplicated` | Need gradients for learning |
| Bias b | `Duplicated` | Need gradients for learning |
| Target | `Const` | Ground truth, not learned |

## Key Points

- Neural network layers are just differentiable functions
- Use `Duplicated` for learnable parameters
- Use `Const` for inputs and targets
- Autodiff computes all gradients in one backward pass
- Stack layers by composing functions
