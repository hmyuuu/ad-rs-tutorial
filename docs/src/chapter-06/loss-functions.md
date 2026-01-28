# Loss Functions

Loss functions are central to machine learning. They measure how well predictions match targets, and their gradients drive learning.

## Mean Squared Error (MSE)

MSE is the most common regression loss:

L = (1/n) Σ (predᵢ - targetᵢ)²

```rust
#![feature(autodiff)]
use std::autodiff::autodiff;

#[autodiff(d_mse, Reverse, Duplicated, Const, Active)]
fn mse_loss(pred: &[f64], target: &[f64]) -> f64 {
    let n = pred.len() as f64;
    pred.iter()
        .zip(target.iter())
        .map(|(p, t)| (p - t) * (p - t))
        .sum::<f64>() / n
}

fn main() {
    let pred = [2.5, 0.0, 2.0, 8.0];
    let target = [3.0, -0.5, 2.0, 7.0];
    let mut grad = [0.0; 4];

    let loss = d_mse(&pred, &mut grad, &target, 1.0);

    println!("MSE Loss: {loss}");
    println!("Gradients: {:?}", grad);
    // Expected: (2/n) * (pred - target)
}
```

The gradient ∂L/∂predᵢ = (2/n)(predᵢ - targetᵢ) tells us how to adjust predictions.

Run: `RUSTFLAGS="-Z autodiff=Enable" cargo run -p mse_loss`

## Binary Cross-Entropy

For classification with probabilities:

L = -(1/n) Σ [yᵢ log(pᵢ) + (1-yᵢ) log(1-pᵢ)]

```rust
#[autodiff(d_bce, Reverse, Duplicated, Const, Active)]
fn bce_loss(pred: &[f64], target: &[f64]) -> f64 {
    let n = pred.len() as f64;
    let eps = 1e-15;  // Numerical stability

    pred.iter()
        .zip(target.iter())
        .map(|(p, t)| {
            let p = p.clamp(eps, 1.0 - eps);
            -(t * p.ln() + (1.0 - t) * (1.0 - p).ln())
        })
        .sum::<f64>() / n
}

fn main() {
    let pred = [0.9, 0.2, 0.8, 0.3];
    let target = [1.0, 0.0, 1.0, 0.0];
    let mut grad = [0.0; 4];

    let loss = d_bce(&pred, &mut grad, &target, 1.0);

    println!("BCE Loss: {loss:.6}");
    println!("Gradients: {:?}", grad);
}
```

Run: `RUSTFLAGS="-Z autodiff=Enable" cargo run -p cross_entropy`

## Interpreting Gradients

The gradient tells us how to adjust predictions:

| Prediction | Target | Gradient | Meaning |
|------------|--------|----------|---------|
| 0.9 | 1.0 | negative | Push higher (good!) |
| 0.2 | 0.0 | positive | Push lower (good!) |
| 0.3 | 1.0 | negative | Push higher (bad prediction) |

## Why Target is Const

Notice that `target` uses `Const`:
- Targets are ground truth — we don't learn them
- No gradient computation needed
- More efficient

## Numerical Stability

Real implementations need care:

```rust
// BAD: log(0) = -∞
let loss = -(t * p.ln());

// GOOD: Clamp to avoid log(0)
let p = p.clamp(1e-15, 1.0 - 1e-15);
let loss = -(t * p.ln());
```

## Key Points

- Loss functions map predictions to a scalar
- Gradients show how to improve predictions
- Use `Const` for targets (no gradients needed)
- Handle numerical edge cases (log(0), division by zero)
- MSE for regression, cross-entropy for classification
