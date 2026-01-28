# Const

The `Const` annotation marks parameters that should **not** be differentiated. The autodiff system treats them as constants.

## When to Use

Use `Const` for:
- Hyperparameters (learning rate, regularization strength)
- Configuration values
- Ground truth labels in loss functions
- Any parameter you don't need gradients for

## Syntax

```rust
#[autodiff(d_func, Reverse, Active, Const, Active)]
fn func(x: f64, config: f64) -> f64 { ... }
//            ↑        ↑
//         Active    Const (no gradient)
```

## Example

```rust
#![feature(autodiff)]
use std::autodiff::autodiff;

// scale is constant - we don't compute ∂f/∂scale
#[autodiff(d_scaled, Reverse, Active, Const, Active)]
fn scaled_square(x: f64, scale: f64) -> f64 {
    scale * x * x
}

fn main() {
    let x = 3.0;
    let scale = 2.0;

    // Only get gradient w.r.t. x, not scale
    let (y, grad_x) = d_scaled(x, scale, 1.0);

    println!("f({x}) = {y}");        // f(3) = 18
    println!("∂f/∂x = {grad_x}");    // ∂f/∂x = 12
}
```

## Generated Function Signature

```rust
// Original
fn scaled_square(x: f64, scale: f64) -> f64

// Generated - note: no gradient for scale
fn d_scaled(x: f64, scale: f64, seed: f64) -> (f64, f64)
//                  ↑                               ↑
//              passed through                 only grad_x
```

The `Const` parameter is passed through unchanged, but no gradient is computed or returned for it.

## Practical Example: Loss Function

In machine learning, targets are constants:

```rust
#[autodiff(d_mse, Reverse, Duplicated, Const, Active)]
fn mse_loss(pred: &[f64], target: &[f64]) -> f64 {
    pred.iter()
        .zip(target.iter())
        .map(|(p, t)| (p - t).powi(2))
        .sum::<f64>() / pred.len() as f64
}
```

Here:
- `pred` is `Duplicated` — we want gradients
- `target` is `Const` — ground truth doesn't need gradients

## Benefits of Const

1. **Efficiency**: No unnecessary gradient computation
2. **Clarity**: Documents which parameters are trainable
3. **Correctness**: Prevents accidental gradient flow

## Key Points

- `Const` parameters pass through unchanged
- No gradient is computed or returned for `Const` parameters
- Use for hyperparameters, targets, and configuration
- Improves both efficiency and code clarity
