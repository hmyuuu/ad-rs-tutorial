# Vector Operations

Vector operations are common in scientific computing and machine learning. Here's how to differentiate them.

## Dot Product

The dot product f(x) = x · w has a simple gradient: ∂f/∂x = w

```rust
#![feature(autodiff)]
use std::autodiff::autodiff;

#[autodiff(d_dot, Reverse, Duplicated, Const, Active)]
fn dot_product(x: &[f64], w: &[f64]) -> f64 {
    x.iter().zip(w.iter()).map(|(xi, wi)| xi * wi).sum()
}

fn main() {
    let x = [1.0, 2.0, 3.0];
    let w = [0.5, 1.5, 2.5];
    let mut grad = [0.0; 3];

    let y = d_dot(&x, &mut grad, &w, 1.0);

    println!("x · w = {y}");           // 11
    println!("∇f = {:?}", grad);       // [0.5, 1.5, 2.5] = w
}
```

Run: `RUSTFLAGS="-Z autodiff=Enable" cargo run -p vector_dot`

## L2 Norm

The L2 norm f(x) = ||x||₂ = √(Σ xᵢ²) has gradient ∂f/∂xᵢ = xᵢ / ||x||₂

```rust
#[autodiff(d_norm, Reverse, Duplicated, Active)]
fn l2_norm(x: &[f64]) -> f64 {
    x.iter().map(|xi| xi * xi).sum::<f64>().sqrt()
}

fn main() {
    let x = [3.0, 4.0];
    let mut grad = [0.0; 2];

    let norm = d_norm(&x, &mut grad, 1.0);

    println!("||x||₂ = {norm}");       // 5
    println!("∇||x||₂ = {:?}", grad);  // [0.6, 0.8] = x/||x||
}
```

Run: `RUSTFLAGS="-Z autodiff=Enable" cargo run -p vector_norm`

## Sum of Squares

A common building block: f(x) = Σ xᵢ²

```rust
#[autodiff(d_sum_sq, Reverse, Duplicated, Active)]
fn sum_squares(x: &[f64]) -> f64 {
    x.iter().map(|xi| xi * xi).sum()
}

fn main() {
    let x = [1.0, 2.0, 3.0];
    let mut grad = [0.0; 3];

    let y = d_sum_sq(&x, &mut grad, 1.0);

    println!("Σxᵢ² = {y}");           // 14
    println!("∇f = {:?}", grad);       // [2, 4, 6] = 2x
}
```

## Weighted Sum

Combining vectors with weights:

```rust
#[autodiff(d_weighted, Reverse, Duplicated, Const, Active)]
fn weighted_sum(x: &[f64], weights: &[f64]) -> f64 {
    x.iter()
        .zip(weights.iter())
        .map(|(xi, wi)| wi * xi)
        .sum()
}
```

Note: `weights` is `Const` because we don't need its gradient.

## Key Patterns

1. **Input vectors**: Use `Duplicated` with a gradient buffer
2. **Constant vectors**: Use `Const` (no gradient buffer needed)
3. **Gradient buffer**: Must be same length as input, initialized to zero
4. **Accumulation**: Gradients are accumulated into the buffer

## Common Gotchas

```rust
// WRONG: Forgetting to zero the gradient buffer
let mut grad = [1.0; 3];  // Should be [0.0; 3]

// WRONG: Buffer wrong size
let x = [1.0, 2.0, 3.0];
let mut grad = [0.0; 2];  // Should be length 3

// RIGHT
let mut grad = [0.0; 3];
d_func(&x, &mut grad, 1.0);
```
