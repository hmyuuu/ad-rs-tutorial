# Multi-Variable Functions

Real-world functions often have multiple inputs. Reverse mode efficiently computes all partial derivatives in a single backward pass.

## Example: Two Variables

```rust
#![feature(autodiff)]
use std::autodiff::autodiff;

// f(x, y) = x² + xy + y²
#[autodiff(d_quadratic, Reverse, Active, Active, Active)]
fn quadratic(x: f64, y: f64) -> f64 {
    x * x + x * y + y * y
}

fn main() {
    let x = 2.0;
    let y = 3.0;

    let (z, grad_x, grad_y) = d_quadratic(x, y, 1.0);

    println!("f({x}, {y}) = {z}");
    println!("∂f/∂x = {grad_x}");  // 2x + y = 7
    println!("∂f/∂y = {grad_y}");  // x + 2y = 8
}
```

## Understanding the Gradients

For f(x, y) = x² + xy + y²:

- ∂f/∂x = 2x + y
- ∂f/∂y = x + 2y

At (2, 3):
- ∂f/∂x = 2(2) + 3 = 7 ✓
- ∂f/∂y = 2 + 2(3) = 8 ✓

## The Gradient Vector

The gradient ∇f is a vector of all partial derivatives:

```
∇f = [∂f/∂x, ∂f/∂y] = [7, 8]
```

This vector points in the direction of steepest ascent.

## Vector Inputs with Duplicated

For array inputs, use `Duplicated`:

```rust
#[autodiff(d_sum_squares, Reverse, Duplicated, Active)]
fn sum_squares(x: &[f64]) -> f64 {
    x.iter().map(|xi| xi * xi).sum()
}

fn main() {
    let x = [1.0, 2.0, 3.0, 4.0];
    let mut grad = [0.0; 4];

    let y = d_sum_squares(&x, &mut grad, 1.0);

    println!("f(x) = {y}");        // 30
    println!("∇f = {:?}", grad);   // [2, 4, 6, 8]
}
```

## Mixed Annotations

Combine `Active`, `Const`, and `Duplicated` as needed:

```rust
// Weighted sum: f(x, w) = Σ wᵢ * xᵢ²
// We want gradients for x, but w is constant
#[autodiff(d_weighted, Reverse, Duplicated, Const, Active)]
fn weighted_sum_squares(x: &[f64], weights: &[f64]) -> f64 {
    x.iter()
        .zip(weights.iter())
        .map(|(xi, wi)| wi * xi * xi)
        .sum()
}

fn main() {
    let x = [1.0, 2.0, 3.0];
    let w = [1.0, 2.0, 3.0];
    let mut grad = [0.0; 3];

    let y = d_weighted(&x, &mut grad, &w, 1.0);

    // ∂f/∂xᵢ = 2 * wᵢ * xᵢ
    println!("∇f = {:?}", grad);  // [2, 8, 18]
}
```

## Key Points

- Each `Active` input adds one gradient to the return tuple
- `Duplicated` inputs use a separate gradient buffer
- All gradients are computed in a single backward pass
- The gradient vector points toward steepest ascent
