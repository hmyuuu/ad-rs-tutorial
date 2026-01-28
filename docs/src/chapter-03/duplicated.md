# Duplicated

The `Duplicated` annotation is used for reference types (slices, arrays) where you want to compute gradients. You provide both the input and a mutable gradient buffer.

## When to Use

Use `Duplicated` for:
- Slice parameters (`&[f64]`)
- Array references
- Any reference type where you need element-wise gradients

## Syntax

```rust
#[autodiff(d_func, Reverse, Duplicated, Active)]
fn func(x: &[f64]) -> f64 { ... }
```

## How It Works

With `Duplicated`, the generated function takes:
1. The original input reference
2. A mutable gradient buffer of the same shape

```rust
// Original
fn dot(x: &[f64]) -> f64

// Generated
fn d_dot(x: &[f64], dx: &mut [f64], seed: f64) -> f64
//                  ↑
//          gradient buffer (filled by autodiff)
```

## Example

```rust
#![feature(autodiff)]
use std::autodiff::autodiff;

#[autodiff(d_sum_squares, Reverse, Duplicated, Active)]
fn sum_squares(x: &[f64]) -> f64 {
    x.iter().map(|xi| xi * xi).sum()
}

fn main() {
    let x = [1.0, 2.0, 3.0];
    let mut grad = [0.0; 3];  // Gradient buffer

    let y = d_sum_squares(&x, &mut grad, 1.0);

    println!("f(x) = {y}");           // 14
    println!("∇f = {:?}", grad);      // [2, 4, 6]
}
```

## Understanding the Gradient

For f(x) = Σ xᵢ²:
- ∂f/∂xᵢ = 2xᵢ

So for x = [1, 2, 3]:
- ∂f/∂x₀ = 2(1) = 2
- ∂f/∂x₁ = 2(2) = 4
- ∂f/∂x₂ = 2(3) = 6

The gradient buffer is filled with [2, 4, 6] ✓

## Multiple Duplicated Parameters

You can have multiple `Duplicated` parameters:

```rust
#[autodiff(d_dot, Reverse, Duplicated, Duplicated, Active)]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(ai, bi)| ai * bi).sum()
}

// Generated:
fn d_dot(
    a: &[f64], da: &mut [f64],  // First Duplicated
    b: &[f64], db: &mut [f64],  // Second Duplicated
    seed: f64
) -> f64
```

## Combining with Const

Often you want gradients for some arrays but not others:

```rust
#[autodiff(d_weighted_sum, Reverse, Duplicated, Const, Active)]
fn weighted_sum(x: &[f64], weights: &[f64]) -> f64 {
    x.iter().zip(weights.iter()).map(|(xi, wi)| xi * wi).sum()
}

// Only x gets a gradient buffer, weights is passed through
fn d_weighted_sum(
    x: &[f64], dx: &mut [f64],
    weights: &[f64],  // No gradient buffer
    seed: f64
) -> f64
```

## Key Points

- `Duplicated` is for reference types (slices, arrays)
- You must provide a mutable gradient buffer
- The buffer must have the same length as the input
- Gradients are accumulated into the buffer
- Initialize the buffer to zeros before calling
