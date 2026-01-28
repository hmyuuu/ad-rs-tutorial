# Basic Usage

Reverse mode AD computes gradients by first running the function forward, then propagating derivatives backward.

## The Reverse Mode Attribute

```rust
#[autodiff(derivative_name, Reverse, ...activity_annotations...)]
fn original_function(...) -> ... { ... }
```

## Simple Example

```rust
#![feature(autodiff)]
use std::autodiff::autodiff;

#[autodiff(d_square, Reverse, Active, Active)]
fn square(x: f64) -> f64 {
    x * x
}

fn main() {
    let x = 4.0;
    let (y, grad) = d_square(x, 1.0);

    println!("f({x}) = {y}");      // 16
    println!("f'({x}) = {grad}");  // 8
}
```

## How Reverse Mode Works

1. **Forward pass**: Compute f(x) and store intermediate values
2. **Backward pass**: Propagate adjoints from output to inputs

For f(x) = x²:

```
Forward:  x=4 → v=x*x=16 → y=16
Backward: ȳ=1 → v̄=ȳ=1 → x̄=v̄*2x=8
```

## The Adjoint Seed

The second argument to the generated function is the "adjoint seed" — the derivative of some downstream loss with respect to the output.

```rust
let (y, grad) = d_square(x, seed);
// grad = seed * (df/dx)
```

With `seed = 1.0`, you get the gradient directly.

With `seed = 2.0`, you get twice the gradient (useful for chain rule).

## Example with Composition

```rust
#[autodiff(d_f, Reverse, Active, Active)]
fn f(x: f64) -> f64 { x * x }

#[autodiff(d_g, Reverse, Active, Active)]
fn g(x: f64) -> f64 { x + 1.0 }

fn main() {
    let x = 3.0;

    // Compute h(x) = g(f(x)) = (x²) + 1
    let (fx, df_dx) = d_f(x, 1.0);
    let (gfx, dg_df) = d_g(fx, 1.0);

    // Chain rule: dh/dx = dg/df * df/dx
    let dh_dx = dg_df * df_dx;

    println!("h({x}) = {gfx}");     // 10
    println!("h'({x}) = {dh_dx}");  // 6
}
```

## When to Use Reverse Mode

Reverse mode is optimal when:
- **Many inputs, one output**: Gradients for all inputs in one pass
- **Machine learning**: Loss functions have scalar output
- **Optimization**: Need gradients for parameter updates

Cost: O(1) backward passes regardless of input count.
