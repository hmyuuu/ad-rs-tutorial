# Scalar Functions

Let's start with simple scalar functions to build intuition.

## Example 1: Square Function

The simplest non-trivial derivative: f(x) = x²

```rust
#![feature(autodiff)]
use std::autodiff::autodiff;

#[autodiff(d_square, Reverse, Active, Active)]
fn square(x: f64) -> f64 {
    x * x
}

fn main() {
    let x = 3.0;
    let (y, grad) = d_square(x, 1.0);

    println!("f({x}) = {y}");        // 9
    println!("f'({x}) = {grad}");    // 6
}
```

Run: `RUSTFLAGS="-Z autodiff=Enable" cargo run -p scalar_square`

## Example 2: Sine Function

Trigonometric functions work seamlessly:

```rust
#[autodiff(d_sin, Reverse, Active, Active)]
fn my_sin(x: f64) -> f64 {
    x.sin()
}

fn main() {
    let x = std::f64::consts::PI / 4.0;  // 45°

    let (y, grad) = d_sin(x, 1.0);

    println!("sin({x:.4}) = {y:.6}");     // 0.707107
    println!("cos({x:.4}) = {grad:.6}");  // 0.707107
}
```

The derivative of sin is cos — autodiff computes this automatically!

Run: `RUSTFLAGS="-Z autodiff=Enable" cargo run -p scalar_sin`

## Example 3: Multi-Variable Quadratic

Functions of multiple variables:

```rust
#[autodiff(d_quad, Reverse, Active, Active, Active)]
fn quadratic(x: f64, y: f64) -> f64 {
    x * x + x * y + y * y
}

fn main() {
    let (z, dx, dy) = d_quad(2.0, 3.0, 1.0);

    println!("f(2, 3) = {z}");    // 19
    println!("∂f/∂x = {dx}");     // 7  (2x + y)
    println!("∂f/∂y = {dy}");     // 8  (x + 2y)
}
```

Run: `RUSTFLAGS="-Z autodiff=Enable" cargo run -p multi_variable`

## Composition of Functions

Autodiff handles function composition through the chain rule:

```rust
#[autodiff(d_composed, Reverse, Active, Active)]
fn composed(x: f64) -> f64 {
    (x * x).sin()  // sin(x²)
}

fn main() {
    let x = 1.0;
    let (y, grad) = d_composed(x, 1.0);

    // d/dx sin(x²) = cos(x²) × 2x
    println!("f({x}) = {y}");
    println!("f'({x}) = {grad}");
}
```

## Key Takeaways

- Standard math functions (sin, cos, exp, log) are supported
- Multiple inputs produce multiple gradients
- Function composition works automatically via chain rule
- The generated function signature depends on activity annotations
