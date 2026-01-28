# Basic Usage

Forward mode propagates derivatives from inputs to outputs, computing directional derivatives.

## The Forward Mode Attribute

```rust
#[autodiff(derivative_name, Forward, Dual, Dual)]
fn original_function(x: f64) -> f64 { ... }
```

Note: Forward mode uses `Dual` instead of `Active`.

## Simple Example

```rust
#![feature(autodiff)]
use std::autodiff::autodiff;

#[autodiff(d_cube, Forward, Dual, Dual)]
fn cube(x: f64) -> f64 {
    x * x * x
}

fn main() {
    let x = 2.0;

    // Forward mode: provide tangent dx, get tangent dy
    let (y, dy) = d_cube(x, 1.0);

    println!("f({x}) = {y}");      // 8
    println!("f'({x}) = {dy}");    // 12
}
```

## How Forward Mode Works

Forward mode computes:
- Primal: y = f(x)
- Tangent: dy = (df/dx) × dx

With dx = 1.0, we get df/dx directly.

```
x=2, dx=1 → cube → y=8, dy=12
```

## The Dual Number Interpretation

Forward mode can be understood through dual numbers: x + εẋ

- ε² = 0 (infinitesimal)
- (x + εẋ)² = x² + 2xẋε

For f(x) = x³:
```
(x + εẋ)³ = x³ + 3x²ẋε
```

The coefficient of ε is the derivative!

## Generated Function Signature

```rust
// Original
fn f(x: f64) -> f64

// Generated (Forward mode)
fn d_f(x: f64, dx: f64) -> (f64, f64)
//            ↑              ↑     ↑
//         tangent        value  tangent
```

Compare to reverse mode:
```rust
// Generated (Reverse mode)
fn d_f(x: f64, seed: f64) -> (f64, f64)
//            ↑                    ↑
//         adjoint              gradient
```

## Directional Derivatives

Forward mode naturally computes directional derivatives:

```rust
let direction = 0.5;  // dx
let (y, dy) = d_f(x, direction);
// dy = (df/dx) * direction
```

This is useful for:
- Computing derivatives in specific directions
- Jacobian-vector products (JVP)
- Sensitivity analysis

## Multi-Output Functions

Forward mode shines with multiple outputs:

```rust
#[autodiff(d_sincos, Forward, Dual, Dual)]
fn sincos(x: f64) -> (f64, f64) {
    (x.sin(), x.cos())
}

fn main() {
    let x = std::f64::consts::PI / 4.0;
    let ((sin_x, cos_x), (d_sin, d_cos)) = d_sincos(x, 1.0);

    // Both derivatives computed in one pass!
    println!("d(sin)/dx = {d_sin}");   // cos(x)
    println!("d(cos)/dx = {d_cos}");   // -sin(x)
}
```

## Key Points

- Forward mode uses `Dual` annotation
- Provide input tangent (dx), get output tangent (dy)
- dy = (df/dx) × dx
- With dx = 1.0, get the derivative directly
- Efficient for few inputs, many outputs
