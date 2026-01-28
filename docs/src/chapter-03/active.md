# Active

The `Active` annotation is used for scalar values that participate in differentiation.

## When to Use

Use `Active` for:
- Scalar input parameters (`f64`, `f32`)
- Scalar return values
- Any value where you want to compute ∂output/∂input

## Syntax

```rust
#[autodiff(d_func, Reverse, Active, Active)]
fn func(x: f64) -> f64 { ... }
//            ↑         ↑
//         input     return
```

## Example

```rust
#![feature(autodiff)]
use std::autodiff::autodiff;

#[autodiff(d_cubic, Reverse, Active, Active)]
fn cubic(x: f64) -> f64 {
    x * x * x
}

fn main() {
    let x = 2.0;
    let (y, grad) = d_cubic(x, 1.0);

    println!("f({x}) = {y}");      // f(2) = 8
    println!("f'({x}) = {grad}");  // f'(2) = 12
}
```

## Generated Function Signature

For reverse mode with `Active` annotations:

```rust
// Original
fn func(x: f64) -> f64

// Generated
fn d_func(x: f64, seed: f64) -> (f64, f64)
//                ↑              ↑     ↑
//           output adjoint    value  gradient
```

The generated function:
1. Takes the original input
2. Takes an adjoint seed for the output
3. Returns both the primal value and the gradient

## Multiple Active Inputs

With multiple `Active` inputs, you get multiple gradients:

```rust
#[autodiff(d_sum, Reverse, Active, Active, Active)]
fn sum(x: f64, y: f64) -> f64 {
    x + y
}

// Generated: fn d_sum(x: f64, y: f64, seed: f64) -> (f64, f64, f64)
//                                                    ↑     ↑     ↑
//                                                  value  dx    dy
```

## Key Points

- `Active` is for scalars only (use `Duplicated` for arrays)
- The seed parameter controls the output adjoint
- Use seed = 1.0 to get the gradient directly
- Each `Active` input adds one gradient to the return tuple
