# Your First Derivative

Let's compute the derivative of the simplest non-trivial function: f(x) = x².

## The Code

```rust
#![feature(autodiff)]

use std::autodiff::autodiff;

#[autodiff(d_square, Reverse, Active, Active)]
fn square(x: f64) -> f64 {
    x * x
}

fn main() {
    let x = 3.0;

    // Compute derivative
    let (y, grad) = d_square(x, 1.0);

    println!("f({x}) = {y}");
    println!("f'({x}) = {grad}");
}
```

Output:
```
f(3) = 9
f'(3) = 6
```

## Understanding the Attribute

```rust
#[autodiff(d_square, Reverse, Active, Active)]
```

Let's break this down:

| Part | Meaning |
|------|---------|
| `d_square` | Name of the generated derivative function |
| `Reverse` | Use reverse-mode AD |
| `Active` (first) | The input `x` participates in differentiation |
| `Active` (second) | The return value participates in differentiation |

## The Generated Function

The `#[autodiff]` attribute generates a new function `d_square` with this signature:

```rust
fn d_square(x: f64, seed: f64) -> (f64, f64)
//          ↑       ↑              ↑     ↑
//          input   output seed    f(x)  df/dx
```

- **`x`**: The input value (same as original)
- **`seed`**: The "adjoint seed" — typically 1.0 for gradients
- **Returns**: A tuple of (function value, gradient)

## Why the Seed?

The seed represents ∂L/∂y where L is some downstream loss. When computing gradients directly, we use seed = 1.0.

For chain rule composition:
```
∂L/∂x = (∂L/∂y) × (∂y/∂x) = seed × gradient
```

With seed = 1.0: ∂L/∂x = 1.0 × 6.0 = 6.0 ✓

## Verifying the Result

We know from calculus:
- f(x) = x²
- f'(x) = 2x
- f'(3) = 2 × 3 = 6 ✓

The autodiff result matches exactly!

## Try It Yourself

Run the example:

```bash
cd examples/01_scalar_square
RUSTFLAGS="-Z autodiff=Enable" cargo run
```

Try modifying the input value and verify the gradient is always 2x.

## Next Steps

Now that you've computed your first derivative, let's explore more complex functions and learn about the different activity annotations.
