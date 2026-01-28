# Control Flow

One of AD's strengths over symbolic differentiation is handling control flow. Autodiff differentiates the **executed path**, not all possible paths.

## If/Else Branches

```rust
#![feature(autodiff)]
use std::autodiff::autodiff;

#[autodiff(d_abs_square, Reverse, Active, Active)]
fn abs_square(x: f64) -> f64 {
    if x >= 0.0 {
        x * x
    } else {
        -(x * x)
    }
}

fn main() {
    // Positive branch: f(x) = x², f'(x) = 2x
    let (y, grad) = d_abs_square(3.0, 1.0);
    println!("f(3) = {y}, f'(3) = {grad}");  // 9, 6

    // Negative branch: f(x) = -x², f'(x) = -2x
    let (y, grad) = d_abs_square(-2.0, 1.0);
    println!("f(-2) = {y}, f'(-2) = {grad}");  // -4, 4
}
```

The derivative depends on which branch executes.

## ReLU Activation

ReLU is a classic example of piecewise differentiation:

```rust
#[autodiff(d_relu, Reverse, Active, Active)]
fn relu(x: f64) -> f64 {
    if x > 0.0 { x } else { 0.0 }
}

fn main() {
    println!("ReLU'(1) = {}", d_relu(1.0, 1.0).1);   // 1
    println!("ReLU'(-1) = {}", d_relu(-1.0, 1.0).1); // 0
    println!("ReLU'(0) = {}", d_relu(0.0, 1.0).1);   // 0
}
```

## For Loops

Loops unroll naturally:

```rust
#[autodiff(d_power, Reverse, Active, Const, Active)]
fn power(x: f64, n: usize) -> f64 {
    let mut result = 1.0;
    for _ in 0..n {
        result *= x;
    }
    result
}

fn main() {
    // x³: derivative is 3x²
    let (y, grad) = d_power(2.0, 3, 1.0);
    println!("2³ = {y}, d/dx = {grad}");  // 8, 12
}
```

## While Loops

While loops work too, as long as they terminate:

```rust
#[autodiff(d_exp_approx, Reverse, Active, Active)]
fn exp_approx(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    let mut k = 1;

    while term.abs() > 1e-10 && k < 100 {
        term *= x / (k as f64);
        sum += term;
        k += 1;
    }
    sum
}

fn main() {
    let x = 1.0;
    let (y, grad) = d_exp_approx(x, 1.0);

    println!("exp({x}) ≈ {y:.6}");
    println!("d/dx exp({x}) ≈ {grad:.6}");
    println!("Actual: {:.6}", x.exp());
}
```

## Key Insight: Path-Based Differentiation

AD differentiates the computation that **actually happens**:

```rust
fn conditional(x: f64, flag: bool) -> f64 {
    if flag {
        x * x      // Path A: derivative = 2x
    } else {
        x * x * x  // Path B: derivative = 3x²
    }
}
```

- With `flag = true`: derivative is 2x
- With `flag = false`: derivative is 3x²

This is different from symbolic differentiation, which would need to handle both branches algebraically.

## Limitations

Some patterns can cause issues:

```rust
// Problematic: derivative undefined at x = 0
fn problematic(x: f64) -> f64 {
    if x == 0.0 {
        0.0
    } else {
        x.sin() / x
    }
}
```

At discontinuities, the derivative may not be well-defined.

Run the full example: `RUSTFLAGS="-Z autodiff=Enable" cargo run -p control_flow`
