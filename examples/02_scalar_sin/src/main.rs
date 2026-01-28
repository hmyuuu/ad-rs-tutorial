//! Example 02: Scalar Sine Function (Taylor Series)
//!
//! Computes the derivative of f(x) = sin(x) using reverse-mode autodiff.
//! Uses Taylor series approximation since std::sin doesn't have Enzyme support yet.
//! The derivative is f'(x) = cos(x).
//!
//! Run with: RUSTFLAGS="-Z autodiff=Enable" cargo +enzyme run -p scalar_sin

#![feature(autodiff)]

use std::autodiff::autodiff_reverse;

/// Taylor series approximation of sin(x)
/// sin(x) ≈ x - x³/3! + x⁵/5! - x⁷/7! + ...
#[autodiff_reverse(d_sin, Active, Active)]
fn my_sin(x: f64) -> f64 {
    let mut sum = 0.0;
    let mut term = x;
    let mut n = 1i32;
    let mut i = 0;

    while i < 10 {
        sum += term;
        term *= -x * x / ((n + 1) * (n + 2)) as f64;
        n += 2;
        i += 1;
    }
    sum
}

fn main() {
    let x = std::f64::consts::PI / 4.0; // 45 degrees

    // Compute primal value: sin(π/4) = √2/2 ≈ 0.707
    let y = my_sin(x);
    println!("sin({:.4}) = {:.6}", x, y);

    // Compute derivative: cos(π/4) = √2/2 ≈ 0.707
    let (_, grad) = d_sin(x, 1.0);
    println!("d/dx sin({:.4}) = {:.6}", x, grad);
    println!("cos({:.4}) = {:.6} (expected)", x, x.cos());
}

// Expected output:
// sin(0.7854) = 0.707107
// d/dx sin(0.7854) = 0.707107
// cos(0.7854) = 0.707107 (expected)
