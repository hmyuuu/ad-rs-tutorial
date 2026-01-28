//! Example 10: Forward Mode Autodiff
//!
//! Demonstrates forward-mode automatic differentiation.
//! Forward mode computes directional derivatives: ∂f/∂x * dx
//!
//! Forward mode is efficient when:
//! - You have few inputs and many outputs
//! - You need directional derivatives
//! - You're computing Jacobian-vector products
//!
//! Run with: RUSTFLAGS="-Z autodiff=Enable" cargo +enzyme run -p forward_mode

#![feature(autodiff)]

use std::autodiff::autodiff_forward;

/// Simple function for forward mode demo
/// f(x) = x³ + 2x
/// f'(x) = 3x² + 2
#[autodiff_forward(d_cubic, Dual, Dual)]
fn cubic(x: f64) -> f64 {
    x * x * x + 2.0 * x
}

/// Multi-output function (good use case for forward mode)
/// Returns (x², x³)
#[autodiff_forward(d_multi_out, Dual, Dual)]
fn multi_output(x: f64) -> (f64, f64) {
    (x * x, x * x * x)
}

fn main() {
    let x = 2.0;

    println!("Forward Mode Autodiff Demo");
    println!("==========================\n");

    // Example 1: Simple cubic function
    println!("Example 1: f(x) = x³ + 2x");
    let y = cubic(x);
    println!("f({x}) = {y}");

    // Forward mode: provide tangent (dx), get output tangent (dy)
    // dy = (df/dx) * dx, so with dx=1, we get df/dx directly
    let (_y, dy) = d_cubic(x, 1.0);
    println!("f'({x}) = {dy}");
    println!("Expected: 3x² + 2 = {}\n", 3.0 * x * x + 2.0);

    // Example 2: Multi-output function
    println!("Example 2: f(x) = (x², x³)");
    let (y1, y2) = multi_output(x);
    println!("f({x}) = ({y1}, {y2})");

    // Forward mode with tangent dx=1 gives both derivatives simultaneously
    let ((dy1, dy2), _) = d_multi_out(x, 1.0);
    println!("df/dx = ({dy1}, {dy2})");
    println!("Expected: (2x, 3x²) = ({}, {})\n", 2.0 * x, 3.0 * x * x);

    // Example 3: Directional derivative
    println!("Example 3: Directional derivative");
    // With dx=0.5, we compute (df/dx) * 0.5
    let (_, dy_half) = d_cubic(x, 0.5);
    println!("(df/dx) * 0.5 = {dy_half}");
    println!("Expected: {} * 0.5 = {}", 3.0 * x * x + 2.0, (3.0 * x * x + 2.0) * 0.5);
}
