//! Example 03: Multi-Variable Function
//!
//! Computes gradients of f(x, y) = x² + xy + y² using reverse-mode autodiff.
//! ∂f/∂x = 2x + y
//! ∂f/∂y = x + 2y
//!
//! Run with: RUSTFLAGS="-Z autodiff=Enable" cargo +enzyme run -p multi_variable

#![feature(autodiff)]

use std::autodiff::autodiff_reverse;

#[autodiff_reverse(d_quadratic, Active, Active, Active)]
fn quadratic(x: f64, y: f64) -> f64 {
    x * x + x * y + y * y
}

fn main() {
    let x = 2.0;
    let y = 3.0;

    // Compute primal value
    let z = quadratic(x, y);
    println!("f({x}, {y}) = {z}");

    // Compute gradients
    // The generated function returns (f(x,y), ∂f/∂x, ∂f/∂y)
    let (_, grad_x, grad_y) = d_quadratic(x, y, 1.0);
    println!("∂f/∂x = {grad_x}");
    println!("∂f/∂y = {grad_y}");

    // Verify against analytical derivatives
    let expected_grad_x = 2.0 * x + y;
    let expected_grad_y = x + 2.0 * y;
    println!("\nExpected:");
    println!("∂f/∂x = 2x + y = {expected_grad_x}");
    println!("∂f/∂y = x + 2y = {expected_grad_y}");
}

// Expected output:
// f(2, 3) = 19
// ∂f/∂x = 7
// ∂f/∂y = 8
//
// Expected:
// ∂f/∂x = 2x + y = 7
// ∂f/∂y = x + 2y = 8
