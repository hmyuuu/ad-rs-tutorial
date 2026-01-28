//! Example 01: Scalar Square Function
//!
//! Computes the derivative of f(x) = x² using reverse-mode autodiff.
//! The derivative is f'(x) = 2x.
//!
//! Run with: RUSTFLAGS="-Z autodiff=Enable" cargo +enzyme run -p scalar_square

#![feature(autodiff)]

use std::autodiff::autodiff_reverse;

// Define the primal function: f(x) = x²
#[autodiff_reverse(d_square, Active, Active)]
fn square(x: f64) -> f64 {
    x * x
}

fn main() {
    let x = 3.0;

    // Compute primal value
    let y = square(x);
    println!("f({x}) = {y}");

    // Compute derivative using generated function
    // d_square(x, seed) returns (f(x), df/dx * seed)
    // With seed = 1.0, we get the gradient directly
    let (_y, grad) = d_square(x, 1.0);
    println!("f'({x}) = {grad}");
    println!("Expected: f'({x}) = {}", 2.0 * x);
}

// Expected output:
// f(3) = 9
// f'(3) = 6
// Expected: f'(3) = 6
