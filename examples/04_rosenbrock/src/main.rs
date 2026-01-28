//! Example 04: Rosenbrock Function Optimization
//!
//! The Rosenbrock function is a classic test for optimization algorithms:
//! f(x, y) = (a - x)² + b(y - x²)²
//!
//! With a=1, b=100, the minimum is at (1, 1) where f(1, 1) = 0.
//!
//! This example demonstrates gradient descent using autodiff.
//!
//! Run with: RUSTFLAGS="-Z autodiff=Enable" cargo +nightly run -p rosenbrock

#![feature(autodiff)]

use std::autodiff::autodiff_reverse;

const A: f64 = 1.0;
const B: f64 = 100.0;

#[autodiff_reverse(d_rosenbrock, Active, Active, Active)]
fn rosenbrock(x: f64, y: f64) -> f64 {
    let term1 = A - x;
    let term2 = y - x * x;
    term1 * term1 + B * term2 * term2
}

fn main() {
    // Starting point
    let mut x = -1.0;
    let mut y = 1.0;
    let learning_rate = 0.001;
    let iterations = 10000;

    println!("Rosenbrock Function Optimization");
    println!("================================");
    println!("Initial point: ({x:.4}, {y:.4})");
    println!("Initial value: f(x, y) = {:.6}", rosenbrock(x, y));
    println!();

    // Gradient descent loop
    for i in 0..iterations {
        let (f_val, grad_x, grad_y) = d_rosenbrock(x, y, 1.0);

        // Update parameters
        x -= learning_rate * grad_x;
        y -= learning_rate * grad_y;

        // Print progress every 2000 iterations
        if i % 2000 == 0 || i == iterations - 1 {
            println!("Iteration {i:5}: f = {f_val:.6}, x = {x:.4}, y = {y:.4}");
        }
    }

    println!();
    println!("Final point: ({x:.4}, {y:.4})");
    println!("Final value: f(x, y) = {:.6}", rosenbrock(x, y));
    println!("Expected minimum: (1.0000, 1.0000) with f = 0.0");
}
