//! Example 06: Vector L2 Norm Gradient
//!
//! Computes the gradient of the L2 norm: f(x) = ||x||₂ = √(Σ xᵢ²)
//! The gradient is: ∂f/∂xᵢ = xᵢ / ||x||₂
//!
//! Run with: RUSTFLAGS="-Z autodiff=Enable" cargo +enzyme run -p vector_norm

#![feature(autodiff)]

use std::autodiff::autodiff_reverse;

/// Computes L2 norm: f(x) = √(Σ xᵢ²)
/// Using manual sqrt implementation for Enzyme compatibility
#[autodiff_reverse(d_l2_norm, Duplicated, Active)]
fn l2_norm(x: &[f64]) -> f64 {
    let mut sum = 0.0;
    let mut i = 0;
    while i < x.len() {
        sum += x[i] * x[i];
        i += 1;
    }
    // Newton-Raphson sqrt approximation
    let mut guess = sum / 2.0;
    if guess == 0.0 {
        return 0.0;
    }
    let mut j = 0;
    while j < 10 {
        guess = (guess + sum / guess) / 2.0;
        j += 1;
    }
    guess
}

fn main() {
    let x = [3.0, 4.0];

    // Compute primal value: ||[3, 4]||₂ = 5
    let norm = l2_norm(&x);
    println!("x = {:?}", x);
    println!("||x||₂ = {norm}");
    println!();

    // Compute gradient
    let mut dx = [0.0; 2];
    let _ = d_l2_norm(&x, &mut dx, 1.0);

    println!("Gradient ∂||x||₂/∂x = {:?}", dx);

    // Expected: [3/5, 4/5] = [0.6, 0.8]
    let expected: Vec<f64> = x.iter().map(|xi| xi / norm).collect();
    println!("Expected: {:?}", expected);
}

// Expected output:
// x = [3.0, 4.0]
// ||x||₂ = 5
//
// Gradient ∂||x||₂/∂x = [0.6, 0.8]
// Expected: [0.6, 0.8]
