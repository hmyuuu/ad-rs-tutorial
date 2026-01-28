//! Example 05: Vector Dot Product Gradient
//!
//! Computes the gradient of f(x) = x · w (dot product) with respect to x.
//! The gradient ∂f/∂x = w (the weight vector).
//!
//! This example demonstrates the Duplicated annotation for slice inputs.
//!
//! Run with: RUSTFLAGS="-Z autodiff=Enable" cargo +enzyme run -p vector_dot

#![feature(autodiff)]

use std::autodiff::autodiff_reverse;

/// Computes dot product: f(x) = x · w = Σ xᵢ * wᵢ
/// Using Duplicated for x means we provide a gradient buffer (dx) alongside x.
/// Using Const for w means we don't compute gradients with respect to w.
#[autodiff_reverse(d_dot_product, Duplicated, Const, Active)]
fn dot_product(x: &[f64], w: &[f64]) -> f64 {
    let mut sum = 0.0;
    let n = x.len();
    let mut i = 0;
    while i < n {
        sum += x[i] * w[i];
        i += 1;
    }
    sum
}

fn main() {
    let x = [1.0, 2.0, 3.0];
    let w = [0.5, 1.5, 2.5];

    // Compute primal value
    let result = dot_product(&x, &w);
    println!("x = {:?}", x);
    println!("w = {:?}", w);
    println!("x · w = {result}");
    println!();

    // Compute gradient with respect to x
    // dx will be filled with ∂f/∂xᵢ for each i
    let mut dx = [0.0; 3];
    let _ = d_dot_product(&x, &mut dx, &w, 1.0);

    println!("Gradient ∂f/∂x = {:?}", dx);
    println!("Expected: {:?} (= w)", w);
}

// Expected output:
// x = [1.0, 2.0, 3.0]
// w = [0.5, 1.5, 2.5]
// x · w = 11
//
// Gradient ∂f/∂x = [0.5, 1.5, 2.5]
// Expected: [0.5, 1.5, 2.5] (= w)
