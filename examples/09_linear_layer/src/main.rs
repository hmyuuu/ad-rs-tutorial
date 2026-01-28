//! Example 09: Linear Layer Gradient
//!
//! Implements a simple linear layer: y = Wx + b
//! and computes gradients with respect to weights W and bias b.
//!
//! For simplicity, we use a 2D input and 2D output.
//!
//! Run with: RUSTFLAGS="-Z autodiff=Enable" cargo +nightly run -p linear_layer

#![feature(autodiff)]

use std::autodiff::autodiff_reverse;

/// Linear layer forward pass with MSE loss
/// Computes: loss = ||Wx + b - target||²
///
/// Parameters:
/// - x: input vector (constant)
/// - weights: 2x2 weight matrix as flat array (we want gradients)
/// - bias: bias vector (we want gradients)
/// - target: target output (constant)
#[autodiff_reverse(d_linear_loss, Const, Duplicated, Duplicated, Const, Active)]
fn linear_loss(x: &[f64], weights: &[f64], bias: &[f64], target: &[f64]) -> f64 {
    // y = Wx + b (2x2 matrix × 2-vector + 2-vector)
    let y0 = weights[0] * x[0] + weights[1] * x[1] + bias[0];
    let y1 = weights[2] * x[0] + weights[3] * x[1] + bias[1];

    // MSE loss
    let diff0 = y0 - target[0];
    let diff1 = y1 - target[1];
    (diff0 * diff0 + diff1 * diff1) / 2.0
}

fn main() {
    // Input
    let x = [1.0, 2.0];

    // Weights (2x2 matrix, row-major)
    let weights = [0.5, 0.5, 0.5, 0.5];

    // Bias
    let bias = [0.1, 0.1];

    // Target
    let target = [1.0, 2.0];

    // Forward pass
    let loss = linear_loss(&x, &weights, &bias, &target);
    println!("Input x: {:?}", x);
    println!("Weights W: {:?} (2x2)", weights);
    println!("Bias b: {:?}", bias);
    println!("Target: {:?}", target);
    println!("Loss: {loss}");
    println!();

    // Compute gradients
    let mut grad_weights = [0.0; 4];
    let mut grad_bias = [0.0; 2];
    let _ = d_linear_loss(
        &x,
        &weights,
        &mut grad_weights,
        &bias,
        &mut grad_bias,
        &target,
        1.0,
    );

    println!("Gradient ∂L/∂W: {:?}", grad_weights);
    println!("Gradient ∂L/∂b: {:?}", grad_bias);
    println!();

    // Simple gradient descent step
    let lr = 0.1;
    let new_weights: Vec<f64> = weights
        .iter()
        .zip(grad_weights.iter())
        .map(|(w, g)| w - lr * g)
        .collect();
    let new_bias: Vec<f64> = bias
        .iter()
        .zip(grad_bias.iter())
        .map(|(b, g)| b - lr * g)
        .collect();

    println!("After one gradient step (lr={lr}):");
    println!("New weights: {:?}", new_weights);
    println!("New bias: {:?}", new_bias);

    // Verify loss decreased
    let new_loss = linear_loss(&x, &new_weights, &new_bias, &target);
    println!("New loss: {new_loss}");
    println!("Loss decreased: {}", new_loss < loss);
}
