//! Example 07: Mean Squared Error Loss
//!
//! Computes the MSE loss and its gradient:
//! L = (1/n) Σ (predᵢ - targetᵢ)²
//!
//! The gradient with respect to predictions:
//! ∂L/∂predᵢ = (2/n) * (predᵢ - targetᵢ)
//!
//! Run with: RUSTFLAGS="-Z autodiff=Enable" cargo +enzyme run -p mse_loss

#![feature(autodiff)]

use std::autodiff::autodiff_reverse;

/// Mean Squared Error loss function
/// pred: predictions (we want gradients for these)
/// target: ground truth (constant, no gradients)
#[autodiff_reverse(d_mse_loss, Duplicated, Const, Active)]
fn mse_loss(pred: &[f64], target: &[f64]) -> f64 {
    let n = pred.len() as f64;
    let mut sum = 0.0;
    let mut i = 0;
    while i < pred.len() {
        let diff = pred[i] - target[i];
        sum += diff * diff;
        i += 1;
    }
    sum / n
}

fn main() {
    let predictions = [2.5, 0.0, 2.0, 8.0];
    let targets = [3.0, -0.5, 2.0, 7.0];

    // Compute loss
    let loss = mse_loss(&predictions, &targets);
    println!("Predictions: {:?}", predictions);
    println!("Targets:     {:?}", targets);
    println!("MSE Loss: {loss}");
    println!();

    // Compute gradients
    let mut grad_pred = [0.0; 4];
    let _ = d_mse_loss(&predictions, &mut grad_pred, &targets, 1.0);

    println!("Gradient ∂L/∂pred: {:?}", grad_pred);

    // Verify against analytical gradient
    let n = predictions.len() as f64;
    let expected: Vec<f64> = predictions
        .iter()
        .zip(targets.iter())
        .map(|(p, t)| 2.0 * (p - t) / n)
        .collect();
    println!("Expected:          {:?}", expected);
}

// Expected output:
// Predictions: [2.5, 0.0, 2.0, 8.0]
// Targets:     [3.0, -0.5, 2.0, 7.0]
// MSE Loss: 0.375
//
// Gradient ∂L/∂pred: [-0.25, 0.25, 0.0, 0.5]
// Expected:          [-0.25, 0.25, 0.0, 0.5]
