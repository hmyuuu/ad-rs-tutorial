//! Example 08: Binary Cross-Entropy Loss
//!
//! Computes the binary cross-entropy loss and its gradient:
//! L = -(1/n) Σ [yᵢ * log(pᵢ) + (1 - yᵢ) * log(1 - pᵢ)]
//!
//! The gradient with respect to predictions:
//! ∂L/∂pᵢ = -(1/n) * [yᵢ/pᵢ - (1 - yᵢ)/(1 - pᵢ)]
//!
//! Run with: RUSTFLAGS="-Z autodiff=Enable" cargo +enzyme run -p cross_entropy

#![feature(autodiff)]

use std::autodiff::autodiff_reverse;

/// Natural log approximation using Taylor series around 1
/// ln(x) for x near 1: ln(1 + u) ≈ u - u²/2 + u³/3 - ...
fn my_ln(x: f64) -> f64 {
    // For x in (0, 2), use ln(x) = ln(1 + (x-1))
    let u = x - 1.0;
    let mut sum = 0.0;
    let mut term = u;
    let mut k = 1;
    while k < 20 {
        sum += term / k as f64;
        term *= -u;
        k += 1;
    }
    sum
}

/// Binary cross-entropy loss
/// pred: predicted probabilities (we want gradients)
/// target: ground truth labels 0 or 1 (constant)
#[autodiff_reverse(d_bce_loss, Duplicated, Const, Active)]
fn bce_loss(pred: &[f64], target: &[f64]) -> f64 {
    let n = pred.len() as f64;
    let eps = 1e-15; // For numerical stability

    let mut sum = 0.0;
    let mut i = 0;
    while i < pred.len() {
        let p = pred[i];
        let t = target[i];
        // Clamp p to avoid log(0)
        let p_clamped = if p < eps {
            eps
        } else if p > 1.0 - eps {
            1.0 - eps
        } else {
            p
        };
        sum += -(t * my_ln(p_clamped) + (1.0 - t) * my_ln(1.0 - p_clamped));
        i += 1;
    }
    sum / n
}

fn main() {
    // Predictions (probabilities) and true labels
    let predictions = [0.9, 0.2, 0.8, 0.3];
    let targets = [1.0, 0.0, 1.0, 0.0];

    // Compute loss
    let loss = bce_loss(&predictions, &targets);
    println!("Predictions: {:?}", predictions);
    println!("Targets:     {:?}", targets);
    println!("BCE Loss: {loss:.6}");
    println!();

    // Compute gradients
    let mut grad_pred = [0.0; 4];
    let _ = d_bce_loss(&predictions, &mut grad_pred, &targets, 1.0);

    println!("Gradient ∂L/∂pred:");
    for (i, g) in grad_pred.iter().enumerate() {
        println!("  pred[{i}]: {g:.6}");
    }

    // Interpretation: negative gradients push predictions toward correct values
    println!("\nInterpretation:");
    println!("  pred[0]=0.9, target=1: grad < 0 → push higher (good prediction)");
    println!("  pred[1]=0.2, target=0: grad > 0 → push lower (good prediction)");
}
