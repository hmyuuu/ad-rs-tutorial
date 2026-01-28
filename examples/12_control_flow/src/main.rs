//! Example 12: Control Flow Differentiation
//!
//! Demonstrates that autodiff correctly handles:
//! - if/else branches
//! - loops (for, while)
//! - early returns
//!
//! The key insight: AD differentiates the actual executed path,
//! not all possible paths.
//!
//! Run with: RUSTFLAGS="-Z autodiff=Enable" cargo +nightly run -p control_flow

#![feature(autodiff)]

use std::autodiff::autodiff_reverse;

/// Function with branching based on input value
/// f(x) = x² if x >= 0
/// f(x) = -x² if x < 0
///
/// Derivatives:
/// f'(x) = 2x if x >= 0
/// f'(x) = -2x if x < 0
#[autodiff_reverse(d_branching, Active, Active)]
fn branching(x: f64) -> f64 {
    if x >= 0.0 {
        x * x
    } else {
        -(x * x)
    }
}

/// Function with a loop: computes x^n using repeated multiplication
/// f(x) = x^n
/// f'(x) = n * x^(n-1)
#[autodiff_reverse(d_power_loop, Active, Const, Active)]
fn power_loop(x: f64, n: usize) -> f64 {
    let mut result = 1.0;
    let mut i = 0;
    while i < n {
        result *= x;
        i += 1;
    }
    result
}

/// Function with while loop: computes exp(x) via Taylor series
/// f(x) ≈ Σ x^k / k! (truncated)
/// f'(x) ≈ exp(x)
#[autodiff_reverse(d_exp_approx, Active, Active)]
fn exp_approx(x: f64) -> f64 {
    let mut sum: f64 = 1.0;
    let mut term: f64 = 1.0;
    let mut k = 1;

    while (if term < 0.0 { -term } else { term }) > 1e-10 && k < 100 {
        term *= x / (k as f64);
        sum += term;
        k += 1;
    }
    sum
}

/// ReLU activation: max(0, x)
/// f'(x) = 1 if x > 0, else 0
#[autodiff_reverse(d_relu, Active, Active)]
fn relu(x: f64) -> f64 {
    if x > 0.0 { x } else { 0.0 }
}

fn main() {
    println!("Control Flow Differentiation Demo");
    println!("==================================\n");

    // 1. Branching
    println!("1. Branching (if/else)");
    for x in [-2.0, 0.0, 3.0] {
        let (y, grad) = d_branching(x, 1.0);
        let expected = if x >= 0.0 { 2.0 * x } else { -2.0 * x };
        println!("   f({x:+}) = {y:+}, f'({x:+}) = {grad:+} (expected: {expected:+})");
    }
    println!();

    // 2. Loop (power function)
    println!("2. Loop (power function)");
    let x = 2.0;
    for n in [2, 3, 4] {
        let (y, grad) = d_power_loop(x, n, 1.0);
        let expected = (n as f64) * x.powi(n as i32 - 1);
        println!("   x^{n} at x={x}: value={y}, gradient={grad} (expected: {expected})");
    }
    println!();

    // 3. While loop (exp approximation)
    println!("3. While loop (exp approximation)");
    let x = 1.0;
    let (y, grad) = d_exp_approx(x, 1.0);
    println!("   exp({x}) ≈ {y:.6}");
    println!("   d/dx exp({x}) ≈ {grad:.6}");
    println!("   Actual exp({x}) = {:.6}", x.exp());
    println!();

    // 4. ReLU (piecewise linear)
    println!("4. ReLU activation");
    for x in [-1.0, 0.0, 1.0] {
        let (y, grad) = d_relu(x, 1.0);
        println!("   ReLU({x:+}) = {y}, ReLU'({x:+}) = {grad}");
    }
}
