//! Example 11: Activity Annotations Demo
//!
//! Demonstrates activity annotations:
//! - Active: scalar return values
//! - Const: non-differentiable parameters
//! - Duplicated: mutable references with gradient buffers
//!
//! Run with: RUSTFLAGS="-Z autodiff=Enable" cargo +nightly run -p activity_demo

#![feature(autodiff)]

use std::autodiff::autodiff_reverse;

/// Demo of Active: used for scalar returns
/// The return value is marked Active to indicate it participates in AD
#[autodiff_reverse(d_active_demo, Active, Active)]
fn active_demo(x: f64) -> f64 {
    x * x * x // x³
}

/// Demo of Const: parameter doesn't participate in differentiation
/// Here 'scale' is constant - we don't compute ∂f/∂scale
#[autodiff_reverse(d_const_demo, Active, Const, Active)]
fn const_demo(x: f64, scale: f64) -> f64 {
    scale * x * x // scale * x²
}

/// Demo of Duplicated: for array/slice parameters
/// We provide both the input and a gradient buffer
#[autodiff_reverse(d_duplicated_demo, Duplicated, Active)]
fn duplicated_demo(x: &[f64]) -> f64 {
    let mut sum = 0.0;
    let mut i = 0;
    while i < x.len() {
        sum += x[i] * x[i];
        i += 1;
    }
    sum // Σ xᵢ²
}

fn main() {
    println!("Activity Annotations Demo");
    println!("=========================\n");

    // 1. Active demo
    println!("1. Active (scalar return)");
    let x = 2.0;
    let (y, grad) = d_active_demo(x, 1.0);
    println!("   f(x) = x³, x = {x}");
    println!("   f({x}) = {y}, f'({x}) = {grad}");
    println!("   Expected: f'(x) = 3x² = {}\n", 3.0 * x * x);

    // 2. Const demo
    println!("2. Const (non-differentiable parameter)");
    let scale = 5.0;
    let (y, grad) = d_const_demo(x, scale, 1.0);
    println!("   f(x) = scale * x², x = {x}, scale = {scale}");
    println!("   f({x}) = {y}, ∂f/∂x = {grad}");
    println!("   Expected: ∂f/∂x = 2 * scale * x = {}\n", 2.0 * scale * x);

    // 3. Duplicated demo
    println!("3. Duplicated (array with gradient buffer)");
    let x_arr = [1.0, 2.0, 3.0];
    let mut grad_arr = [0.0; 3];
    let y = d_duplicated_demo(&x_arr, &mut grad_arr, 1.0);
    println!("   f(x) = Σ xᵢ², x = {:?}", x_arr);
    println!("   f(x) = {y}, ∇f = {:?}", grad_arr);
    println!("   Expected: ∇f = 2x = {:?}", x_arr.map(|xi| 2.0 * xi));
}
