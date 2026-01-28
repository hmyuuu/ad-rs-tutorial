//! Complex Number Function Differentiation
//!
//! Demonstrates autodiff on functions involving complex numbers.
//! Complex numbers are represented as (real, imag) pairs.

#![feature(autodiff)]
use std::autodiff::autodiff_reverse;

/// Complex multiplication: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
fn complex_mul(a_re: f64, a_im: f64, b_re: f64, b_im: f64) -> (f64, f64) {
    (a_re * b_re - a_im * b_im, a_re * b_im + a_im * b_re)
}

/// Complex squared magnitude: |z|² = re² + im²
/// ∂|z|²/∂re = 2*re, ∂|z|²/∂im = 2*im
#[autodiff_reverse(d_complex_mag_sq, Active, Active, Active)]
fn complex_mag_squared(re: f64, im: f64) -> f64 {
    re * re + im * im
}

/// Complex polynomial: f(z) = z² + c (Mandelbrot iteration)
/// z² = (re + im*i)² = (re² - im²) + 2*re*im*i
/// Returns |z² + c|²
#[autodiff_reverse(d_mandelbrot_step, Active, Active, Active, Active, Active)]
fn mandelbrot_step(z_re: f64, z_im: f64, c_re: f64, c_im: f64) -> f64 {
    // z² + c
    let (sq_re, sq_im) = complex_mul(z_re, z_im, z_re, z_im);
    let new_re = sq_re + c_re;
    let new_im = sq_im + c_im;
    // Return magnitude squared
    new_re * new_re + new_im * new_im
}

/// Complex exponential approximation: exp(z) = exp(re)(cos(im) + i*sin(im))
/// Returns |exp(z)|² = exp(2*re)
#[autodiff_reverse(d_complex_exp_mag, Active, Active, Active)]
fn complex_exp_mag_squared(re: f64, _im: f64) -> f64 {
    // |exp(z)|² = |exp(re)|² * |cos(im) + i*sin(im)|²
    //           = exp(2*re) * 1 = exp(2*re)
    let mut exp_2re = 1.0;
    let x = 2.0 * re;
    let mut term = 1.0;
    let mut k = 1;
    while k < 50 {
        term *= x / (k as f64);
        exp_2re += term;
        k += 1;
    }
    exp_2re
}

/// Wirtinger derivative helper: for f(z, z*) = |z|²
/// The gradient w.r.t. (re, im) relates to Wirtinger derivatives
fn main() {
    println!("Complex Number Differentiation");
    println!("==============================\n");

    // Test 1: Complex magnitude squared
    println!("1. |z|² = re² + im²");
    let re = 3.0;
    let im = 4.0;
    let (mag_sq, d_re, d_im) = d_complex_mag_sq(re, im, 1.0);

    println!("   z = {} + {}i", re, im);
    println!("   |z|² = {} (expected: {})", mag_sq, re * re + im * im);
    println!("   ∂|z|²/∂re = {} (expected: {})", d_re, 2.0 * re);
    println!("   ∂|z|²/∂im = {} (expected: {})", d_im, 2.0 * im);
    println!();

    // Test 2: Mandelbrot iteration step
    println!("2. Mandelbrot: f(z,c) = |z² + c|²");
    let z_re = 0.5;
    let z_im = 0.5;
    let c_re = -0.5;
    let c_im = 0.5;

    let (result, dz_re, dz_im, dc_re, dc_im) = d_mandelbrot_step(z_re, z_im, c_re, c_im, 1.0);

    // Manual: z² = (0.5+0.5i)² = 0.25 - 0.25 + 0.5i = 0.5i
    // z² + c = -0.5 + i
    // |z² + c|² = 0.25 + 1 = 1.25
    println!("   z = {} + {}i, c = {} + {}i", z_re, z_im, c_re, c_im);
    println!("   |z² + c|² = {:.4} (expected: 1.25)", result);
    println!("   ∂f/∂z_re = {:.4}, ∂f/∂z_im = {:.4}", dz_re, dz_im);
    println!("   ∂f/∂c_re = {:.4}, ∂f/∂c_im = {:.4}", dc_re, dc_im);
    println!();

    // Test 3: Complex exponential magnitude
    println!("3. |exp(z)|² = exp(2·re)");
    let re = 1.0;
    let im = 2.0; // im doesn't affect magnitude

    let (exp_mag, d_re, d_im) = d_complex_exp_mag(re, im, 1.0);
    let expected = (2.0 * re).exp();
    let expected_d_re = 2.0 * expected; // d/d(re) exp(2*re) = 2*exp(2*re)

    println!("   z = {} + {}i", re, im);
    println!("   |exp(z)|² = {:.6} (expected: {:.6})", exp_mag, expected);
    println!(
        "   ∂|exp(z)|²/∂re = {:.6} (expected: {:.6})",
        d_re, expected_d_re
    );
    println!("   ∂|exp(z)|²/∂im = {:.6} (expected: 0)", d_im);
}
