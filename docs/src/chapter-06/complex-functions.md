# Complex Number Functions

This example demonstrates autodiff on functions involving complex numbers, represented as (real, imaginary) pairs.

## Complex Number Representation

Since Rust's autodiff doesn't directly support `num-complex`, we represent complex numbers as separate real and imaginary components:

```rust
// z = 3 + 4i represented as:
let re = 3.0;
let im = 4.0;
```

## Example 1: Complex Magnitude Squared

```rust
/// |z|² = re² + im²
/// ∂|z|²/∂re = 2·re, ∂|z|²/∂im = 2·im
#[autodiff_reverse(d_complex_mag_sq, Active, Active, Active)]
fn complex_mag_squared(re: f64, im: f64) -> f64 {
    re * re + im * im
}
```

## Example 2: Mandelbrot Iteration

The Mandelbrot set uses the iteration z → z² + c:

```rust
/// Returns |z² + c|²
#[autodiff_reverse(d_mandelbrot_step, Active, Active, Active, Active, Active)]
fn mandelbrot_step(z_re: f64, z_im: f64, c_re: f64, c_im: f64) -> f64 {
    let (sq_re, sq_im) = complex_mul(z_re, z_im, z_re, z_im);
    let new_re = sq_re + c_re;
    let new_im = sq_im + c_im;
    new_re * new_re + new_im * new_im
}
```

## Example 3: Complex Exponential

For exp(z) = exp(re)·(cos(im) + i·sin(im)), the magnitude is:

```rust
/// |exp(z)|² = exp(2·re)
#[autodiff_reverse(d_complex_exp_mag, Active, Active, Active)]
fn complex_exp_mag_squared(re: f64, _im: f64) -> f64 {
    // Taylor series for exp(2·re)
    my_exp(2.0 * re)
}
```

## Key Insights

1. **Wirtinger Derivatives**: For complex functions, gradients w.r.t. (re, im) relate to Wirtinger derivatives ∂f/∂z and ∂f/∂z*
2. **Real-valued Loss**: AD computes gradients of real-valued functions, so we typically differentiate |f(z)|² or Re(f(z))
3. **Component-wise**: Treating re and im as separate variables works naturally with reverse-mode AD

Run the example:
```bash
RUSTFLAGS="-Z autodiff=Enable" cargo +enzyme run -p complex_function
```
