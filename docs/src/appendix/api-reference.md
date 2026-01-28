# API Reference

Quick reference for Rust's autodiff API.

## The autodiff Attribute

```rust
#[autodiff(name, mode, ...annotations)]
fn original_function(...) -> ... { ... }
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| `name` | Name of generated derivative function |
| `mode` | `Reverse` or `Forward` |
| `annotations` | Activity annotation for each parameter and return |

## Modes

### Reverse Mode

```rust
#[autodiff(d_f, Reverse, Active, Active)]
fn f(x: f64) -> f64 { ... }

// Generated: fn d_f(x: f64, seed: f64) -> (f64, f64)
```

- Computes gradients from output to inputs
- Efficient for many inputs, few outputs
- Used in machine learning (backpropagation)

### Forward Mode

```rust
#[autodiff(d_f, Forward, Dual, Dual)]
fn f(x: f64) -> f64 { ... }

// Generated: fn d_f(x: f64, dx: f64) -> (f64, f64)
```

- Computes derivatives from inputs to outputs
- Efficient for few inputs, many outputs
- Used for Jacobian-vector products

## Activity Annotations

### Active (Reverse Mode)

For scalar values participating in differentiation.

```rust
#[autodiff(d_f, Reverse, Active, Active)]
fn f(x: f64) -> f64 { x * x }

// d_f(x, seed) -> (value, gradient)
```

### Dual (Forward Mode)

For scalar values in forward mode.

```rust
#[autodiff(d_f, Forward, Dual, Dual)]
fn f(x: f64) -> f64 { x * x }

// d_f(x, tangent) -> (value, output_tangent)
```

### Const

For parameters that don't need gradients.

```rust
#[autodiff(d_f, Reverse, Active, Const, Active)]
fn f(x: f64, config: f64) -> f64 { config * x }

// d_f(x, config, seed) -> (value, grad_x)
// No gradient for config
```

### Duplicated

For reference types (slices) with gradient buffers.

```rust
#[autodiff(d_f, Reverse, Duplicated, Active)]
fn f(x: &[f64]) -> f64 { x.iter().sum() }

// d_f(x, grad_x, seed) -> value
// grad_x is filled with gradients
```

### DuplicatedNoNeed

Like Duplicated, but doesn't return primal value.

```rust
#[autodiff(d_f, Reverse, DuplicatedNoNeed, Active)]
fn f(x: &[f64]) -> f64 { x.iter().sum() }

// d_f(x, grad_x, seed) -> ()
// Only computes gradients, not primal
```

## Generated Function Signatures

### Reverse Mode Examples

```rust
// Single Active input
#[autodiff(d_f, Reverse, Active, Active)]
fn f(x: f64) -> f64
// → fn d_f(x: f64, seed: f64) -> (f64, f64)

// Two Active inputs
#[autodiff(d_f, Reverse, Active, Active, Active)]
fn f(x: f64, y: f64) -> f64
// → fn d_f(x: f64, y: f64, seed: f64) -> (f64, f64, f64)

// Duplicated input
#[autodiff(d_f, Reverse, Duplicated, Active)]
fn f(x: &[f64]) -> f64
// → fn d_f(x: &[f64], dx: &mut [f64], seed: f64) -> f64

// Mixed
#[autodiff(d_f, Reverse, Duplicated, Const, Active)]
fn f(x: &[f64], c: f64) -> f64
// → fn d_f(x: &[f64], dx: &mut [f64], c: f64, seed: f64) -> f64
```

### Forward Mode Examples

```rust
// Single Dual input
#[autodiff(d_f, Forward, Dual, Dual)]
fn f(x: f64) -> f64
// → fn d_f(x: f64, dx: f64) -> (f64, f64)
```

## Required Setup

```rust
// In your Rust file
#![feature(autodiff)]
use std::autodiff::autodiff;
```

```bash
# When compiling
RUSTFLAGS="-Z autodiff=Enable" cargo build
```

```toml
# rust-toolchain.toml
[toolchain]
channel = "nightly"
```
