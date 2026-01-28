# Setup

Rust's autodiff feature is experimental and requires a compiler with the Enzyme backend.

## Prerequisites

You need a Rust toolchain with Enzyme support. There are two options:

### Option 1: Use Pre-built Enzyme Toolchain (Recommended)

If you have access to a pre-built Enzyme toolchain:

```bash
# Link the enzyme toolchain
rustup toolchain link enzyme /path/to/enzyme-toolchain

# Verify it works
RUSTFLAGS="-Z autodiff=Enable" cargo +enzyme build
```

### Option 2: Build from Source

See [rust-lang/rust#124509](https://github.com/rust-lang/rust/issues/124509) for instructions on building rustc with Enzyme support.

## Project Configuration

Create a `rust-toolchain.toml` in your project root:

```toml
[toolchain]
channel = "enzyme"
components = ["rustfmt", "clippy", "llvm-tools"]
```

Or specify the toolchain explicitly when running commands:

```bash
cargo +enzyme build
```

## Enable the Feature

Autodiff requires two things:

### 1. Feature Flag in Code

Add this at the top of your Rust file:

```rust
#![feature(autodiff)]

use std::autodiff::autodiff_reverse;
// or for forward mode:
use std::autodiff::autodiff_forward;
```

### 2. Compiler Flag

Enable the autodiff backend when compiling:

```bash
RUSTFLAGS="-Z autodiff=Enable" cargo +enzyme build
```

Or for running:

```bash
RUSTFLAGS="-Z autodiff=Enable" cargo +enzyme run
```

## Verify Installation

Create a simple test file:

```rust
#![feature(autodiff)]

use std::autodiff::autodiff_reverse;

#[autodiff_reverse(d_square, Active, Active)]
fn square(x: f64) -> f64 {
    x * x
}

fn main() {
    let (y, grad) = d_square(3.0, 1.0);
    println!("f(3) = {y}, f'(3) = {grad}");
}
```

Run it:

```bash
RUSTFLAGS="-Z autodiff=Enable" cargo +enzyme run
```

Expected output:
```
f(3) = 9, f'(3) = 6
```

If you see this, you're ready to go!

## Important Limitations

When writing code for Enzyme autodiff, be aware of these limitations:

### Use `while` loops instead of `for` iterators

Enzyme cannot deduce types for Rust's `Range` iterators:

```rust
// DON'T - Enzyme can't handle this
for i in 0..n {
    sum += x[i];
}

// DO - Use while loops
let mut i = 0;
while i < n {
    sum += x[i];
    i += 1;
}
```

### Avoid unsupported std functions

Some std functions like `abs()`, `sqrt()`, `sin()` may not work. Use manual implementations instead. See the [Troubleshooting](../appendix/troubleshooting.md) section for details.

## Troubleshooting

**Error: `feature(autodiff)` is not enabled`**
- Make sure you're using the enzyme toolchain: `cargo +enzyme`

**Error: `autodiff backend not found`**
- Your toolchain doesn't have Enzyme. Use a toolchain with Enzyme support.

**Error: `Cannot deduce type of extract`**
- You're using a `for` loop with Range iterator. Convert to `while` loop.

**Linker errors**
- Ensure `llvm-tools` component is installed: `rustup component add llvm-tools`
