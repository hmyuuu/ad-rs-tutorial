# Rust Autodiff Tutorial

Welcome to the Rust Autodiff Tutorial! This guide will teach you how to use Rust's experimental automatic differentiation feature (`std::autodiff`) to compute derivatives of functions automatically.

## Current Status

> **Note**: The autodiff feature requires the Enzyme LLVM plugin, which is not yet distributed via rustup. This tutorial documents the API and concepts so you're ready when it becomes available.
>
> Track progress: [rust-lang/rust#124509](https://github.com/rust-lang/rust/issues/124509)

## What You'll Learn

- **Automatic Differentiation Fundamentals**: Understand what AD is and how it differs from numerical and symbolic differentiation
- **Forward and Reverse Mode**: Learn when to use each mode and why
- **Rust's Autodiff API**: Master the `#[autodiff]` attribute and activity annotations
- **Practical Applications**: Apply AD to real problems like optimization and machine learning

## Prerequisites

- Basic Rust knowledge (functions, types, references)
- High school calculus (understanding of derivatives)
- No prior AD experience required!

## How to Use This Tutorial

Each chapter builds on the previous one. We recommend following them in order:

1. **Chapter 1**: Understand what automatic differentiation is
2. **Chapter 2**: Set up your environment and compute your first derivative
3. **Chapter 3**: Learn about activity annotations
4. **Chapter 4-5**: Deep dive into reverse and forward modes
5. **Chapter 6**: Work through practical examples
6. **Chapter 7**: Explore advanced topics

## Running the Examples

All examples in this tutorial are available in the `examples/` directory. Run them with:

```bash
RUSTFLAGS="-Z autodiff=Enable" cargo run -p <example_name>
```

For example:
```bash
RUSTFLAGS="-Z autodiff=Enable" cargo run -p scalar_square
```

## Important Note

Rust's autodiff feature is **experimental** and requires the nightly compiler. The API may change in future releases. This tutorial is based on the current implementation as of 2025.

Let's get started!
