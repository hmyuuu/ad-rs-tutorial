# Troubleshooting

Common issues and their solutions.

## Enzyme-Specific Issues

### "Cannot deduce type of extract" error

**Cause**: Enzyme cannot handle Rust's `Range` iterators (`for i in 0..n`).

**Solution**: Use `while` loops instead:

```rust
// DON'T
for i in 0..n {
    sum += x[i];
}

// DO
let mut i = 0;
while i < n {
    sum += x[i];
    i += 1;
}
```

### "No augmented forward pass found" error

**Cause**: Using std library functions that Enzyme doesn't support (e.g., `abs()`, `sqrt()`, `sin()`).

**Solution**: Implement these functions manually:

```rust
// Manual abs
let abs_x = if x < 0.0 { -x } else { x };

// Newton-Raphson sqrt
let mut guess = x / 2.0;
let mut j = 0;
while j < 10 {
    guess = (guess + x / guess) / 2.0;
    j += 1;
}
```

### "did not recognize Activity: DuplicatedNoNeed"

**Cause**: `DuplicatedNoNeed` is not supported in the current Enzyme version.

**Solution**: Use `Duplicated` instead and ignore the primal return value if not needed.

## Compilation Errors

### "feature `autodiff` is not enabled"

**Solution**: Add the feature flag at the top of your file:

```rust
#![feature(autodiff)]
```

### "cannot find macro `autodiff`" or "unresolved import"

**Solution**: Add the import:

```rust
use std::autodiff::autodiff;
```

### "unknown `-Z` flag: `autodiff`"

**Solution**: Update to a recent nightly:

```bash
rustup update nightly
```

### Linker errors mentioning LLVM or Enzyme

**Solution**: Ensure llvm-tools is installed:

```bash
rustup component add llvm-tools
```

## Runtime Issues

### Gradients are all zero

**Possible causes**:

1. Seed is 0.0:
   ```rust
   // WRONG
   let (y, grad) = d_f(x, 0.0);

   // RIGHT
   let (y, grad) = d_f(x, 1.0);
   ```

2. Parameter marked as `Const`:
   ```rust
   // WRONG - x won't get gradient
   #[autodiff(d_f, Reverse, Const, Active)]

   // RIGHT
   #[autodiff(d_f, Reverse, Active, Active)]
   ```

3. Computation doesn't depend on input:
   ```rust
   fn f(x: f64) -> f64 {
       42.0  // Constant - gradient is 0
   }
   ```

### Gradient buffer unchanged

**Cause**: Using `Const` instead of `Duplicated` for arrays:

```rust
// WRONG
#[autodiff(d_f, Reverse, Const, Active)]
fn f(x: &[f64]) -> f64 { ... }

// RIGHT
#[autodiff(d_f, Reverse, Duplicated, Active)]
fn f(x: &[f64]) -> f64 { ... }
```

### Wrong number of return values

**Cause**: Mismatch between annotations and destructuring:

```rust
#[autodiff(d_f, Reverse, Active, Active, Active)]
fn f(x: f64, y: f64) -> f64 { ... }

// WRONG - missing grad_y
let (result, grad_x) = d_f(x, y, 1.0);

// RIGHT
let (result, grad_x, grad_y) = d_f(x, y, 1.0);
```

### NaN or Inf in gradients

**Possible causes**:

1. Division by zero in the function
2. Log of zero or negative number
3. Overflow in exponentials

**Solution**: Add numerical guards:

```rust
// Guard against log(0)
let safe_x = x.max(1e-15);
safe_x.ln()

// Guard against division by zero
let safe_denom = denom.max(1e-15);
num / safe_denom
```

## Performance Issues

### Slow compilation

Autodiff adds compilation overhead. For faster iteration:

1. Use `cargo check` for syntax errors
2. Only enable autodiff when needed
3. Consider splitting into smaller functions

### Slow runtime

1. Ensure release mode: `cargo run --release`
2. Mark non-differentiated params as `Const`

## Unsupported Features

Some Rust features may not work with autodiff:

- Certain FFI calls
- Some unsafe operations
- Complex trait bounds

If you hit an unsupported feature, try:
1. Simplifying the function
2. Breaking into smaller pieces
3. Using a wrapper function

## Getting More Help

1. Check the [Rust autodiff tracking issue](https://github.com/rust-lang/rust/issues/124509)
2. Search existing issues for similar problems
3. Create a minimal reproduction case
4. File an issue with details about your setup
