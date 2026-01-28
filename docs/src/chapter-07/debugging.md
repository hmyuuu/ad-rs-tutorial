# Debugging Tips

When autodiff doesn't work as expected, here are strategies to diagnose and fix issues.

## Common Issues

### 1. Gradient is Zero When It Shouldn't Be

**Symptom**: All gradients are 0.0

**Possible causes**:
- Forgot to pass seed = 1.0
- Variable marked as `Const` instead of `Active`/`Duplicated`
- Computation doesn't depend on the input

```rust
// WRONG: seed is 0
let (y, grad) = d_f(x, 0.0);  // grad will be 0!

// RIGHT
let (y, grad) = d_f(x, 1.0);
```

### 2. Gradient Buffer Not Updated

**Symptom**: Gradient buffer stays at initial values

**Possible causes**:
- Buffer not passed correctly
- Wrong annotation (Const instead of Duplicated)

```rust
// WRONG: Using Const for array we want gradients for
#[autodiff(d_f, Reverse, Const, Active)]
fn f(x: &[f64]) -> f64 { ... }

// RIGHT: Use Duplicated
#[autodiff(d_f, Reverse, Duplicated, Active)]
fn f(x: &[f64]) -> f64 { ... }
```

### 3. Compilation Errors

**Symptom**: Cryptic errors about types or lifetimes

**Check**:
- Feature flag enabled: `#![feature(autodiff)]`
- Correct import: `use std::autodiff::autodiff;`
- RUSTFLAGS set: `-Z autodiff=Enable`

### 4. Wrong Number of Return Values

**Symptom**: Tuple destructuring fails

**Cause**: Mismatch between annotations and expected returns

```rust
// With two Active inputs, you get three return values
#[autodiff(d_f, Reverse, Active, Active, Active)]
fn f(x: f64, y: f64) -> f64 { ... }

// WRONG
let (result, grad) = d_f(x, y, 1.0);

// RIGHT
let (result, grad_x, grad_y) = d_f(x, y, 1.0);
```

## Verification Strategies

### 1. Finite Difference Check

Compare autodiff gradient to numerical approximation:

```rust
fn finite_diff(f: impl Fn(f64) -> f64, x: f64, h: f64) -> f64 {
    (f(x + h) - f(x - h)) / (2.0 * h)
}

fn verify_gradient() {
    let x = 2.0;
    let (_, autodiff_grad) = d_f(x, 1.0);
    let numerical_grad = finite_diff(f, x, 1e-5);

    let error = (autodiff_grad - numerical_grad).abs();
    assert!(error < 1e-4, "Gradient mismatch: {error}");
}
```

### 2. Known Derivatives

Test against functions with known derivatives:

| Function | Derivative | Test Point |
|----------|------------|------------|
| x² | 2x | x=3 → 6 |
| sin(x) | cos(x) | x=0 → 1 |
| eˣ | eˣ | x=0 → 1 |
| ln(x) | 1/x | x=2 → 0.5 |

### 3. Gradient Checking in ML

For loss functions, verify:
- Gradient is zero at minimum
- Gradient points away from minimum elsewhere

## Debugging Workflow

1. **Simplify**: Reduce to minimal reproducing example
2. **Verify annotations**: Check each parameter's annotation
3. **Check seed**: Ensure seed = 1.0 for direct gradients
4. **Numerical check**: Compare to finite differences
5. **Print intermediates**: Add debug prints in the function

## Error Messages

| Error | Likely Cause |
|-------|--------------|
| "feature not enabled" | Missing `#![feature(autodiff)]` |
| "unknown flag -Z" | Old nightly, run `rustup update` |
| "type mismatch" | Wrong annotation for parameter type |
| "cannot find autodiff" | Missing `use std::autodiff::autodiff` |

## Getting Help

If you're stuck:
1. Check the examples in this tutorial
2. Simplify to a minimal case
3. Verify with finite differences
4. Check Rust autodiff documentation and issues
