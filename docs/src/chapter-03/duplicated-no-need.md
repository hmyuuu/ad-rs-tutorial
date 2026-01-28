# DuplicatedNoNeed

> **Note**: `DuplicatedNoNeed` is **not currently supported** in the Enzyme backend. Use `Duplicated` instead and ignore the primal return value if not needed.

## Original Intent

The `DuplicatedNoNeed` annotation was designed as an optimization for when you only need gradients, not the primal (forward) output value.

## Current Status

As of the current Enzyme version, attempting to use `DuplicatedNoNeed` will result in:

```
error: did not recognize Activity: `DuplicatedNoNeed`
```

## Workaround

Use `Duplicated` instead and simply ignore the primal return value:

```rust
#![feature(autodiff)]
use std::autodiff::autodiff_reverse;

#[autodiff_reverse(d_sum_squares, Duplicated, Active)]
fn sum_squares(x: &[f64]) -> f64 {
    let mut sum = 0.0;
    let mut i = 0;
    while i < x.len() {
        sum += x[i] * x[i];
        i += 1;
    }
    sum
}

fn main() {
    let x = [1.0, 2.0, 3.0];
    let mut grad = [0.0; 3];

    // Ignore the primal return value with _
    let _ = d_sum_squares(&x, &mut grad, 1.0);

    println!("∇f = {:?}", grad);  // [2, 4, 6]
}
```

## Key Points

- `DuplicatedNoNeed` is not currently supported
- Use `Duplicated` as a workaround
- Ignore the primal return value with `let _ = ...` if not needed
- This may change in future Enzyme versions
