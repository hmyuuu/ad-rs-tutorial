# When to Use Forward Mode

Choosing between forward and reverse mode depends on the shape of your function.

## The Rule of Thumb

| Inputs | Outputs | Best Mode |
|--------|---------|-----------|
| Few | Many | **Forward** |
| Many | Few | **Reverse** |
| 1 | 1 | Either |

## Why the Difference?

**Forward mode**: One pass computes derivative w.r.t. one input
- Cost for n inputs: n × forward passes

**Reverse mode**: One pass computes derivatives w.r.t. all inputs
- Cost for m outputs: m × backward passes

## Forward Mode Use Cases

### 1. Jacobian-Vector Products (JVP)

Computing J × v where J is the Jacobian:

```rust
#[autodiff(jvp_f, Forward, Dual, Dual)]
fn f(x: &[f64]) -> Vec<f64> { ... }

// JVP: J × v
let (y, jv) = jvp_f(&x, &v);
```

### 2. Few Inputs, Many Outputs

```rust
// f: R¹ → R¹⁰⁰
// Forward mode: 1 pass
// Reverse mode: 100 passes
#[autodiff(d_f, Forward, Dual, Dual)]
fn f(x: f64) -> [f64; 100] { ... }
```

### 3. Sensitivity Analysis

How does output change with small input perturbation?

```rust
let perturbation = 0.01;
let (_, sensitivity) = d_f(x, perturbation);
```

### 4. Taylor Series Computation

Forward mode naturally extends to higher derivatives for Taylor expansion.

## Reverse Mode Use Cases

### 1. Machine Learning (Gradients)

```rust
// Loss: R^n → R¹ (n = millions of parameters)
// Reverse mode: 1 pass for all gradients
#[autodiff(d_loss, Reverse, Duplicated, Active)]
fn loss(params: &[f64]) -> f64 { ... }
```

### 2. Vector-Jacobian Products (VJP)

Computing vᵀ × J:

```rust
#[autodiff(vjp_f, Reverse, Duplicated, Active)]
fn f(x: &[f64]) -> f64 { ... }
```

### 3. Optimization

Any scalar objective with many parameters.

## Comparison Table

| Aspect | Forward | Reverse |
|--------|---------|---------|
| Computes | JVP (J × v) | VJP (vᵀ × J) |
| Memory | O(1) extra | O(n) for tape |
| Best for | Few inputs | Few outputs |
| ML use | Rare | Common |

## Hybrid Approaches

For full Jacobian computation:
- n inputs, m outputs
- Forward: n passes
- Reverse: m passes
- Choose based on min(n, m)

## Key Points

- Forward mode: efficient for few inputs, many outputs
- Reverse mode: efficient for many inputs, few outputs
- ML typically uses reverse mode (scalar loss, many params)
- Forward mode useful for JVPs and sensitivity analysis
- When in doubt, profile both approaches
