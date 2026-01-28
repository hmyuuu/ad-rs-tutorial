# Forward vs Reverse Mode

AD has two fundamental modes: **forward mode** and **reverse mode**. Understanding when to use each is crucial for efficiency.

## Forward Mode

Forward mode propagates derivatives alongside the computation, from inputs to outputs.

```
Input → ... → Output
  ↓           ↓
 dx  → ... → dy
```

For each input perturbation dx, we compute the corresponding output perturbation dy.

### How It Works

Consider f(x) = x³:

| Step | Primal | Tangent (derivative) |
|------|--------|---------------------|
| v₀ = x | v₀ = 2 | v̇₀ = 1 (seed) |
| v₁ = v₀ × v₀ | v₁ = 4 | v̇₁ = v̇₀ × v₀ + v₀ × v̇₀ = 4 |
| v₂ = v₁ × v₀ | v₂ = 8 | v̇₂ = v̇₁ × v₀ + v₁ × v̇₀ = 12 |

Result: f(2) = 8, f'(2) = 12 ✓ (since d(x³)/dx = 3x² = 12)

### Complexity

- **One forward pass** computes the derivative with respect to **one input**
- For n inputs: need n forward passes
- Cost: O(n × cost of f)

## Reverse Mode

Reverse mode first computes the function, then propagates derivatives backward from outputs to inputs.

```
Input → ... → Output
  ↑           ↑
 x̄   ← ... ← ȳ
```

Starting from an output sensitivity ȳ, we compute input sensitivities x̄.

### How It Works

Consider f(x) = x³ again:

**Forward pass** (compute and store intermediate values):
| Step | Value |
|------|-------|
| v₀ = x | 2 |
| v₁ = v₀ × v₀ | 4 |
| v₂ = v₁ × v₀ | 8 |

**Reverse pass** (propagate adjoints backward):
| Step | Adjoint |
|------|---------|
| v̄₂ = 1 | (seed) |
| v̄₁ = v̄₂ × v₀ = 2 | |
| v̄₀ = v̄₂ × v₁ + v̄₁ × 2v₀ = 4 + 8 = 12 | |

Result: f'(2) = 12 ✓

### Complexity

- **One reverse pass** computes derivatives with respect to **all inputs**
- For m outputs: need m reverse passes
- Cost: O(m × cost of f)

## When to Use Which?

| Scenario | Inputs | Outputs | Best Mode |
|----------|--------|---------|-----------|
| Scalar function | 1 | 1 | Either |
| Gradient (ML) | Many | 1 | **Reverse** |
| Jacobian column | 1 | Many | **Forward** |
| Full Jacobian | n | m | Depends |

### Machine Learning: Reverse Mode Wins

In ML, we typically have:
- **Many inputs**: millions of parameters
- **One output**: scalar loss

Reverse mode computes all gradients in one backward pass — this is why it's called "backpropagation"!

### Forward Mode Use Cases

Forward mode is better when:
- Few inputs, many outputs
- Computing directional derivatives
- Jacobian-vector products

## In Rust

```rust
// Reverse mode
#[autodiff(d_f, Reverse, Active, Active)]
fn f(x: f64) -> f64 { ... }

// Forward mode
#[autodiff(d_f, Forward, Dual, Dual)]
fn f(x: f64) -> f64 { ... }
```

We'll explore both modes in detail in later chapters.
