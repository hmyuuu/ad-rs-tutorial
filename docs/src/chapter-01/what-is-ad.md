# What is Automatic Differentiation?

Automatic Differentiation (AD) computes exact derivatives of functions by applying the chain rule systematically to elementary operations.

## Three Ways to Compute Derivatives

### 1. Numerical Differentiation (Finite Differences)

Approximate the derivative using small perturbations:

```
f'(x) ≈ (f(x + h) - f(x)) / h
```

**Pros**: Simple to implement, works for any function

**Cons**: Approximation errors, numerical instability, slow for many variables

### 2. Symbolic Differentiation

Apply calculus rules to derive an expression for the derivative:

```
f(x) = x² + sin(x)
f'(x) = 2x + cos(x)
```

**Pros**: Exact results, human-readable expressions

**Cons**: Expression swell (derivatives can be much larger than original), can't handle control flow

### 3. Automatic Differentiation

Decompose the function into elementary operations and apply the chain rule:

```rust
// f(x) = x² + sin(x)
// Decomposed:
// v₁ = x * x      // d(v₁)/dx = 2x
// v₂ = sin(x)    // d(v₂)/dx = cos(x)
// y = v₁ + v₂    // dy/dx = d(v₁)/dx + d(v₂)/dx = 2x + cos(x)
```

**Pros**: Exact (to machine precision), efficient, handles control flow
**Cons**: Requires tool support

## Why AD Matters

AD is the backbone of modern machine learning. When you train a neural network, you need gradients of the loss function with respect to millions of parameters. AD makes this tractable:

- **Exact**: No approximation errors like finite differences
- **Efficient**: Computes gradients in time proportional to the forward pass
- **Automatic**: No manual derivation required

## The Chain Rule

AD works by applying the chain rule. For a composition of functions:

```
y = f(g(x))
dy/dx = (df/dg) × (dg/dx)
```

Every program is ultimately a composition of elementary operations (+, -, ×, ÷, sin, cos, exp, log, etc.). AD tracks how each operation affects the derivative.

## Example: Computing f(x) = x² Step by Step

```rust
fn square(x: f64) -> f64 {
    x * x  // One multiplication
}
```

AD sees this as:
1. Input: x
2. Operation: multiply x by x
3. Output: x²

The derivative rule for multiplication: d(a × b)/dx = a × (db/dx) + b × (da/dx)

For x × x: d(x × x)/dx = x × 1 + x × 1 = 2x ✓

This is exactly what calculus tells us, but computed automatically!
