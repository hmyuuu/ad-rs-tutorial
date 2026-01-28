# Quantum Optimal Control

This example demonstrates autodiff for quantum optimal control - finding control pulses to implement quantum gates.

## Problem Setup

We want to find control pulses that transform a qubit from |0⟩ to |1⟩ (implementing an X gate).

The system evolves under:
- Free Hamiltonian: H₀ = ω₀ σz/2 (natural precession)
- Control Hamiltonian: Hc(t) = Ω(t) σx/2 (applied pulses)

## State Representation

A qubit state |ψ⟩ = c₀|0⟩ + c₁|1⟩ is stored as:
```rust
// [Re(c0), Im(c0), Re(c1), Im(c1)]
let state = [1.0, 0.0, 0.0, 0.0];  // |0⟩
```

## Rotation Gates

```rust
/// Rx(θ): rotation around X-axis
fn apply_rx(state: &mut [f64; 4], theta: f64) {
    let c = (theta / 2.0).cos();
    let s = (theta / 2.0).sin();
    // ... matrix multiplication
}
```

## Fidelity Optimization

The goal is to maximize fidelity F = |⟨ψ_target|ψ_final⟩|²:

```rust
#[autodiff_reverse(d_infidelity, Duplicated, Active)]
fn infidelity(controls: &[f64; N_STEPS]) -> f64 {
    let mut state = [1.0, 0.0, 0.0, 0.0];  // |0⟩
    let target = [0.0, 0.0, 1.0, 0.0];     // |1⟩

    // Apply control sequence
    for i in 0..N_STEPS {
        apply_rz(&mut state, omega0);
        apply_rx(&mut state, controls[i]);
    }

    // Return 1 - fidelity
    1.0 - compute_fidelity(&state, &target)
}
```

## Gradient-Based Optimization

```rust
let lr = 0.5;
for iter in 0..50 {
    let mut grad = [0.0; N_STEPS];
    let loss = d_infidelity(&controls, &mut grad, 1.0);

    for i in 0..N_STEPS {
        controls[i] -= lr * grad[i];
    }
}
```

## Results

The optimizer finds pulses that achieve >99% fidelity, with total rotation ≈ π (as expected for X gate).

Run the example:
```bash
RUSTFLAGS="-Z autodiff=Enable" cargo +enzyme run -p quantum_control
```
