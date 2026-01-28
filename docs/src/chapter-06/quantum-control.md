# Quantum Optimal Control

This example demonstrates autodiff for quantum optimal control (QOC) - finding control pulses to implement high-fidelity quantum gates using gradient-based optimization.

## Background

Quantum optimal control is essential for implementing precise quantum operations in quantum computing and quantum simulation. The GRAPE (Gradient Ascent Pulse Engineering) algorithm [1] uses gradient information to optimize control pulses, making autodiff a natural fit.

## Problem Formulation

### System Hamiltonian

A driven qubit evolves under the time-dependent Hamiltonian:

$$H(t) = \frac{\omega_0}{2}\sigma_z + \frac{\Omega(t)}{2}\sigma_x$$

where:
- $\omega_0$ is the qubit frequency (drift term)
- $\Omega(t)$ is the time-dependent control amplitude
- $\sigma_x, \sigma_z$ are Pauli matrices

### Time Evolution

The unitary evolution operator for a small time step $\Delta t$ is:

$$U(\Delta t) = \exp(-i H \Delta t) \approx R_z(\omega_0 \Delta t) \cdot R_x(\Omega \Delta t)$$

where the rotation operators are:

$$R_x(\theta) = \exp\left(-i\frac{\theta}{2}\sigma_x\right) = \begin{pmatrix} \cos(\theta/2) & -i\sin(\theta/2) \\ -i\sin(\theta/2) & \cos(\theta/2) \end{pmatrix}$$

$$R_z(\theta) = \exp\left(-i\frac{\theta}{2}\sigma_z\right) = \begin{pmatrix} e^{-i\theta/2} & 0 \\ 0 & e^{i\theta/2} \end{pmatrix}$$

### Objective Function

The gate fidelity measures how close the final state is to the target:

$$F = |\langle\psi_{\text{target}}|\psi_{\text{final}}\rangle|^2$$

We minimize the infidelity $1 - F$ using gradient descent.

## Implementation

### State Representation

A qubit state $|\psi\rangle = c_0|0\rangle + c_1|1\rangle$ is stored as real/imaginary pairs:

```rust
// [Re(c0), Im(c0), Re(c1), Im(c1)]
let state = [1.0, 0.0, 0.0, 0.0];  // |0⟩
let target = [0.0, 0.0, 1.0, 0.0]; // |1⟩
```

### Rotation Gates

```rust
/// Rx(θ) = exp(-i θ σx/2)
fn apply_rx(state: &mut [f64; 4], theta: f64) {
    let c = my_cos(theta / 2.0);
    let s = my_sin(theta / 2.0);

    let (re0, im0, re1, im1) = (state[0], state[1], state[2], state[3]);

    // |0⟩ → cos(θ/2)|0⟩ - i sin(θ/2)|1⟩
    state[0] = c * re0 + s * im1;
    state[1] = c * im0 - s * re1;
    // |1⟩ → -i sin(θ/2)|0⟩ + cos(θ/2)|1⟩
    state[2] = s * im0 + c * re1;
    state[3] = -s * re0 + c * im1;
}
```

### Autodiff for Gradients

The key insight is that autodiff computes $\partial F / \partial \Omega_k$ for all time steps simultaneously:

```rust
#[autodiff_reverse(d_infidelity, Duplicated, Active)]
fn infidelity(controls: &[f64; N_STEPS]) -> f64 {
    let mut state = [1.0, 0.0, 0.0, 0.0];

    let mut i = 0;
    while i < N_STEPS {
        apply_rz(&mut state, omega0);
        apply_rx(&mut state, controls[i]);
        i += 1;
    }

    // Compute 1 - |⟨target|state⟩|²
    let overlap_re = target[2] * state[2] + target[3] * state[3];
    let overlap_im = target[2] * state[3] - target[3] * state[2];
    1.0 - (overlap_re * overlap_re + overlap_im * overlap_im)
}
```

### Adam Optimizer

We use the Adam optimizer [2] for faster convergence:

```rust
// Adam update rule
m[i] = β₁ * m[i] + (1 - β₁) * g;
v[i] = β₂ * v[i] + (1 - β₂) * g²;
m_hat = m[i] / (1 - β₁^t);
v_hat = v[i] / (1 - β₂^t);
controls[i] -= lr * m_hat / (√v_hat + ε);
```

## Results

With 8 time steps and 500 Adam iterations, we achieve:

| Metric | Value |
|--------|-------|
| Final fidelity | **99.9994%** |
| Gradient magnitude | ~10⁻⁴ |
| Total pulse area | 3.134 ≈ π |

The optimized pulse sequence compensates for the drift Hamiltonian while accumulating a total rotation of π around the X-axis.

## Physical Interpretation

The optimal control pulses show characteristic features:
- **Symmetric structure**: The pulse sequence is approximately time-symmetric
- **Compensation**: Larger pulses in the middle compensate for accumulated phase from $H_0$
- **Total rotation**: Sum of pulses ≈ π, as required for an X gate

## Run the Example

```bash
RUSTFLAGS="-Z autodiff=Enable" cargo +enzyme run -p quantum_control
```

## References

1. N. Khaneja, T. Reiss, C. Kehlet, T. Schulte-Herbrüggen, and S. J. Glaser, "Optimal control of coupled spin dynamics: design of NMR pulse sequences by gradient ascent algorithms," *Journal of Magnetic Resonance*, vol. 172, no. 2, pp. 296-305, 2005. [DOI: 10.1016/j.jmr.2004.11.004](https://doi.org/10.1016/j.jmr.2004.11.004)

2. D. P. Kingma and J. Ba, "Adam: A Method for Stochastic Optimization," *arXiv:1412.6980*, 2014. [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)

3. C. P. Koch et al., "Quantum optimal control in quantum technologies. Strategic report on current status, visions and goals for research in Europe," *EPJ Quantum Technology*, vol. 9, no. 19, 2022. [DOI: 10.1140/epjqt/s40507-022-00138-x](https://doi.org/10.1140/epjqt/s40507-022-00138-x)
