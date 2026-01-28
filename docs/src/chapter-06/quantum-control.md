# Quantum Optimal Control

This example uses autodiff to find control pulses that implement high-fidelity quantum gates via the GRAPE algorithm.

## Problem Setup

A driven qubit evolves under:

\\[ H(t) = \frac{\omega_0}{2}\sigma_z + \frac{\Omega(t)}{2}\sigma_x \\]

where \\(\omega_0\\) is the drift frequency and \\(\Omega(t)\\) is the control amplitude.

For small time steps, the evolution factorizes as \\(U \approx R_z(\omega_0 \Delta t) \cdot R_x(\Omega \Delta t)\\), where:

\\[ R_x(\theta) = \begin{pmatrix} \cos(\theta/2) & -i\sin(\theta/2) \\\\ -i\sin(\theta/2) & \cos(\theta/2) \end{pmatrix}, \quad R_z(\theta) = \begin{pmatrix} e^{-i\theta/2} & 0 \\\\ 0 & e^{i\theta/2} \end{pmatrix} \\]

We minimize the infidelity \\(1 - F\\) where \\(F = |\langle\psi_{\text{target}}|\psi_{\text{final}}\rangle|^2\\).

## Implementation

Qubit states are stored as `[Re(c0), Im(c0), Re(c1), Im(c1)]`:

```rust
#[autodiff_reverse(d_infidelity, Duplicated, Active)]
fn infidelity(controls: &[f64; N_STEPS]) -> f64 {
    let mut state = [1.0, 0.0, 0.0, 0.0]; // |0⟩

    let mut i = 0;
    while i < N_STEPS {
        apply_rz(&mut state, omega0);
        apply_rx(&mut state, controls[i]);
        i += 1;
    }

    // 1 - |⟨target|state⟩|²
    let overlap_re = target[2] * state[2] + target[3] * state[3];
    let overlap_im = target[2] * state[3] - target[3] * state[2];
    1.0 - (overlap_re * overlap_re + overlap_im * overlap_im)
}
```

## Results

With 8 time steps and 500 Adam iterations:

| Metric | Value |
|--------|-------|
| Final fidelity | **99.9994%** |
| Total pulse area | 3.134 ≈ π |

## Run

```bash
RUSTFLAGS="-Z autodiff=Enable" cargo +enzyme run -p quantum_control
```

## References

1. Khaneja et al., "Optimal control of coupled spin dynamics," *J. Magn. Reson.* 172, 296 (2005). [DOI](https://doi.org/10.1016/j.jmr.2004.11.004)
2. Kingma & Ba, "Adam: A Method for Stochastic Optimization," [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)
