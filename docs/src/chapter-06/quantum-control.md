# Quantum Optimal Control

This example uses autodiff to find control pulses that implement high-fidelity quantum gates via the GRAPE algorithm.

## Problem Setup

**Hamiltonian:**

\\[ H(t) = \frac{\omega_0}{2}\sigma_z + \frac{\Omega(t)}{2}\sigma_x \\]

where \\(\omega_0\\) is the drift frequency and \\(\Omega(t)\\) is the control amplitude.

**Time evolution:**

\\[ U = e^{-i(\mathbf{n}\cdot\boldsymbol{\sigma})\theta/2} = \cos(\theta/2)I - i\sin(\theta/2)(\mathbf{n}\cdot\boldsymbol{\sigma}) \\]

where \\(\theta = \sqrt{\omega_0^2 + \Omega^2}\Delta t\\) and \\(\mathbf{n} = (\Omega, 0, \omega_0)/\sqrt{\omega_0^2 + \Omega^2}\\).

**Objective:** Minimize infidelity \\(1 - F\\) where \\(F = |\langle\psi_{\text{target}}|\psi_{\text{final}}\rangle|^2\\).

## Results

With `N_STEPS = 100`, `N_ITERS = 200`:

| Metric | Value |
|--------|-------|
| Final fidelity | **>99.99%** |
| Total pulse area | 3.09 ≈ π |

## Run

```bash
RUSTFLAGS="-Z autodiff=Enable" cargo +enzyme run -p quantum_control
```

## References

1. Khaneja et al., "Optimal control of coupled spin dynamics," *J. Magn. Reson.* 172, 296 (2005). [DOI](https://doi.org/10.1016/j.jmr.2004.11.004)
2. Kingma & Ba, "Adam: A Method for Stochastic Optimization," [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)
