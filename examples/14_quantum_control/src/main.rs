//! Quantum Optimal Control with Autodiff
//!
//! Demonstrates autodiff for quantum optimal control:
//! - State evolution under parameterized Hamiltonian
//! - Fidelity optimization
//! - Gradient-based pulse optimization
//!
//! Simplified model: 2-level quantum system (qubit)

#![feature(autodiff)]
use std::autodiff::autodiff_reverse;

/// Taylor series sin(x)
fn my_sin(x: f64) -> f64 {
    let mut sum = x;
    let mut term = x;
    let mut k = 1;
    while k < 25 {
        term *= -x * x / ((2 * k) as f64 * (2 * k + 1) as f64);
        sum += term;
        k += 1;
    }
    sum
}

/// Taylor series cos(x)
fn my_cos(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    let mut k = 1;
    while k < 25 {
        term *= -x * x / ((2 * k - 1) as f64 * (2 * k) as f64);
        sum += term;
        k += 1;
    }
    sum
}

/// Simulate qubit evolution under control pulse
/// H(t) = ω₀ σz/2 + Ω(t) σx/2
///
/// For small time steps, evolution is approximately:
/// |ψ(t+dt)⟩ ≈ exp(-i H dt) |ψ(t)⟩
///
/// We use a simplified model where control parameters directly
/// affect the final state amplitudes.
///
/// State: [Re(c0), Im(c0), Re(c1), Im(c1)] where |ψ⟩ = c0|0⟩ + c1|1⟩
/// Control: pulse amplitudes at each time step

const N_STEPS: usize = 4;

/// Apply rotation around X-axis: Rx(θ) = exp(-i θ σx/2)
/// |0⟩ → cos(θ/2)|0⟩ - i sin(θ/2)|1⟩
/// |1⟩ → -i sin(θ/2)|0⟩ + cos(θ/2)|1⟩
fn apply_rx(state: &mut [f64; 4], theta: f64) {
    let c = my_cos(theta / 2.0);
    let s = my_sin(theta / 2.0);

    let re0 = state[0];
    let im0 = state[1];
    let re1 = state[2];
    let im1 = state[3];

    // New c0 = c * c0 - i*s * c1 = (c*re0 + s*im1) + i(c*im0 - s*re1)
    state[0] = c * re0 + s * im1;
    state[1] = c * im0 - s * re1;
    // New c1 = -i*s * c0 + c * c1 = (s*im0 + c*re1) + i(-s*re0 + c*im1)
    state[2] = s * im0 + c * re1;
    state[3] = -s * re0 + c * im1;
}

/// Apply rotation around Z-axis: Rz(θ) = exp(-i θ σz/2)
/// |0⟩ → exp(-iθ/2)|0⟩
/// |1⟩ → exp(+iθ/2)|1⟩
fn apply_rz(state: &mut [f64; 4], theta: f64) {
    let c = my_cos(theta / 2.0);
    let s = my_sin(theta / 2.0);

    let re0 = state[0];
    let im0 = state[1];
    let re1 = state[2];
    let im1 = state[3];

    // c0 → exp(-iθ/2) * c0 = (c + is)(re0 + i*im0) = (c*re0 + s*im0) + i(c*im0 - s*re0)
    state[0] = c * re0 + s * im0;
    state[1] = c * im0 - s * re0;
    // c1 → exp(+iθ/2) * c1 = (c - is)(re1 + i*im1) = (c*re1 - s*im1) + i(c*im1 + s*re1)
    state[2] = c * re1 - s * im1;
    state[3] = c * im1 + s * re1;
}

/// Quantum gate fidelity: F = |⟨ψ_target|ψ_final⟩|²
/// We want to maximize this (minimize 1 - F)
#[autodiff_reverse(d_infidelity, Duplicated, Active)]
fn infidelity(controls: &[f64; N_STEPS]) -> f64 {
    // Initial state: |0⟩ = [1, 0, 0, 0]
    let mut state = [1.0, 0.0, 0.0, 0.0];

    // Target state: |1⟩ = [0, 0, 1, 0] (X gate on |0⟩)
    let target = [0.0, 0.0, 1.0, 0.0];

    // Fixed system frequency
    let omega0 = 0.1;

    // Time evolution with control pulses
    let mut i = 0;
    while i < N_STEPS {
        // Free evolution (Z rotation)
        apply_rz(&mut state, omega0);
        // Control pulse (X rotation)
        apply_rx(&mut state, controls[i]);
        i += 1;
    }

    // Compute fidelity: |⟨target|state⟩|²
    // ⟨target|state⟩ = target* · state (complex inner product)
    let re_overlap =
        target[0] * state[0] + target[1] * state[1] + target[2] * state[2] + target[3] * state[3];
    let im_overlap =
        target[0] * state[1] - target[1] * state[0] + target[2] * state[3] - target[3] * state[2];

    let fidelity = re_overlap * re_overlap + im_overlap * im_overlap;

    // Return infidelity (to minimize)
    1.0 - fidelity
}

/// Energy cost: penalize large control amplitudes
#[autodiff_reverse(d_energy, Duplicated, Active)]
fn energy_cost(controls: &[f64; N_STEPS]) -> f64 {
    let mut sum = 0.0;
    let mut i = 0;
    while i < N_STEPS {
        sum += controls[i] * controls[i];
        i += 1;
    }
    sum
}

fn main() {
    println!("Quantum Optimal Control with Autodiff");
    println!("======================================\n");

    println!("Goal: Find control pulses to implement X gate (|0⟩ → |1⟩)\n");

    // Initial guess: small random-ish pulses
    let mut controls = [0.5, 0.3, 0.2, 0.1];

    println!("Initial controls: {:?}", controls);

    let mut grad = [0.0; N_STEPS];
    let initial_infid = d_infidelity(&controls, &mut grad, 1.0);
    println!("Initial infidelity: {:.6}", initial_infid);
    println!("Initial fidelity: {:.6}", 1.0 - initial_infid);
    println!("Gradient ∂(infidelity)/∂controls: {:?}\n", grad);

    // Gradient descent optimization
    println!("Running gradient descent optimization...\n");

    let lr = 0.5;
    let lambda = 0.01; // Energy penalty weight

    for iter in 0..50 {
        // Compute gradients
        let mut grad_infid = [0.0; N_STEPS];
        let mut grad_energy = [0.0; N_STEPS];

        let infid = d_infidelity(&controls, &mut grad_infid, 1.0);
        let energy = d_energy(&controls, &mut grad_energy, 1.0);

        // Total loss = infidelity + lambda * energy
        let loss = infid + lambda * energy;

        // Update controls
        for i in 0..N_STEPS {
            controls[i] -= lr * (grad_infid[i] + lambda * grad_energy[i]);
        }

        if iter % 10 == 0 || iter == 49 {
            println!(
                "Iter {:2}: loss={:.6}, fidelity={:.6}, energy={:.4}",
                iter,
                loss,
                1.0 - infid,
                energy
            );
        }
    }

    println!("\nFinal controls: {:?}", controls);

    // Verify final result
    let mut final_grad = [0.0; N_STEPS];
    let final_infid = d_infidelity(&controls, &mut final_grad, 1.0);
    println!("Final fidelity: {:.6}", 1.0 - final_infid);
    println!(
        "Final gradient magnitude: {:.6}",
        final_grad.iter().map(|x| x * x).sum::<f64>().sqrt()
    );

    // Analytical solution for comparison
    println!("\nNote: Optimal X gate requires total rotation of π around X-axis");
    println!(
        "Sum of control pulses: {:.4} (target ≈ π = {:.4})",
        controls.iter().sum::<f64>(),
        std::f64::consts::PI
    );
}
