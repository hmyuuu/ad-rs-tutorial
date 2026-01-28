//! Quantum Optimal Control with Autodiff
//!
//! Demonstrates autodiff for quantum optimal control:
//! - State evolution under parameterized Hamiltonian
//! - Fidelity optimization with GRAPE-like algorithm
//! - Gradient-based pulse optimization
//!
//! Model: 2-level quantum system (qubit) with drift and control

#![feature(autodiff)]
use std::autodiff::autodiff_reverse;

/// Taylor series sin(x) - high precision
fn my_sin(x: f64) -> f64 {
    let mut sum = x;
    let mut term = x;
    let mut k = 1;
    while k < 30 {
        term *= -x * x / ((2 * k) as f64 * (2 * k + 1) as f64);
        sum += term;
        k += 1;
    }
    sum
}

/// Taylor series cos(x) - high precision
fn my_cos(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    let mut k = 1;
    while k < 30 {
        term *= -x * x / ((2 * k - 1) as f64 * (2 * k) as f64);
        sum += term;
        k += 1;
    }
    sum
}

/// State: [Re(c0), Im(c0), Re(c1), Im(c1)] where |ψ⟩ = c0|0⟩ + c1|1⟩
const N_STEPS: usize = 8;

/// Apply rotation around X-axis: Rx(θ) = exp(-i θ σx/2)
fn apply_rx(state: &mut [f64; 4], theta: f64) {
    let c = my_cos(theta / 2.0);
    let s = my_sin(theta / 2.0);

    let re0 = state[0];
    let im0 = state[1];
    let re1 = state[2];
    let im1 = state[3];

    state[0] = c * re0 + s * im1;
    state[1] = c * im0 - s * re1;
    state[2] = s * im0 + c * re1;
    state[3] = -s * re0 + c * im1;
}

/// Apply rotation around Z-axis: Rz(θ) = exp(-i θ σz/2)
fn apply_rz(state: &mut [f64; 4], theta: f64) {
    let c = my_cos(theta / 2.0);
    let s = my_sin(theta / 2.0);

    let re0 = state[0];
    let im0 = state[1];
    let re1 = state[2];
    let im1 = state[3];

    state[0] = c * re0 + s * im0;
    state[1] = c * im0 - s * re0;
    state[2] = c * re1 - s * im1;
    state[3] = c * im1 + s * re1;
}

/// Quantum gate fidelity: F = |⟨ψ_target|ψ_final⟩|²
#[autodiff_reverse(d_infidelity, Duplicated, Active)]
fn infidelity(controls: &[f64; N_STEPS]) -> f64 {
    let mut state = [1.0, 0.0, 0.0, 0.0]; // |0⟩
    let target = [0.0, 0.0, 1.0, 0.0]; // |1⟩ (X gate target)

    let omega0 = 0.05; // Smaller drift for easier control

    let mut i = 0;
    while i < N_STEPS {
        apply_rz(&mut state, omega0);
        apply_rx(&mut state, controls[i]);
        i += 1;
    }

    let re_overlap =
        target[0] * state[0] + target[1] * state[1] + target[2] * state[2] + target[3] * state[3];
    let im_overlap =
        target[0] * state[1] - target[1] * state[0] + target[2] * state[3] - target[3] * state[2];

    1.0 - (re_overlap * re_overlap + im_overlap * im_overlap)
}

/// Energy cost for regularization
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
    println!("Goal: Find control pulses to implement X gate (|0⟩ → |1⟩)");
    println!(
        "Method: GRAPE-like gradient descent with {} time steps\n",
        N_STEPS
    );

    // Initialize with π/N_STEPS per step (analytical solution hint)
    let pi = std::f64::consts::PI;
    let mut controls = [pi / (N_STEPS as f64); N_STEPS];

    println!("Initial controls (π/{} each): {:?}", N_STEPS, controls);

    let mut grad = [0.0; N_STEPS];
    let initial_infid = d_infidelity(&controls, &mut grad, 1.0);
    println!("Initial fidelity: {:.6}\n", 1.0 - initial_infid);

    // Adam optimizer parameters
    let mut m = [0.0; N_STEPS]; // First moment
    let mut v = [0.0; N_STEPS]; // Second moment
    let beta1 = 0.9;
    let beta2 = 0.999;
    let epsilon = 1e-8;
    let lr = 0.1;
    let lambda = 0.0001; // Very small energy penalty for high fidelity

    println!("Running Adam optimization...\n");

    for iter in 0..500 {
        let mut grad_infid = [0.0; N_STEPS];
        let mut grad_energy = [0.0; N_STEPS];

        let infid = d_infidelity(&controls, &mut grad_infid, 1.0);
        let energy = d_energy(&controls, &mut grad_energy, 1.0);

        // Adam update
        for i in 0..N_STEPS {
            let g = grad_infid[i] + lambda * grad_energy[i];
            m[i] = beta1 * m[i] + (1.0 - beta1) * g;
            v[i] = beta2 * v[i] + (1.0 - beta2) * g * g;

            let m_hat = m[i] / (1.0 - beta1.powi(iter as i32 + 1));
            let v_hat = v[i] / (1.0 - beta2.powi(iter as i32 + 1));

            controls[i] -= lr * m_hat / (v_hat.sqrt() + epsilon);
        }

        if iter % 100 == 0 || iter == 499 {
            println!(
                "Iter {:3}: fidelity={:.8}, energy={:.4}",
                iter,
                1.0 - infid,
                energy
            );
        }
    }

    println!("\n--- Final Results ---");
    println!("Controls: {:?}", controls);

    let mut final_grad = [0.0; N_STEPS];
    let final_infid = d_infidelity(&controls, &mut final_grad, 1.0);
    let final_fidelity = 1.0 - final_infid;

    println!("Final fidelity: {:.10}", final_fidelity);
    println!(
        "Gradient magnitude: {:.2e}",
        final_grad.iter().map(|x| x * x).sum::<f64>().sqrt()
    );
    println!(
        "Sum of pulses: {:.6} (target π = {:.6})",
        controls.iter().sum::<f64>(),
        pi
    );

    // Verify with analytical X gate
    println!("\n--- Verification ---");
    if final_fidelity > 0.9999 {
        println!("SUCCESS: Achieved >99.99% fidelity!");
    } else if final_fidelity > 0.999 {
        println!("GOOD: Achieved >99.9% fidelity");
    } else if final_fidelity > 0.99 {
        println!("OK: Achieved >99% fidelity");
    } else {
        println!("Need more optimization iterations");
    }
}
