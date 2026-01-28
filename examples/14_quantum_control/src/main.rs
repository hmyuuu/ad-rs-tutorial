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

/// Newton-Raphson sqrt
fn my_sqrt(x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let mut guess = x / 2.0;
    let mut i = 0;
    while i < 20 {
        guess = (guess + x / guess) / 2.0;
        i += 1;
    }
    guess
}

/// State: [Re(c0), Im(c0), Re(c1), Im(c1)] where |ψ⟩ = c0|0⟩ + c1|1⟩
const N_STEPS: usize = 100;
const DT: f64 = 1.0 / N_STEPS as f64;
const N_ITERS: usize = 200;
const OMEGA0: f64 = 1.0; // Drift frequency

/// Exact time evolution: U = exp(-i(ω₀σz/2 + Ωσx/2)Δt)
/// Uses: e^{-i(n·σ)θ/2} = cos(θ/2)I - i sin(θ/2)(n·σ)
fn apply_exact_step(state: &mut [f64; 4], omega: f64) {
    let omega_eff = my_sqrt(OMEGA0 * OMEGA0 + omega * omega);
    let theta = omega_eff * DT;

    // Normalized rotation axis: n = (omega, 0, omega0) / omega_eff
    let nx = omega / omega_eff;
    let nz = OMEGA0 / omega_eff;

    let c = my_cos(theta / 2.0);
    let s = my_sin(theta / 2.0);

    let (re0, im0, re1, im1) = (state[0], state[1], state[2], state[3]);

    // U = cos(θ/2)I - i sin(θ/2)(nx σx + nz σz)
    // U|0⟩ = (cos - i nz sin)|0⟩ - i nx sin|1⟩
    // U|1⟩ = -i nx sin|0⟩ + (cos + i nz sin)|1⟩
    state[0] = c * re0 + s * nz * im0 + s * nx * im1;
    state[1] = c * im0 - s * nz * re0 - s * nx * re1;
    state[2] = c * re1 - s * nz * im1 + s * nx * im0;
    state[3] = c * im1 + s * nz * re1 - s * nx * re0;
}

/// Quantum gate fidelity: F = |⟨ψ_target|ψ_final⟩|²
#[autodiff_reverse(d_infidelity, Duplicated, Active)]
fn infidelity(controls: &[f64; N_STEPS]) -> f64 {
    let mut state = [1.0, 0.0, 0.0, 0.0]; // |0⟩
    let target = [0.0, 0.0, 1.0, 0.0]; // |1⟩ (X gate target)

    let mut i = 0;
    while i < N_STEPS {
        apply_exact_step(&mut state, controls[i]);
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
        "Method: Exact propagator (no Trotter error), {} steps\n",
        N_STEPS
    );

    let pi = std::f64::consts::PI;
    let mut controls = [1.0; N_STEPS]; // Small initial perturbation

    println!("Initial controls (0.1 each)");

    let mut grad = [0.0; N_STEPS];
    let initial_infid = d_infidelity(&controls, &mut grad, 1.0);
    println!("Initial fidelity: {:.6}\n", 1.0 - initial_infid);

    // Adam optimizer
    let mut m = [0.0; N_STEPS];
    let mut v = [0.0; N_STEPS];
    let (beta1, beta2, epsilon, lr, lambda) = (0.9, 0.999, 1e-8, 0.3, 0.0);

    println!("Running Adam optimization...\n");

    for iter in 0..N_ITERS {
        let mut grad_infid = [0.0; N_STEPS];
        let mut grad_energy = [0.0; N_STEPS];

        let infid = d_infidelity(&controls, &mut grad_infid, 1.0);
        let _energy = d_energy(&controls, &mut grad_energy, 1.0);

        for i in 0..N_STEPS {
            let g = grad_infid[i] + lambda * grad_energy[i];
            m[i] = beta1 * m[i] + (1.0 - beta1) * g;
            v[i] = beta2 * v[i] + (1.0 - beta2) * g * g;
            let m_hat = m[i] / (1.0 - beta1.powi(iter as i32 + 1));
            let v_hat = v[i] / (1.0 - beta2.powi(iter as i32 + 1));
            controls[i] -= lr * m_hat / (v_hat.sqrt() + epsilon);
        }

        if iter % 200 == 0 || iter == N_ITERS - 1 {
            println!("Iter {:4}: fidelity={:.10}", iter, 1.0 - infid);
        }
    }

    println!("\n--- Final Results ---");
    let mut final_grad = [0.0; N_STEPS];
    let final_infid = d_infidelity(&controls, &mut final_grad, 1.0);
    let final_fidelity = 1.0 - final_infid;

    println!("Final fidelity: {:.10}", final_fidelity);
    println!(
        "Total pulse area: {:.6} (target π = {:.6})",
        controls.iter().sum::<f64>() * DT,
        pi
    );

    println!("\n--- Verification ---");
    if final_fidelity > 0.9999 {
        println!("SUCCESS: Achieved >99.99% fidelity!");
    } else if final_fidelity > 0.999 {
        println!("GOOD: Achieved >99.9% fidelity");
    } else {
        println!("Need more optimization iterations");
    }
}
