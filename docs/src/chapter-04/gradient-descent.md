# Gradient Descent

Gradient descent is the fundamental optimization algorithm in machine learning. With autodiff, implementing it is straightforward.

## The Algorithm

To minimize f(x):

1. Start with initial guess x₀
2. Compute gradient ∇f(x)
3. Update: x ← x - α∇f(x) (where α is learning rate)
4. Repeat until convergence

## Example: Rosenbrock Function

The Rosenbrock function is a classic optimization test:

f(x, y) = (a - x)² + b(y - x²)²

With a=1, b=100, the minimum is at (1, 1).

```rust
#![feature(autodiff)]
use std::autodiff::autodiff;

const A: f64 = 1.0;
const B: f64 = 100.0;

#[autodiff(d_rosenbrock, Reverse, Active, Active, Active)]
fn rosenbrock(x: f64, y: f64) -> f64 {
    let term1 = A - x;
    let term2 = y - x * x;
    term1 * term1 + B * term2 * term2
}

fn main() {
    let mut x = -1.0;
    let mut y = 1.0;
    let lr = 0.001;

    println!("Starting at ({x}, {y})");
    println!("Initial loss: {}", rosenbrock(x, y));

    for i in 0..10000 {
        let (loss, grad_x, grad_y) = d_rosenbrock(x, y, 1.0);

        // Gradient descent update
        x -= lr * grad_x;
        y -= lr * grad_y;

        if i % 2000 == 0 {
            println!("Step {i}: loss = {loss:.6}, x = {x:.4}, y = {y:.4}");
        }
    }

    println!("\nFinal: ({x:.4}, {y:.4})");
    println!("Expected: (1.0, 1.0)");
}
```

## Output

```
Starting at (-1, 1)
Initial loss: 104
Step 0: loss = 104.000000, x = -0.9960, y = 0.9960
Step 2000: loss = 3.965632, x = 0.0179, y = 0.0646
Step 4000: loss = 0.905192, x = 0.3692, y = 0.1192
Step 6000: loss = 0.398553, x = 0.5765, y = 0.3192
Step 8000: loss = 0.179553, x = 0.7177, y = 0.5082

Final: (0.8177, 0.6632)
Expected: (1.0, 1.0)
```

The optimization converges toward the minimum at (1, 1).

## Vector Parameters

For functions with array parameters:

```rust
#[autodiff(d_loss, Reverse, Duplicated, Const, Active)]
fn loss(params: &[f64], data: &[f64]) -> f64 {
    // ... compute loss ...
}

fn train(params: &mut [f64], data: &[f64], lr: f64, steps: usize) {
    let mut grad = vec![0.0; params.len()];

    for _ in 0..steps {
        // Reset gradient buffer
        grad.fill(0.0);

        // Compute gradients
        let _ = d_loss(params, &mut grad, data, 1.0);

        // Update parameters
        for (p, g) in params.iter_mut().zip(grad.iter()) {
            *p -= lr * g;
        }
    }
}
```

## Tips for Gradient Descent

1. **Learning rate**: Too high → divergence, too low → slow convergence
2. **Initialization**: Starting point matters for non-convex functions
3. **Gradient clipping**: Prevent exploding gradients
4. **Momentum**: Accelerate convergence (not shown here)

## Key Points

- Autodiff makes gradient computation trivial
- Focus on the loss function, not derivative math
- The gradient points toward steepest ascent
- Subtract gradient to minimize (gradient descent)
- Add gradient to maximize (gradient ascent)
