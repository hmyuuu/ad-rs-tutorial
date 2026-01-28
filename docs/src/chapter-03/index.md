# Activity Annotations

Activity annotations tell the autodiff system how each parameter and return value participates in differentiation. Choosing the right annotation is crucial for correctness and efficiency.

In this chapter, you'll learn about:

- **Active**: For scalar values that need gradients
- **Const**: For parameters that don't need gradients
- **Duplicated**: For arrays/slices with gradient buffers
- **DuplicatedNoNeed**: For gradient-only computation (optimization)

Understanding these annotations is key to using Rust's autodiff effectively.
