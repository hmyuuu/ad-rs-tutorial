# Reverse Mode

Reverse mode automatic differentiation is the workhorse of modern machine learning. It computes gradients of a scalar output with respect to all inputs in a single backward pass.

In this chapter, you'll learn:

- How to use `#[autodiff_reverse]` (or `Reverse` mode)
- Computing gradients of multi-variable functions
- Implementing gradient descent optimization

Reverse mode is ideal when you have many inputs and a single scalar output — exactly the situation in training neural networks.
