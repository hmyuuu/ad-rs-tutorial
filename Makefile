.PHONY: build test docs docs-serve lint clean fmt check

# Toolchain with Enzyme support
TOOLCHAIN ?= +enzyme

# Build all examples
build:
	RUSTFLAGS="-Z autodiff=Enable" cargo $(TOOLCHAIN) build --workspace

# Run all examples
run-all:
	@for example in 01_scalar_square 02_scalar_sin 03_multi_variable 04_rosenbrock \
		05_vector_dot 06_vector_norm 07_mse_loss 08_cross_entropy \
		09_linear_layer 10_forward_mode 11_activity_demo 12_control_flow \
		13_complex_function 14_quantum_control; do \
		echo "Running $$example..."; \
		RUSTFLAGS="-Z autodiff=Enable" cargo $(TOOLCHAIN) run -p $$(echo $$example | sed 's/^[0-9]*_//') || exit 1; \
		echo ""; \
	done

# Run tests
test:
	RUSTFLAGS="-Z autodiff=Enable" cargo $(TOOLCHAIN) test --workspace

# Build documentation
docs:
	mdbook build docs

# Serve documentation locally
docs-serve:
	mdbook serve docs --open

# Run clippy linter
lint:
	cargo clippy --workspace -- -D warnings

# Format code
fmt:
	cargo fmt --all

# Check formatting
check:
	cargo fmt --all -- --check
	cargo clippy --workspace -- -D warnings

# Clean build artifacts
clean:
	cargo clean
	rm -rf docs/book
