# AXI Runtime

A Rust workspace for AXI (Advanced eXtensible Interface) runtime components.

## CI Status

[![Rust Tests](https://github.com/gaurshalendra2412/axi-runtime/actions/workflows/rust.yml/badge.svg)](https://github.com/gaurshalendra2412/axi-runtime/actions/workflows/rust.yml)
[![Python Tests](https://github.com/gaurshalendra2412/axi-runtime/actions/workflows/python.yml/badge.svg)](https://github.com/gaurshalendra2412/axi-runtime/actions/workflows/python.yml)

## Project Structure

```text
crates/
  ├── axi-ir/      # Intermediate representation
  ├── axi-dpo/     # Data path optimization
  └── axi-cache/   # Cache management
python/            # Python bindings/tools

cargo build --verbose
cargo test --workspace --verbose



