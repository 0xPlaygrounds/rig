ci:
    just fmt
    just clippy

clippy:
    cargo clippy --all-features --all-targets

fmt:
    cargo fmt -- --check
