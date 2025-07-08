ci:
    just fmt
    just clippy

clippy:
    cargo clippy --all-features --all-targets

fmt:
    cargo fmt -- --check

build-wasm:
    cargo build -p rig-wasm --release --target wasm32-unknown-unknown
    wasm-bindgen \
        --target nodejs \
        --out-dir rig-wasm/pkg/generated \
        target/wasm32-unknown-unknown/release/rig_wasm.wasm
