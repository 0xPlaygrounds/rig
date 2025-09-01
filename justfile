ci:
    just fmt
    just clippy

example example_name:
    RUST_LOG=trace cargo run --example {{example_name}}

clippy:
    cargo clippy --all-features --all-targets

fmt:
    cargo fmt -- --check

# requires wabt, binaryen and wasm-bindgen-cli to be installed
# all of which can be installed as linux packages
build-wasm:
    cargo build -p rig-wasm --release --target wasm32-unknown-unknown
    wasm-bindgen \
        --target experimental-nodejs-module \
        --out-dir rig-wasm/pkg/src/generated \
        target/wasm32-unknown-unknown/release/rig_wasm.wasm

# build-wasm-full
bwf:
    just build-wasm && npm run build --prefix ./rig-wasm/pkg

# Runs a command that compiles the docs then opens it as if it were the official docs on Docs.rs
# Requires nightly toolchain
doc:
    RUSTDOCFLAGS="--cfg docsrs" cargo +nightly doc --package rig-core --all-features --open
