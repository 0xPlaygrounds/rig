ci:
    just fmt
    just clippy

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
