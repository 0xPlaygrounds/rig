use std::{env, fs, io, path::PathBuf};

const ARTIFACTS: [&str; 3] = ["config.json", "tokenizer.json", "model.gguf"];

fn main() -> io::Result<()> {
    let manifest_dir =
        PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").ok_or_else(|| {
            io::Error::new(io::ErrorKind::NotFound, "CARGO_MANIFEST_DIR is not set")
        })?);
    let output_dir = PathBuf::from(
        env::var_os("OUT_DIR")
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "OUT_DIR is not set"))?,
    );
    let model_dir = env::var_os("MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| manifest_dir.join("model"));
    let complete_model = ARTIFACTS
        .iter()
        .all(|artifact| model_dir.join(artifact).is_file());

    println!("cargo:rerun-if-env-changed=MODEL_DIR");
    println!("cargo:rerun-if-changed={}", model_dir.display());
    for artifact in ARTIFACTS {
        let source = model_dir.join(artifact);
        let destination = output_dir.join(artifact);
        println!("cargo:rerun-if-changed={}", source.display());
        if complete_model {
            fs::copy(source, destination)?;
        } else {
            fs::write(destination, [])?;
        }
    }

    println!(
        "cargo:rustc-env=RIG_CANDLE_WASM_MODEL_EMBEDDED={}",
        u8::from(complete_model)
    );
    Ok(())
}
