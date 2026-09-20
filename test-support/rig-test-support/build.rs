//! Resolve source declarations once per build, not once per nextest process.
#[allow(dead_code)]
#[path = "src/scenario_registry.rs"]
mod discovery;
#[path = "src/provenance/manifest.rs"]
mod manifest;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR")?).join("../..");
    println!(
        "cargo:rerun-if-changed={}",
        manifest::Manifest::path(&root).display()
    );
    let manifest = discovery::load(&root).map_err(std::io::Error::other)?;
    for provider in &manifest.providers {
        println!(
            "cargo:rerun-if-changed={}",
            root.join("crates/rig-cassette")
                .join(&provider.source_dir)
                .display()
        );
    }
    let output = std::path::PathBuf::from(std::env::var("OUT_DIR")?).join("scenarios.json");
    std::fs::write(output, serde_json::to_vec(&manifest)?)?;
    Ok(())
}
