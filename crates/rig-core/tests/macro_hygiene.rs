use std::{path::PathBuf, process::Command};

#[test]
fn completion_parent_macro_uses_only_rig_core_as_a_direct_dependency()
-> Result<(), Box<dyn std::error::Error>> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let fixture = manifest_dir.join("tests/fixtures/telemetry_macro_consumer/Cargo.toml");
    let target_dir = manifest_dir.join("../../target/rig-core-telemetry-macro-fixture");

    let output = Command::new(env!("CARGO"))
        .arg("check")
        .arg("--quiet")
        .arg("--manifest-path")
        .arg(&fixture)
        .arg("--target-dir")
        .arg(&target_dir)
        .output()?;

    if !output.status.success() {
        return Err(format!(
            "telemetry macro dependency-hygiene fixture failed:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        )
        .into());
    }

    Ok(())
}
