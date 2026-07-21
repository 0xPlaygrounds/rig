use std::{path::PathBuf, process::Command};

#[test]
fn generated_paths_follow_cargo_dependency_renames() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let target_dir = manifest_dir.join("../../target/rig-derive-rename-fixtures");

    for fixture in ["core_renamed", "agent_renamed", "facade_renamed"] {
        let manifest = manifest_dir
            .join("tests/fixtures")
            .join(fixture)
            .join("Cargo.toml");
        let output = Command::new(env!("CARGO"))
            .args(["check", "--quiet", "--manifest-path"])
            .arg(&manifest)
            .arg("--target-dir")
            .arg(&target_dir)
            .output()?;

        if !output.status.success() {
            return Err(format!(
                "dependency-rename fixture `{fixture}` failed:\n{}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr),
            )
            .into());
        }
    }

    Ok(())
}
