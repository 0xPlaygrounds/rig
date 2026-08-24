use std::{path::PathBuf, process::Command};

fn check_fixture(fixture: &str) -> Result<std::process::Output, Box<dyn std::error::Error>> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let target_dir = manifest_dir.join("../../target/rig-derive-rename-fixtures");

    let manifest = manifest_dir
        .join("tests/fixtures")
        .join(fixture)
        .join("Cargo.toml");
    Ok(Command::new(env!("CARGO"))
        .args(["check", "--quiet", "--manifest-path"])
        .arg(&manifest)
        .arg("--target-dir")
        .arg(&target_dir)
        .output()?)
}

/// The fixture manifests deliberately have no direct `serde`/`serde_json`
/// dependency: compiling them also proves the generated code resolves those
/// through Rig's re-exports instead of the caller's Cargo.toml.
#[test]
fn generated_paths_follow_cargo_dependency_renames() -> Result<(), Box<dyn std::error::Error>> {
    for fixture in ["core_renamed", "agent_renamed", "facade_renamed"] {
        let output = check_fixture(fixture)?;

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

/// A contextual tool compiles with `rig-core` alone (here under a Cargo
/// rename): `Tool`/`ToolContext` are rig-core's, so the macro never needs
/// `rig`/`rig-agent` to be reachable.
#[test]
fn contextual_tool_compiles_with_rig_core_only() -> Result<(), Box<dyn std::error::Error>> {
    let output = check_fixture("core_only_contextual")?;

    if !output.status.success() {
        return Err(format!(
            "fixture `core_only_contextual` failed:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        )
        .into());
    }

    Ok(())
}
