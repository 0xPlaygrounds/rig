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

/// A contextual tool without `rig`/`rig-agent` reachable must fail with the
/// macro's targeted diagnostic, not an unresolved `::rig_agent` path error.
#[test]
fn contextual_tool_without_runtime_crate_gets_a_targeted_error()
-> Result<(), Box<dyn std::error::Error>> {
    let output = check_fixture("core_only_contextual")?;

    if output.status.success() {
        return Err("fixture `core_only_contextual` unexpectedly compiled".into());
    }

    let stderr = String::from_utf8_lossy(&output.stderr);
    if !stderr.contains(
        "contextual tools (`&mut ToolContext`) require a dependency on `rig` or `rig-agent`",
    ) {
        return Err(format!(
            "fixture `core_only_contextual` failed without the targeted diagnostic:\n{stderr}"
        )
        .into());
    }

    Ok(())
}
