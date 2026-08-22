//! A downstream crate depending on `rig-core` + `rig-run` only can drive a
//! full agent turn: erase a model into a `ModelHandle`, build a `ToolSet` /
//! `ToolCatalog` from `PortableDynamicTool`s, construct an `AgentRun` from a
//! `RunSpec`, call `prepare_request`, and dispatch a tool by name — the seams a
//! systems driver (an ECS plugin) needs, with no `rig-agent` in its graph.
//!
//! The fixture at `tests/fixtures/core_run_driver` is *run*, not just checked,
//! so the seams are exercised end to end; its dependency graph is read back
//! through `cargo metadata` so the guard fails the day rig-agent creeps in.

use std::{path::PathBuf, process::Command};

fn fixture_manifest() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/core_run_driver/Cargo.toml")
}

fn target_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/core-run-driver-fixture")
}

#[test]
fn core_plus_run_drives_a_tool_turn_without_rig_agent() -> Result<(), Box<dyn std::error::Error>> {
    let output = Command::new(env!("CARGO"))
        .args(["run", "--quiet", "--manifest-path"])
        .arg(fixture_manifest())
        .arg("--target-dir")
        .arg(target_dir())
        .output()?;
    if !output.status.success() {
        return Err(format!(
            "core+run driver fixture failed:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        )
        .into());
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    if !stdout.contains("core-run-driver: ok") {
        return Err(format!("fixture did not report success:\n{stdout}").into());
    }
    Ok(())
}

#[test]
fn core_run_driver_fixture_graph_has_no_rig_agent() -> Result<(), Box<dyn std::error::Error>> {
    let output = Command::new(env!("CARGO"))
        .args(["metadata", "--format-version", "1", "--manifest-path"])
        .arg(fixture_manifest())
        .output()?;
    if !output.status.success() {
        return Err(format!(
            "cargo metadata failed:\n{}",
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }
    let metadata: serde_json::Value = serde_json::from_slice(&output.stdout)?;
    let names: Vec<&str> = metadata["packages"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|package| package["name"].as_str())
        .collect();
    if !(names.contains(&"rig-core") && names.contains(&"rig-run")) {
        return Err("fixture graph should contain rig-core and rig-run".into());
    }
    for forbidden in ["rig-agent", "rig", "tokio", "reqwest"] {
        if names.contains(&forbidden) {
            return Err(format!(
                "`{forbidden}` reached the core+run driver fixture's dependency graph"
            )
            .into());
        }
    }
    Ok(())
}
