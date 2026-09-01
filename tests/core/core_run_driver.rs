//! A downstream crate depending on `rig-core` **only** can drive a full agent
//! turn from the run vocabulary: erase a model into a `ModelHandle`, build a
//! `ToolSet` / `ToolCatalog` from `PortableDynamicTool`s, describe the run
//! with a `RunSpec`, call `prepare_request` for each model call, dispatch a
//! tool by name, thread the result back with the transcript helpers and
//! validate the transcript — the seams a systems driver (an ECS plugin) needs,
//! with neither `rig-agent` nor `AgentRun` in its graph.
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
fn core_alone_drives_a_tool_turn_without_rig_agent() -> Result<(), Box<dyn std::error::Error>> {
    let output = Command::new(env!("CARGO"))
        .args(["run", "--quiet", "--manifest-path"])
        .arg(fixture_manifest())
        .arg("--target-dir")
        .arg(target_dir())
        .output()?;
    if !output.status.success() {
        return Err(format!(
            "core driver fixture failed:\n{}\n{}",
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
fn core_run_driver_fixture_graph_is_rig_core_only() -> Result<(), Box<dyn std::error::Error>> {
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
    if !names.contains(&"rig-core") {
        return Err("fixture graph should contain rig-core".into());
    }
    for forbidden in ["rig-agent", "rig-run", "rig", "tokio", "reqwest"] {
        if names.contains(&forbidden) {
            return Err(format!(
                "`{forbidden}` reached the core driver fixture's dependency graph"
            )
            .into());
        }
    }
    Ok(())
}
