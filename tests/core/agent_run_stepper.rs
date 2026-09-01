//! `AgentRun` is hostable without rig-agent's futures driver: a downstream
//! crate depending on `rig-agent` (default features off) and `rig-core` can
//! build a run from a `RunSpec`, step it (`next_step` / `model_response` /
//! `tool_results`), prepare each request and dispatch tools by name — with no
//! async runtime, transport, MCP or `rig` facade in its graph. This is the
//! seam a second driver (an ECS schedule, a job system) that keeps `AgentRun`
//! as the loop needs.
//!
//! The fixture at `tests/fixtures/agent_run_stepper` is *run*, not just
//! checked; its dependency graph is read back through `cargo metadata` so the
//! guard fails the day a runtime creeps into rig-agent's default-off graph.

use std::{path::PathBuf, process::Command};

fn fixture_manifest() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/agent_run_stepper/Cargo.toml")
}

fn target_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/agent-run-stepper-fixture")
}

#[test]
fn agent_run_is_steppable_without_the_futures_driver() -> Result<(), Box<dyn std::error::Error>> {
    let output = Command::new(env!("CARGO"))
        .args(["run", "--quiet", "--manifest-path"])
        .arg(fixture_manifest())
        .arg("--target-dir")
        .arg(target_dir())
        .output()?;
    if !output.status.success() {
        return Err(format!(
            "agent-run stepper fixture failed:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        )
        .into());
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    if !stdout.contains("agent-run-stepper: ok") {
        return Err(format!("fixture did not report success:\n{stdout}").into());
    }
    Ok(())
}

#[test]
fn agent_run_stepper_fixture_graph_is_runtime_free() -> Result<(), Box<dyn std::error::Error>> {
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
    if !(names.contains(&"rig-core") && names.contains(&"rig-agent")) {
        return Err("fixture graph should contain rig-core and rig-agent".into());
    }
    for forbidden in ["rig", "tokio", "reqwest", "rmcp"] {
        if names.contains(&forbidden) {
            return Err(format!(
                "`{forbidden}` reached the agent-run stepper fixture's dependency graph"
            )
            .into());
        }
    }
    Ok(())
}
