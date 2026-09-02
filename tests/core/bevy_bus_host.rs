//! The Bevy host fixture: the bound claims proven in a running Bevy world.
//!
//! `tests/fixtures/bevy_bus_host` is its own workspace (bevy stays out of the
//! main lock file) depending on rig-core only on the rig side and on
//! `bevy_ecs` + `bevy_tasks` only on the host side, pinned by git revision.
//! This test runs the fixture binary and checks that graph.

use std::{path::PathBuf, process::Command};

fn fixture_manifest() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/bevy_bus_host/Cargo.toml")
}

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/bevy_bus_host")
}

fn target_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/bevy-bus-host-fixture")
}

#[test]
fn bevy_host_runs_the_six_proofs() -> Result<(), Box<dyn std::error::Error>> {
    let output = Command::new("cargo")
        .current_dir(fixture_dir())
        .env_remove("RUSTUP_TOOLCHAIN")
        .env_remove("CARGO")
        .args(["run", "--quiet", "--manifest-path"])
        .arg(fixture_manifest())
        .arg("--target-dir")
        .arg(target_dir())
        .output()?;
    if !output.status.success() {
        return Err(format!(
            "bevy bus host fixture failed:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        )
        .into());
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    if !stdout.contains("bevy-bus-host: ok") {
        return Err(format!("fixture did not report success:\n{stdout}").into());
    }
    Ok(())
}

#[test]
fn bevy_host_fixture_graph_is_rig_core_and_bevy_ecs_tasks_only()
-> Result<(), Box<dyn std::error::Error>> {
    let output = Command::new("cargo")
        .current_dir(fixture_dir())
        .env_remove("RUSTUP_TOOLCHAIN")
        .env_remove("CARGO")
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
    let packages = metadata["packages"].as_array().cloned().unwrap_or_default();
    let names: Vec<&str> = packages
        .iter()
        .filter_map(|package| package["name"].as_str())
        .collect();
    for required in ["rig-core", "bevy_ecs", "bevy_tasks"] {
        if !names.contains(&required) {
            return Err(format!("fixture graph should contain `{required}`").into());
        }
    }
    for forbidden in [
        "rig",
        "rig-agent",
        "tokio",
        "reqwest",
        "rmcp",
        "bevy",
        "bevy_app",
    ] {
        if names.contains(&forbidden) {
            return Err(format!("`{forbidden}` reached the bevy bus host fixture's graph").into());
        }
    }
    // The rig side of the graph is rig-core alone.
    let rig_side: Vec<&str> = names
        .iter()
        .copied()
        .filter(|name| name.starts_with("rig") && *name != "rig-bevy-bus-host-fixture")
        .collect();
    if rig_side != ["rig-core"] {
        return Err(
            format!("rig-side dependencies must be rig-core only, got {rig_side:?}").into(),
        );
    }
    // Bevy is pinned by revision, not version.
    let manifest = std::fs::read_to_string(fixture_manifest())?;
    for line in manifest.lines().filter(|line| line.starts_with("bevy_")) {
        if !line.contains("rev = \"823bcc935\"") {
            return Err(format!("bevy dependency is not pinned by rev: {line}").into());
        }
    }
    Ok(())
}
