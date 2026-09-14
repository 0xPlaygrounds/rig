//! rig-ecs inherits Bevy requirements from the workspace.
//! The tooling check validates crates.io sources through Cargo metadata.

use std::path::PathBuf;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// rig-ecs's manifest takes every Bevy crate from the workspace table, so
/// requirements are declared once.
#[test]
fn rig_ecs_takes_bevy_from_the_workspace() -> Result<(), Box<dyn std::error::Error>> {
    let manifest = std::fs::read_to_string(workspace_root().join("crates/rig-ecs/Cargo.toml"))?;
    for line in manifest.lines().map(str::trim) {
        if line.starts_with("bevy_") && !line.contains("workspace = true") {
            return Err(format!("rig-ecs pins Bevy itself: `{line}`").into());
        }
    }
    Ok(())
}
