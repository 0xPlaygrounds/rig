use std::{path::PathBuf, process::Command};

fn cargo_check(
    manifest: &std::path::Path,
    target_dir: &std::path::Path,
    args: &[&str],
) -> Result<std::process::Output, Box<dyn std::error::Error>> {
    Ok(Command::new(env!("CARGO"))
        .arg("check")
        .arg("--quiet")
        .arg("--manifest-path")
        .arg(manifest)
        .arg("--target-dir")
        .arg(target_dir)
        .args(args)
        .output()?)
}

#[test]
#[cfg_attr(
    not(feature = "facade-build-tests"),
    ignore = "slow nested cargo check; run with --features facade-build-tests or --all-features"
)]
fn portable_tool_facade_is_feature_additive() -> Result<(), Box<dyn std::error::Error>> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let fixture = root.join("tests/fixtures/tool_facade/Cargo.toml");
    let target_dir = root.join("target/tool-facade-fixture");

    for (name, args) in [
        ("core-only", &["--no-default-features"][..]),
        (
            "agent",
            &["--no-default-features", "--features", "agent"][..],
        ),
        (
            "agent-ecs",
            &["--no-default-features", "--features", "agent,ecs"][..],
        ),
        ("default", &[][..]),
        ("all-features", &["--all-features"][..]),
    ] {
        let output = cargo_check(&fixture, &target_dir, args)?;
        if !output.status.success() {
            return Err(format!(
                "tool facade fixture `{name}` failed:\n{}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr),
            )
            .into());
        }
    }

    Ok(())
}

#[test]
#[cfg_attr(
    not(feature = "facade-build-tests"),
    ignore = "slow nested cargo check; run with --features facade-build-tests or --all-features"
)]
fn contextual_classic_tool_is_rejected_by_ecs() -> Result<(), Box<dyn std::error::Error>> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let fixture = root.join("tests/fixtures/tool_facade/contextual_ecs/Cargo.toml");
    let target_dir = root.join("target/tool-facade-contextual-rejection");
    let output = cargo_check(&fixture, &target_dir, &[])?;

    if output.status.success() {
        return Err(
            "contextual classic tool unexpectedly satisfied the ECS runtime.s portable bound"
                .into(),
        );
    }

    let stderr = String::from_utf8_lossy(&output.stderr);
    if !stderr.contains("PortableTool") || !stderr.contains("ContextualClassicTool") {
        return Err(format!("unexpected compiler diagnostic:\n{stderr}").into());
    }

    Ok(())
}
