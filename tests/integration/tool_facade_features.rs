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
#[ignore = "a nested Cargo build: cargo xtask verify --check guard-facade-features"]
fn portable_tool_facade_is_feature_additive() -> Result<(), Box<dyn std::error::Error>> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let fixture = root.join("tests/fixtures/tool_facade/Cargo.toml");
    let target_dir = root.join("target/tool-facade-fixture");

    // This checks feature additivity against the workspace's dependency set,
    // not a fresh registry resolution. Cargo adapts the copied lockfile to the
    // standalone fixture while retaining the workspace's compatible versions.
    std::fs::copy(
        root.join("Cargo.lock"),
        fixture.with_file_name("Cargo.lock"),
    )?;

    for (name, args) in [
        ("core-only", &["--no-default-features"][..]),
        (
            "agent",
            &["--no-default-features", "--features", "agent"][..],
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
