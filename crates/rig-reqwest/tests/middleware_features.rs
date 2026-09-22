#![cfg(not(target_family = "wasm"))]

use std::{path::PathBuf, process::Command};

#[test]
fn middleware_clients_are_available_for_each_tls_selector() -> anyhow::Result<()> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let fixture = manifest_dir.join("tests/fixtures/middleware_features");
    let target = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| manifest_dir.join("../../target"))
        .join("reqwest-middleware-feature-fixture");

    // Reuse workspace versions but resolve each selector outside its unified feature graph.
    std::fs::copy(
        manifest_dir.join("../../Cargo.lock"),
        fixture.join("Cargo.lock"),
    )?;
    for selector in ["native-tls", "rustls", "generic"] {
        // Metadata can list inactive weak optional dependencies; inspect the active tree.
        let tree = Command::new(env!("CARGO"))
            .current_dir(&fixture)
            .args([
                "tree",
                "--edges",
                "normal,build",
                "--prefix",
                "none",
                "--format",
                "{p}",
                "--no-default-features",
                "--features",
                selector,
            ])
            .output()?;
        anyhow::ensure!(
            tree.status.success(),
            "{selector} dependency tree failed: {}",
            String::from_utf8_lossy(&tree.stderr)
        );
        let unwanted = if selector == "native-tls" {
            "rustls"
        } else {
            "native-tls"
        };
        anyhow::ensure!(
            !String::from_utf8_lossy(&tree.stdout)
                .lines()
                .any(|line| line.starts_with(&format!("{unwanted} v"))),
            "{selector} unexpectedly activated {unwanted}"
        );

        let output = Command::new(env!("CARGO"))
            .current_dir(&fixture)
            .args([
                "run",
                "--locked",
                "--quiet",
                "--no-default-features",
                "--features",
                selector,
            ])
            .arg("--target-dir")
            .arg(&target)
            .output()?;
        anyhow::ensure!(
            output.status.success(),
            "middleware fixture {selector} failed:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    Ok(())
}
