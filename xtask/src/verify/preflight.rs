//! Cheap prerequisite probes before any compilation: a missing tool is a
//! failed run, never a pass.
use super::*;
use std::collections::BTreeSet;

pub(super) fn version_matches(actual: &str, required: &str) -> bool {
    actual.split_whitespace().any(|v| v == required)
}

pub(super) fn run(root: &Path, plan: &[Check]) -> Result<()> {
    let mut probes: BTreeSet<(&str, Vec<&str>)> = BTreeSet::new();
    let mut wasm = false;
    let mut runner = false;
    for check in plan {
        for step in &check.steps {
            match (step.program.as_str(), step.args.first().map(String::as_str)) {
                ("cargo", first) => {
                    probes.insert(("cargo", vec!["--version"]));
                    probes.insert(("rustc", vec!["--version"]));
                    if let Some(tool @ ("fmt" | "clippy" | "nextest")) = first {
                        probes.insert(("cargo", vec![tool, "--version"]));
                    }
                }
                ("@registrations", _) => {
                    probes.insert(("cargo", vec!["nextest", "--version"]));
                }
                ("@native-only", _) => wasm = true,
                (p, _) if p.starts_with('@') => {}
                (p, _) => {
                    probes.insert((p, vec!["--version"]));
                }
            }
            wasm |= step.args.iter().any(|a| a == "wasm32-unknown-unknown");
            runner |= step
                .env
                .contains_key("CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUNNER");
        }
        if [
            "clippy",
            "doctests",
            "docs",
            "full-tests",
            "full-test-build",
            "dependency-floors",
        ]
        .contains(&check.id.as_str())
        {
            probes.insert(("protoc", vec!["--version"]));
        }
        if check.id == "full-tests" {
            probes.insert(("docker", vec!["info", "--format", "{{.ServerVersion}}"]));
        }
    }
    if runner {
        probes.insert(("node", vec!["--version"]));
        probes.insert(("wasm-bindgen-test-runner", vec!["--version"]));
    }
    let mut failures = Vec::new();
    for (program, args) in probes {
        match output(root, program, &args) {
            Ok(version) => {
                let version = version.trim();
                if program == "python3" {
                    let mut parts = version
                        .strip_prefix("Python ")
                        .unwrap_or("")
                        .split('.')
                        .map(|v| v.parse::<u32>().unwrap_or(0));
                    let (major, minor) = (parts.next().unwrap_or(0), parts.next().unwrap_or(0));
                    if (major, minor) < (3, 11) {
                        failures.push(format!("Python >=3.11 required; got {version}"));
                    }
                }
                if program == "rustc" {
                    let pin = std::fs::read_to_string(root.join("rust-toolchain.toml"))?;
                    if let Some(channel) = pin.lines().find_map(|l| {
                        l.trim()
                            .strip_prefix("channel = ")
                            .map(|v| v.trim_matches('"'))
                    }) && !version_matches(version, channel)
                    {
                        failures.push(format!(
                            "rustc must match repository toolchain {channel}; got {version}"
                        ));
                    }
                }
                if program == "wasm-bindgen-test-runner" {
                    let lock = std::fs::read_to_string(root.join("Cargo.lock"))?;
                    let required = lock
                        .split("[[package]]")
                        .find(|p| p.lines().any(|l| l == "name = \"wasm-bindgen\""))
                        .and_then(|p| {
                            p.lines().find_map(|l| {
                                l.strip_prefix("version = \"")
                                    .and_then(|v| v.strip_suffix('"'))
                            })
                        })
                        .ok_or_else(|| invalid("missing wasm-bindgen lockfile version"))?;
                    if !version_matches(version, required) {
                        failures.push(format!(
                            "wasm-bindgen-test-runner must match Cargo.lock {required}; got {version}"
                        ));
                    }
                }
            }
            Err(e) => failures.push(format!("{program} {args:?}: {e}")),
        }
    }
    if wasm {
        match output(
            root,
            "rustc",
            &[
                "--print",
                "target-libdir",
                "--target",
                "wasm32-unknown-unknown",
            ],
        ) {
            Ok(dir) if Path::new(dir.trim()).is_dir() => {}
            other => failures.push(format!(
                "install wasm32-unknown-unknown for the repository toolchain: {other:?}"
            )),
        }
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(invalid(format!(
            "missing or incompatible prerequisites (no checks executed):\n{}",
            failures.join("\n")
        )))
    }
}
