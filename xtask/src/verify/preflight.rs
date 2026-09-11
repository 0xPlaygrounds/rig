//! Cheap prerequisite checks for the selected commands, before compilation.
use super::*;
use std::{collections::BTreeSet, io::Write};

pub(super) fn version_matches(actual: &str, required: &str) -> bool {
    actual.split_whitespace().any(|v| v == required)
}

pub(super) fn run(root: &Path, plan: &[Check]) -> Result<()> {
    let mut probes: BTreeSet<(&str, Vec<&str>)> = BTreeSet::new();
    let mut wasm = false;
    let mut runner = false;
    for check in plan {
        for step in &check.steps {
            let first = step.args.first().map(String::as_str);
            match step.program.as_str() {
                "cargo" => {
                    probes.insert(("cargo", vec!["--version"]));
                    probes.insert(("rustc", vec!["--version"]));
                    match first {
                        Some("fmt") => {
                            probes.insert(("cargo", vec!["fmt", "--version"]));
                        }
                        Some("clippy") => {
                            probes.insert(("cargo", vec!["clippy", "--version"]));
                        }
                        Some("nextest") => {
                            probes.insert(("cargo", vec!["nextest", "--version"]));
                        }
                        _ => {}
                    }
                }
                "@registrations" => {
                    probes.insert(("cargo", vec!["nextest", "--version"]));
                }
                "@native-only" => {
                    wasm = true;
                }
                p if p.starts_with('@') => {}
                p => {
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
            "workspace-check",
            "full-tests",
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
        println!("PREFLIGHT {program} {args:?}");
        std::io::stdout().flush()?;
        match output(root, program, &args) {
            Ok(version) => {
                if program == "python3" {
                    let parts: Vec<_> = version
                        .trim()
                        .strip_prefix("Python ")
                        .unwrap_or("")
                        .split('.')
                        .collect();
                    let major = parts
                        .first()
                        .and_then(|v| v.parse::<u32>().ok())
                        .unwrap_or(0);
                    let minor = parts
                        .get(1)
                        .and_then(|v| v.parse::<u32>().ok())
                        .unwrap_or(0);
                    if (major, minor) < (3, 10) {
                        failures.push(format!("Python >=3.10 required; got {}", version.trim()));
                    }
                }
                if program == "rustc" {
                    let pin = std::fs::read_to_string(root.join("rust-toolchain.toml"))?;
                    if let Some(channel) = pin.lines().find_map(|l| {
                        l.trim()
                            .strip_prefix("channel = ")
                            .map(|v| v.trim_matches('"'))
                    }) && !version_matches(&version, channel)
                    {
                        failures.push(format!(
                            "rustc must match repository toolchain {channel}; got {}",
                            version.trim()
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
                    if !version_matches(&version, required) {
                        failures.push(format!(
                            "wasm-bindgen-test-runner must match Cargo.lock {required}; got {}",
                            version.trim()
                        ));
                    }
                }
                println!("  available: {}", version.trim());
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
