//! Cheap prerequisite checks for the selected commands, before compilation.
use super::*;
use std::{collections::BTreeSet, io::Write};

pub(super) fn version_matches(actual: &str, required: &str) -> bool {
    actual.split_whitespace().any(|v| v == required)
}

pub(super) fn uses_runtime_model(check: &Check) -> bool {
    // These commands execute rig-fastembed's model-loading doctest. Nextest
    // executes neither doctests nor example main functions.
    matches!(check.id.as_str(), "doctests" | "package-rig-fastembed")
}

pub(super) fn configure_model_cache(metadata: &Value, plan: &mut [Check]) -> Result<()> {
    let target = metadata["target_directory"]
        .as_str()
        .ok_or_else(|| invalid("Cargo metadata missing target directory"))?;
    let cache = Path::new(target).join("verify/fastembed-cache");
    let cache = cache
        .to_str()
        .ok_or_else(|| invalid("non-UTF-8 model cache path"))?;
    for check in plan.iter_mut().filter(|c| uses_runtime_model(c)) {
        for step in &mut check.steps {
            // fastembed 4.9.1 gives HF_HOME precedence over FASTEMBED_CACHE_DIR.
            // Set both explicitly; this configuration is displayed/fingerprinted.
            step.env.insert("FASTEMBED_CACHE_DIR".into(), cache.into());
            step.env.insert("HF_HOME".into(), cache.into());
        }
    }
    Ok(())
}

pub(super) fn prepare_model_cache(root: &Path, plan: &[Check]) -> Result<()> {
    let Some(cache) = plan
        .iter()
        .filter(|c| uses_runtime_model(c))
        .flat_map(|c| &c.steps)
        .find_map(|s| s.env.get("FASTEMBED_CACHE_DIR"))
    else {
        return Ok(());
    };
    let relative = "crates/rig-fastembed/.fastembed_cache";
    let source = root.join(relative);
    match std::fs::symlink_metadata(&source) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(e.into()),
        Ok(meta) if !meta.is_dir() || meta.file_type().is_symlink() => {
            return Err(invalid(
                "cannot migrate a symlinked or non-directory fastembed cache",
            ));
        }
        Ok(_) => {}
    }
    for ancestor in source.ancestors().skip(1).take_while(|p| *p != root) {
        if std::fs::symlink_metadata(ancestor)?
            .file_type()
            .is_symlink()
        {
            return Err(invalid(
                "cannot migrate a cache through a symlinked directory",
            ));
        }
    }
    if !output(root, "git", &["ls-files", "--", relative])?
        .trim()
        .is_empty()
        || !process::capture(root, "git", &["check-ignore", "-q", relative])?
            .status
            .success()
    {
        return Err(invalid(
            "cannot migrate a tracked or non-ignored fastembed cache",
        ));
    }
    let canonical = source.canonicalize()?;
    let mut pending = vec![source.clone()];
    while let Some(path) = pending.pop() {
        let meta = std::fs::symlink_metadata(&path)?;
        if meta.file_type().is_symlink() {
            if std::fs::read_link(&path)?.is_absolute()
                || !path.canonicalize()?.starts_with(&canonical)
            {
                return Err(invalid(format!(
                    "model cache link would change meaning after migration: {}",
                    path.display()
                )));
            }
        } else if meta.is_dir() {
            for entry in std::fs::read_dir(path)? {
                pending.push(entry?.path());
            }
        } else if !meta.is_file() {
            return Err(invalid("unsupported model cache entry"));
        }
    }
    let destination = Path::new(cache);
    if std::fs::symlink_metadata(destination).is_ok() {
        return Err(invalid(format!(
            "both model caches exist; preserve and reconcile {} and {} before continuing",
            source.display(),
            destination.display()
        )));
    }
    std::fs::create_dir_all(
        destination
            .parent()
            .ok_or_else(|| invalid("model cache has no parent"))?,
    )?;
    println!(
        "PREPARE move downloaded model cache {} -> {}; preserve blobs and relative links before fingerprints",
        source.display(),
        destination.display()
    );
    // Cross-device destinations fail safely; never delete downloads or follow links.
    std::fs::rename(source, destination)?;
    Ok(())
}

// These integration tests launch Cargo in independent workspaces. Resolve their
// ignored lockfiles before taking any check fingerprints, not halfway through
// the expensive plan. Keep the resulting locks covered by normal input hashing.
pub(super) fn fixture_manifests(plan: &[Check]) -> BTreeSet<&'static str> {
    let mut fixtures = BTreeSet::new();
    for check in plan {
        match check.id.as_str() {
            "macro-hygiene" | "package-rig-core" | "full-tests" => {
                fixtures
                    .insert("crates/rig-core/tests/fixtures/telemetry_macro_consumer/Cargo.toml");
            }
            _ => {}
        }
        if matches!(check.id.as_str(), "derive" | "package-rig-derive") {
            fixtures.extend([
                "crates/rig-derive/tests/fixtures/core_renamed/Cargo.toml",
                "crates/rig-derive/tests/fixtures/agent_renamed/Cargo.toml",
                "crates/rig-derive/tests/fixtures/facade_renamed/Cargo.toml",
                "crates/rig-derive/tests/fixtures/core_only_contextual/Cargo.toml",
            ]);
        }
        if matches!(
            check.id.as_str(),
            "full-tests" | "package-rig" | "provider-tool_facade_features"
        ) {
            fixtures.insert("tests/fixtures/tool_facade/Cargo.toml");
        }
    }
    fixtures
}

// Only these exact generated inputs have proven, exclusive test ownership.
// Unknown fixture assets still take the normal conservative selection path.
pub(super) fn fixture_lock_owner(path: &str) -> Option<&'static str> {
    let manifest = path.strip_suffix("Cargo.lock")?.to_owned() + "Cargo.toml";
    for id in ["macro-hygiene", "derive", "provider-tool_facade_features"] {
        let check = Check {
            id: id.into(),
            reason: String::new(),
            steps: vec![],
        };
        if fixture_manifests(&[check]).contains(manifest.as_str()) {
            return Some(id);
        }
    }
    None
}

pub(super) fn prepare_fixtures(root: &Path, plan: &[Check]) -> Result<()> {
    for manifest in fixture_manifests(plan) {
        let path = root.join(manifest);
        // Do not let preparation follow a fixture or lockfile out of the tree.
        for input in [path.clone(), path.with_file_name("Cargo.lock")] {
            for ancestor in input.ancestors().take_while(|p| *p != root) {
                match std::fs::symlink_metadata(ancestor) {
                    Ok(meta) if meta.file_type().is_symlink() => {
                        return Err(invalid(format!(
                            "cannot prepare symlinked fixture input: {}",
                            ancestor.display()
                        )));
                    }
                    Ok(_) => {}
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                    Err(e) => return Err(e.into()),
                }
            }
        }
        println!(
            "PREPARE cargo metadata --format-version 1 --manifest-path {manifest}: resolve fixture lockfile before verification inputs freeze (no compilation)"
        );
        let started = std::time::Instant::now();
        // Unlike generate-lockfile, metadata retains an existing valid lock.
        // --no-deps would bypass resolution and leave missing locks unprepared.
        output(
            root,
            "cargo",
            &[
                "metadata",
                "--format-version",
                "1",
                "--manifest-path",
                manifest,
            ],
        )?;
        println!(
            "PREPARED {manifest}: {:.3}s measured",
            started.elapsed().as_secs_f64()
        );
    }
    Ok(())
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
                    if (major, minor) < (3, 11) {
                        failures.push(format!("Python >=3.11 required; got {}", version.trim()));
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
