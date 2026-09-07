use super::*;
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    io::Write,
    path::PathBuf,
    time::Instant,
};

struct PlannerLock(File);
impl Drop for PlannerLock {
    fn drop(&mut self) {
        // A concurrent fork can briefly inherit the open file description.
        // Closing our descriptor alone need not release its flock immediately.
        // Explicitly unlock on every return, including a failed check.
        let _ = self.0.unlock();
    }
}

fn tracked_inputs(root: &Path) -> Result<Vec<String>> {
    let mut files = std::collections::BTreeSet::new();
    for args in [
        vec!["ls-files", "-z"],
        vec!["ls-files", "--others", "--exclude-standard", "-z"],
    ] {
        files.extend(
            output(root, "git", &args)?
                .split('\0')
                .filter(|s| !s.is_empty())
                .map(str::to_owned),
        );
    }
    Ok(files.into_iter().collect())
}
pub(super) fn digest(root: &Path, files: &[String], config: &[u8]) -> Result<String> {
    let mut hash = Sha256::new();
    hash.update(b"rig-verify-local-v2\0");
    hash.update(config);
    let mut existing = Vec::new();
    for name in files {
        // The timing report describes measurements; no executable consumes it.
        // All other source, fixture, lock, config and documentation bytes count.
        if name == "DEVELOPING.md" {
            continue;
        }
        hash.update(name.len().to_le_bytes());
        hash.update(name.as_bytes());
        let path = root.join(name);
        match fs::symlink_metadata(&path) {
            Ok(meta) => {
                if meta.file_type().is_symlink() {
                    return Err(invalid(format!(
                        "cannot reuse verification across symlink input {name}; use --no-reuse"
                    )));
                }
                if !meta.is_file() {
                    return Err(invalid(format!("unsupported verification input {name}")));
                }
                existing.push(name.as_str());
                #[cfg(unix)]
                {
                    use std::os::unix::fs::PermissionsExt;
                    hash.update(meta.permissions().mode().to_le_bytes());
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => hash.update(b"<deleted>"),
            Err(e) => return Err(e.into()),
        }
    }
    // Git hashes the actual working bytes in optimized native code. Reading
    // every cassette through unoptimized sha2 added seconds to cheap checks.
    // --no-filters preserves byte-sensitive fixtures regardless of attributes;
    // the index/status cache is deliberately not trusted for content identity.
    for paths in existing.chunks(256) {
        let mut args = vec!["hash-object", "--no-filters", "--"];
        args.extend_from_slice(paths);
        hash.update(output(root, "git", &args)?);
    }
    Ok(format!("{:x}", hash.finalize()))
}
pub(super) fn config(root: &Path, check: &Check) -> Result<Vec<u8>> {
    let mut s = format!("{} {:?}\nroot={}\n", check.id, check.steps, root.display());
    // Hash, never print, environment values: test switches and toolchain flags
    // must invalidate reuse, and credentials must not enter a result file.
    for (key, value) in std::env::vars_os().collect::<BTreeMap<_, _>>() {
        s.push_str(&format!("{key:?}={value:?}\n"));
    }
    // Cargo searches ancestor configuration and CARGO_HOME, not just Git.
    let mut config_dirs: Vec<PathBuf> = root.ancestors().map(|p| p.join(".cargo")).collect();
    if let Some(home) = std::env::var_os("CARGO_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|p| PathBuf::from(p).join(".cargo")))
    {
        config_dirs.push(home);
    }
    for directory in config_dirs {
        for name in ["config", "config.toml"] {
            let path = directory.join(name);
            match fs::read(&path) {
                Ok(bytes) => {
                    s.push_str(&format!("{}:{:x}\n", path.display(), Sha256::digest(bytes)))
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => return Err(e.into()),
            }
        }
    }
    let mut nextest_configs = Vec::new();
    if let Some(home) = std::env::var_os("HOME").map(PathBuf::from) {
        nextest_configs.push(home.join(".config/nextest/config.toml"));
        nextest_configs.push(home.join("Library/Application Support/nextest/config.toml"));
    }
    if let Some(home) = std::env::var_os("XDG_CONFIG_HOME").map(PathBuf::from) {
        nextest_configs.push(home.join("nextest/config.toml"));
    }
    for path in nextest_configs {
        match fs::read(&path) {
            Ok(bytes) => s.push_str(&format!("{}:{:x}\n", path.display(), Sha256::digest(bytes))),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e.into()),
        }
    }
    for (program, args) in [
        ("rustc", vec!["-Vv"]),
        ("cargo", vec!["-V"]),
        ("git", vec!["--version"]),
        ("cargo", vec!["nextest", "--version"]),
        ("node", vec!["--version"]),
        ("wasm-bindgen-test-runner", vec!["--version"]),
    ] {
        let value = Command::new(program).args(&args).current_dir(root).output();
        s.push_str(&format!("{program} {args:?}: {value:?}\n"));
    }
    Ok(s.into_bytes())
}
pub(super) fn command(root: &Path, step: &Step) -> Command {
    let mut cmd = Command::new(&step.program);
    cmd.args(&step.args).current_dir(root).envs(&step.env);
    // Verification is always replay. CLI/environment retry overrides must not
    // defeat the no-retry telemetry contract in the guards profile.
    cmd.env("RIG_PROVIDER_TEST_MODE", "replay")
        .env_remove("RIG_REGENERATE_GOLDEN")
        .env_remove("NEXTEST_RETRIES");
    cmd
}
fn internal(root: &Path, directory: &Path, step: &Step) -> Result<()> {
    match step.program.as_str() {
        "@layout" => crate::test_layout::check(root).map_err(invalid),
        "@scenarios" => crate::scenarios::run(root, Vec::new()).map_err(|e| invalid(e.to_string())),
        "@registrations" => {
            let json = output(
                root,
                "cargo",
                &step.args.iter().map(String::as_str).collect::<Vec<_>>(),
            )?;
            let file = directory.join("registrations.json");
            fs::write(&file, json)?;
            crate::scenarios::run(root, vec![file.to_string_lossy().into_owned()])
                .map_err(|e| invalid(e.to_string()))
        }
        "@fixture-paths" => {
            for path in tracked_inputs(root)?.into_iter().filter(|p| {
                p.starts_with("crates/")
                    && p.ends_with(".rs")
                    && (p.contains("/src/") || p.contains("/tests/"))
            }) {
                let text = match fs::read_to_string(root.join(&path)) {
                    Ok(text) => text,
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
                    Err(e) => return Err(e.into()),
                };
                for line in text.lines().filter(|l| {
                    !l.trim_start().starts_with("///") && !l.trim_start().starts_with("//!")
                }) {
                    if line.contains("\"tests/data") || line.contains("\"./tests/data") {
                        return Err(invalid(format!(
                            "CWD-relative fixture path in {path}; anchor it to CARGO_MANIFEST_DIR"
                        )));
                    }
                }
            }
            Ok(())
        }
        "@native-only" => {
            let package = step
                .args
                .first()
                .ok_or_else(|| invalid("native-only package missing"))?;
            let expected = step
                .args
                .get(1)
                .ok_or_else(|| invalid("native-only diagnostic missing"))?;
            let result = Command::new("cargo")
                .args([
                    "check",
                    "--locked",
                    "--package",
                    package,
                    "--target",
                    "wasm32-unknown-unknown",
                ])
                .env("CARGO_TERM_COLOR", "never")
                .current_dir(root)
                .output()?;
            let stderr = String::from_utf8_lossy(&result.stderr);
            print!("{stderr}");
            let count = stderr
                .lines()
                .filter(|l| {
                    (l.starts_with("error:") || l.starts_with("error["))
                        && !l.contains("could not compile")
                })
                .count();
            if result.status.success() || !stderr.contains(expected) || count != 1 {
                return Err(invalid(format!(
                    "{package}: expected exactly one native-only diagnostic, got {count}"
                )));
            }
            Ok(())
        }
        _ => Err(invalid(format!(
            "unknown internal command {}",
            step.program
        ))),
    }
}
pub(super) fn run(root: &Path, metadata: &Value, opts: &Options, plan: &[Check]) -> Result<()> {
    if plan.is_empty() {
        println!("No working changes; no verification result claimed.");
        return Ok(());
    }
    let target = metadata["target_directory"]
        .as_str()
        .ok_or_else(|| invalid("Cargo metadata missing target directory"))?;
    let directory = PathBuf::from(target).join("verify");
    fs::create_dir_all(&directory)?;
    // An OS lock, not a stale pid/marker. Only our planner participates; we
    // neither kill nor wait on unrelated builds. Our Cargo children are serial.
    let lock = OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(directory.join("planner.lock"))?;
    lock.try_lock().map_err(|e| {
        invalid(format!(
            "another verification planner owns {}: {e}; retry after it finishes",
            directory.display()
        ))
    })?;
    let _lock = PlannerLock(lock);
    for check in plan {
        let start = Instant::now();
        let receipt = directory.join(format!("{}.json", check.id));
        let inputs = tracked_inputs(root)?;
        let configuration = config(root, check)?;
        let external_path_dependency = metadata["packages"].as_array().is_some_and(|packages| {
            packages.iter().any(|p| {
                p["dependencies"].as_array().is_some_and(|deps| {
                    deps.iter().any(|d| {
                        d["path"]
                            .as_str()
                            .is_some_and(|p| !Path::new(p).starts_with(root))
                    })
                })
            })
        });
        let before = match if external_path_dependency {
            Err(invalid(
                "external path dependency: local source fingerprint cannot certify its inputs",
            ))
        } else {
            digest(root, &inputs, &configuration)
        } {
            Ok(digest) => Some(digest),
            Err(error) => {
                println!(
                    "NO REUSE {}: {error}; execute fresh without a receipt",
                    check.id
                );
                None
            }
        };
        let prior = fs::read(&receipt)
            .ok()
            .and_then(|s| serde_json::from_slice::<Value>(&s).ok());
        if opts.reuse
            && !["full-tests", "dependency-floors"].contains(&check.id.as_str())
            && std::env::var_os("CI").is_none()
            && prior.as_ref().is_some_and(|p| {
                before
                    .as_ref()
                    .is_some_and(|hash| p["fingerprint"] == *hash)
                    && p["success"] == true
            })
        {
            println!(
                "REUSE {}: all current inputs, commands, environment and tool versions match ({:.3}s)",
                check.id,
                start.elapsed().as_secs_f64()
            );
            continue;
        }
        println!(
            "RUN {}: {}",
            check.id,
            if !opts.reuse {
                "fresh execution requested"
            } else {
                "no matching successful local result"
            }
        );
        // Invalidate before running, so a failed rerun cannot expose old success.
        if receipt.exists() {
            fs::remove_file(&receipt)?;
        }
        for step in &check.steps {
            if step.program.starts_with('@') {
                internal(root, &directory, step)?;
            } else {
                let status = command(root, step).status()?;
                if !status.success() {
                    return Err(invalid(format!(
                        "required check {} failed: {} {:?} ({status})",
                        check.id, step.program, step.args
                    )));
                }
            }
        }
        if before.is_some()
            && before != Some(digest(root, &tracked_inputs(root)?, &config(root, check)?)?)
        {
            return Err(invalid(format!(
                "inputs changed during {}; result not reusable; rerun",
                check.id
            )));
        }
        let elapsed = start.elapsed().as_secs_f64();
        if before.is_none() {
            println!("PASS {}: {elapsed:.3}s; no reusable result", check.id);
            continue;
        }
        let value =
            serde_json::json!({"schema":1,"success":true,"fingerprint":before,"seconds":elapsed});
        let tmp = receipt.with_extension("tmp");
        let mut f = File::create(&tmp)?;
        f.write_all(&serde_json::to_vec_pretty(&value)?)?;
        f.sync_all()?;
        fs::rename(tmp, receipt)?;
        println!("PASS {}: {elapsed:.3}s", check.id);
    }
    println!("All selected checks passed. This does not certify independent review or remote CI.");
    Ok(())
}
