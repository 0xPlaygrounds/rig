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

pub(super) fn tracked_inputs(root: &Path) -> Result<Vec<String>> {
    Ok(output(root, "git", &["ls-files", "-z"])?
        .split('\0')
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .collect())
}
pub(super) fn other_inputs(root: &Path, target: Option<&Path>) -> Result<Vec<String>> {
    let mut pending = Vec::new();
    for ignored in [false, true] {
        let mut args = vec![
            "ls-files",
            "--others",
            "--exclude-standard",
            "--directory",
            "--no-empty-directory",
            "-z",
        ];
        if ignored {
            args.push("--ignored");
        }
        pending.extend(
            output(root, "git", &args)?
                .split('\0')
                .filter(|s| !s.is_empty())
                .map(|s| root.join(s)),
        );
    }
    let mut files = Vec::new();
    while let Some(path) = pending.pop() {
        // Only disposable, untracked build outputs are pruned. A tracked file
        // within these directories still enters through tracked_inputs.
        if path.starts_with(root.join("target"))
            || target.is_some_and(|target| path.starts_with(target))
        {
            continue;
        }
        let metadata = fs::symlink_metadata(&path)?;
        if metadata.is_dir() && !metadata.file_type().is_symlink() {
            for entry in fs::read_dir(&path)? {
                pending.push(entry?.path());
            }
        } else {
            files.push(
                path.strip_prefix(root)
                    .map_err(|e| invalid(e.to_string()))?
                    .to_str()
                    .ok_or_else(|| invalid("non-UTF-8 ignored input"))?
                    .to_string(),
            );
        }
    }
    Ok(files)
}

fn manifest_paths(root: &Path, directory: &Path, value: &toml::Value) -> Result<()> {
    match value {
        toml::Value::Table(table) => {
            for (key, value) in table {
                if key == "path"
                    && let Some(path) = value.as_str()
                {
                    let resolved = fs::canonicalize(directory.join(path))?;
                    if !resolved.starts_with(root) {
                        return Err(invalid(format!(
                            "external manifest path {path}: cannot certify external source"
                        )));
                    }
                }
                manifest_paths(root, directory, value)?;
            }
        }
        toml::Value::Array(values) => {
            for value in values {
                manifest_paths(root, directory, value)?;
            }
        }
        _ => {}
    }
    Ok(())
}
fn guard_manifest_paths(root: &Path, inputs: &[String]) -> Result<()> {
    // Include nested/patch crate manifests too: --no-deps only reports workspace
    // members, and an internal path crate can itself depend on external source.
    let manifests = inputs
        .iter()
        .filter(|name| {
            Path::new(name)
                .file_name()
                .is_some_and(|name| name == "Cargo.toml")
        })
        .map(|name| root.join(name));
    for manifest in manifests {
        let text = match fs::read_to_string(&manifest) {
            Ok(text) => text,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
            Err(e) => return Err(e.into()),
        };
        let value: toml::Value = toml::from_str(&text).map_err(|e| invalid(e.to_string()))?;
        manifest_paths(root, manifest.parent().unwrap_or(root), &value)?;
    }
    Ok(())
}
fn guard_config_paths(bytes: &[u8]) -> Result<()> {
    let text = std::str::from_utf8(bytes).map_err(|e| invalid(e.to_string()))?;
    let value: toml::Value = toml::from_str(text).map_err(|e| invalid(e.to_string()))?;
    // Cargo configuration can override sources outside metadata --no-deps.
    // Conservatively execute fresh instead of inferring their dependency graph.
    if ["paths", "patch", "replace", "source"]
        .iter()
        .any(|key| value.get(key).is_some())
    {
        return Err(invalid(
            "Cargo source/path override: execute fresh without a reusable receipt",
        ));
    }
    Ok(())
}

fn input_hashes(root: &Path, files: &[String]) -> Result<BTreeMap<String, String>> {
    let mut values = BTreeMap::new();
    let mut existing = Vec::new();
    for name in files {
        // Explicit reporting-only precedent. Do not exempt other Markdown:
        // rustdoc, include_str!, fixtures and build scripts may consume it.
        if name == "DEVELOPING.md" {
            continue;
        }
        match fs::symlink_metadata(root.join(name)) {
            Ok(meta) => {
                if !meta.is_file() || meta.file_type().is_symlink() {
                    return Err(invalid(format!(
                        "unsupported input {name}; execute fresh without a reusable receipt"
                    )));
                }
                #[cfg(not(unix))]
                let mode = String::new();
                #[cfg(unix)]
                let mode = {
                    use std::os::unix::fs::PermissionsExt;
                    meta.permissions().mode().to_string()
                };
                values.insert(name.clone(), mode);
                existing.push(name.as_str());
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                values.insert(name.clone(), "<deleted>".into());
            }
            Err(e) => return Err(e.into()),
        }
    }
    // Git hashes actual working bytes, without filters or trusting index stat data.
    for paths in existing.chunks(256) {
        let mut args = vec!["hash-object", "--no-filters", "--"];
        args.extend_from_slice(paths);
        let hashes = output(root, "git", &args)?;
        if hashes.lines().count() != paths.len() {
            return Err(invalid("incomplete input hashes"));
        }
        for (name, hash) in paths.iter().zip(hashes.lines()) {
            if let Some(value) = values.get_mut(*name) {
                value.push(':');
                value.push_str(hash);
            }
        }
    }
    Ok(values)
}
fn fingerprint(files: &BTreeMap<String, String>, config: &[u8]) -> Result<String> {
    let mut hash = Sha256::new();
    hash.update(b"rig-verify-local-v3\0");
    hash.update(Sha256::digest(config));
    hash.update(serde_json::to_vec(files)?);
    Ok(format!("{:x}", hash.finalize()))
}
#[cfg(test)]
pub(super) fn digest(root: &Path, files: &[String], config: &[u8]) -> Result<String> {
    fingerprint(&input_hashes(root, files)?, config)
}
fn identity(root: &Path, metadata: &Value, check: &Check) -> Result<Value> {
    if metadata["packages"].as_array().is_some_and(|packages| {
        packages.iter().any(|p| {
            p["dependencies"].as_array().is_some_and(|deps| {
                deps.iter().any(|d| {
                    d["path"]
                        .as_str()
                        .is_some_and(|p| !Path::new(p).starts_with(root))
                })
            })
        })
    }) {
        return Err(invalid(
            "external path dependency: local inputs cannot certify its source",
        ));
    }
    if std::env::vars_os().any(|(key, _)| {
        key.to_string_lossy().starts_with("CARGO_SOURCE_")
            || key.to_string_lossy().starts_with("CARGO_PATCH_")
    }) {
        return Err(invalid(
            "Cargo source/patch environment override: cannot certify external source",
        ));
    }
    let mut inputs = tracked_inputs(root)?;
    inputs.extend(other_inputs(
        root,
        metadata["target_directory"].as_str().map(Path::new),
    )?);
    inputs.sort();
    inputs.dedup();
    guard_manifest_paths(root, &inputs)?;
    let files = input_hashes(root, &inputs)?;
    let config = config(root, check)?;
    Ok(
        serde_json::json!({"fingerprint": fingerprint(&files, &config)?,
        "inputs": files, "configuration": format!("{:x}", Sha256::digest(config))}),
    )
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
                    guard_config_paths(&bytes)?;
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
        let value = process::capture(root, program, &args);
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
fn internal(root: &Path, directory: &Path, log: &Path, step: &Step) -> Result<()> {
    match step.program.as_str() {
        "@layout" => crate::test_layout::check(root).map_err(invalid),
        "@scenarios" => crate::scenarios::run(root, Vec::new()).map_err(|e| invalid(e.to_string())),
        "@registrations" => {
            let cargo = Step {
                program: "cargo".into(),
                args: step.args.clone(),
                env: step.env.clone(),
            };
            let result = process::run(root, &cargo, log)?;
            if !result.status.success() {
                return Err(invalid("registration discovery failed"));
            }
            let json = result.stdout;
            let file = directory.join("registrations.json");
            fs::write(&file, json)?;
            crate::scenarios::run(root, vec![file.to_string_lossy().into_owned()])
                .map_err(|e| invalid(e.to_string()))
        }
        "@fixture-paths" => {
            for path in tracked_inputs(root)?
                .into_iter()
                .chain(other_inputs(root, directory.parent())?)
                .filter(|p| {
                    p.starts_with("crates/")
                        && p.ends_with(".rs")
                        && (p.contains("/src/") || p.contains("/tests/"))
                })
            {
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
            let cargo = Step::new(
                "cargo",
                &[
                    "check",
                    "--locked",
                    "--package",
                    package,
                    "--target",
                    "wasm32-unknown-unknown",
                ],
            )
            .env("CARGO_TERM_COLOR", "never");
            let result = process::run(root, &cargo, log)?;
            let stderr = String::from_utf8_lossy(&result.stderr);

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
pub(super) fn policy(opts: &Options, check: &Check) -> &'static str {
    if preflight::uses_runtime_model(check) {
        "mandatory fresh: external runtime models are non-reusable; no success receipt"
    } else if ["full-tests", "dependency-floors"].contains(&check.id.as_str()) {
        "mandatory fresh: services/dependency resolution are non-reusable"
    } else if opts.mode == Mode::Full || std::env::var_os("CI").is_some() {
        "mandatory fresh: full mode or CI"
    } else if !opts.reuse {
        "mandatory execution: reuse not requested (PR reuse requires --reuse)"
    } else {
        "reuse-eligible: requires matching successful inputs and configuration"
    }
}
fn reusable(opts: &Options, check: &Check) -> bool {
    policy(opts, check).starts_with("reuse-eligible")
}
fn prior(directory: &Path, check: &Check) -> Option<Value> {
    fs::read(directory.join(format!("{}.json", check.id)))
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
}
fn matches(prior: &Value, now: &Value) -> bool {
    prior["success"] == true && prior["schema"] == 2 && prior["fingerprint"] == now["fingerprint"]
}
fn difference(prior: &Value, now: &Value) -> String {
    if !prior["inputs"].is_object() {
        return "old receipt format; input details unavailable".into();
    }
    let mut changes = std::collections::BTreeSet::new();
    for state in [prior, now] {
        if let Some(inputs) = state["inputs"].as_object() {
            for name in inputs.keys() {
                if prior["inputs"].get(name) != now["inputs"].get(name) {
                    changes.insert(name);
                }
            }
        }
    }
    format!(
        "changed inputs: {changes:?}; command/toolchain/target/environment/Cargo or nextest configuration changed: {}",
        prior["configuration"] != now["configuration"]
    )
}
fn explain(opts: &Options, check: &Check, old: Option<&Value>, now: Option<&Value>) -> bool {
    let matching = old.zip(now).is_some_and(|(old, now)| matches(old, now));
    if let Some(old) = old {
        if let Some(seconds) = old["seconds"].as_f64() {
            println!(
                "HISTORY {}: {seconds:.3}s measured on prior execution; not an estimate",
                check.id
            );
        }
        if !matching {
            println!(
                "NO REUSE {}: {}",
                check.id,
                now.map_or_else(
                    || "inputs cannot be certified".into(),
                    |now| difference(old, now)
                )
            );
        } else if !reusable(opts, check) {
            println!(
                "NO REUSE {}: matching prior success, but {}",
                check.id,
                policy(opts, check)
            );
        }
    } else {
        println!("NO REUSE {}: no readable successful receipt", check.id);
    }
    matching && reusable(opts, check)
}
fn directory(metadata: &Value) -> Result<PathBuf> {
    Ok(PathBuf::from(
        metadata["target_directory"]
            .as_str()
            .ok_or_else(|| invalid("Cargo metadata missing target directory"))?,
    )
    .join("verify"))
}
pub(super) fn preview(root: &Path, metadata: &Value, opts: &Options, plan: &[Check]) -> Result<()> {
    let directory = directory(metadata)?;
    for check in plan {
        let now = identity(root, metadata, check);
        if let Err(error) = &now {
            println!("NO REUSE {}: {error}", check.id);
        }
        if explain(
            opts,
            check,
            prior(&directory, check).as_ref(),
            now.as_ref().ok(),
        ) {
            println!("WOULD REUSE {}: current inputs match", check.id);
        }
    }
    println!("Dry run only; no checks or prerequisites certified.");
    Ok(())
}
fn quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', "'\\''"))
}
fn continuation(root: &Path, opts: &Options) -> String {
    let mut args = match opts.mode {
        Mode::Pr => "--pr".to_string(),
        Mode::Changed => "--changed".to_string(),
        Mode::Full => "--full".to_string(),
        Mode::Check => format!("--check {}", quote(opts.check.as_deref().unwrap_or(""))),
    };
    if let Some(base) = &opts.base {
        args.push_str(&format!(" --base {}", quote(base)));
    }
    if opts.mode != Mode::Full {
        args.push_str(" --reuse");
    }
    format!(
        "cd {} && cargo xtask verify {args}",
        quote(&root.to_string_lossy())
    )
}
fn summary(
    root: &Path,
    metadata: &Value,
    opts: &Options,
    plan: &[Check],
    completed: &BTreeMap<String, Option<Value>>,
    failed: Option<&str>,
    elapsed: f64,
) -> Result<bool> {
    let directory = directory(metadata)?;
    println!(
        "SUMMARY: {}/{} completed; {elapsed:.3}s elapsed; logs: {}",
        completed.len(),
        plan.len(),
        directory.display()
    );
    let mut current = true;
    for check in plan {
        let now = if process::interrupted() {
            None
        } else {
            identity(root, metadata, check).ok()
        };
        let mut check_current = true;
        if let Some(before) = completed.get(&check.id) {
            match (before, &now) {
                (Some(before), Some(now)) if before["fingerprint"] == now["fingerprint"] => {
                    if preflight::uses_runtime_model(check) {
                        println!(
                            "SUCCESS {}: current source inputs; runtime models executed fresh, no reusable certification",
                            check.id
                        );
                    } else {
                        println!("SUCCESS {}: current inputs", check.id);
                    }
                }
                (None, _) => println!(
                    "SUCCESS {}: executed fresh with unsupported inputs; no reusable certification",
                    check.id
                ),
                (Some(_), None) => {
                    println!(
                        "UNVALIDATED SUCCESS {}: current inputs could not be checked; continuation must revalidate",
                        check.id
                    );
                    current = false;
                    check_current = false;
                }
                _ => {
                    println!(
                        "OLD-INPUT SUCCESS {}: does not verify the current tree",
                        check.id
                    );
                    current = false;
                    check_current = false;
                }
            }
        } else if failed == Some(check.id.as_str()) {
            println!("FAILED/INTERRUPTED {}: no success recorded", check.id);
        } else {
            println!("NOT RUN {}", check.id);
        }
        let valid = prior(&directory, check)
            .zip(now)
            .is_some_and(|(old, now)| matches(&old, &now));
        if valid
            && !preflight::uses_runtime_model(check)
            && !["full-tests", "dependency-floors"].contains(&check.id.as_str())
            && opts.mode != Mode::Full
            && std::env::var_os("CI").is_none()
        {
            println!(
                "STILL-VALID REUSABLE {}: continuation with --reuse may reuse",
                check.id
            );
        } else if !completed.contains_key(&check.id) || !check_current {
            println!(
                "REMAINING {}: requires execution; {}",
                check.id,
                policy(opts, check)
            );
        }
    }
    if failed.is_some() || completed.len() != plan.len() || !current {
        println!(
            "Freeze inputs after fixes, then continue (same environment/target):\n{}",
            continuation(root, opts)
        );
        println!(
            "Continuation always reexecutes selected full-tests, dependency-floors, and model-loading checks; no automatic rerun."
        );
    }
    std::io::stdout().flush()?;
    Ok(current)
}
pub(super) fn run(root: &Path, metadata: &Value, opts: &Options, plan: &[Check]) -> Result<()> {
    if plan.is_empty() {
        println!("No executable changes; no verification result claimed.");
        return Ok(());
    }
    let directory = directory(metadata)?;
    fs::create_dir_all(&directory)?;
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
    let start = Instant::now();
    let mut completed = BTreeMap::new();
    if let Err(error) = preflight::run(root, plan)
        .and_then(|()| preflight::prepare_model_cache(root, plan))
        .and_then(|()| preflight::prepare_fixtures(root, plan))
    {
        summary(
            root,
            metadata,
            opts,
            plan,
            &completed,
            None,
            start.elapsed().as_secs_f64(),
        )?;
        return Err(error);
    }
    for check in plan {
        let check_start = Instant::now();
        let receipt = directory.join(format!("{}.json", check.id));
        let log = directory.join(format!("{}.log", check.id));
        println!(
            "PROGRESS {}/{} completed; ACTIVE {}; {:.1}s elapsed; log: {}",
            completed.len(),
            plan.len(),
            check.id,
            start.elapsed().as_secs_f64(),
            log.display()
        );
        std::io::stdout().flush()?;
        let result = (|| -> Result<Option<Value>> {
            if process::interrupted() {
                return Err(invalid("verification interrupted"));
            }
            let before = match identity(root, metadata, check) {
                Ok(value) => Some(value),
                Err(error) => {
                    println!("NO REUSE {}: {error}", check.id);
                    None
                }
            };
            if explain(
                opts,
                check,
                prior(&directory, check).as_ref(),
                before.as_ref(),
            ) {
                println!("REUSE {}: matching current inputs", check.id);
                return Ok(before);
            }
            if receipt.exists() {
                fs::remove_file(&receipt)?;
            }
            File::create(&log)?;
            println!("RUN {}: {}", check.id, policy(opts, check));
            for step in &check.steps {
                if process::interrupted() {
                    return Err(invalid("verification interrupted"));
                }
                if step.program.starts_with('@') {
                    println!(
                        "PHASE {} {:?}; log: {}",
                        step.program,
                        step.args,
                        log.display()
                    );
                    std::io::stdout().flush()?;
                    let mut internal_log = OpenOptions::new().append(true).open(&log)?;
                    writeln!(
                        internal_log,
                        "INTERNAL {} {:?}: diagnostics on console",
                        step.program, step.args
                    )?;
                    internal(root, &directory, &log, step)?;
                    writeln!(internal_log, "PASS {}", step.program)?;
                } else {
                    let result = process::run(root, step, &log)?;
                    if !result.status.success() {
                        return Err(invalid(format!(
                            "required check {} failed: {} {:?} ({})",
                            check.id, step.program, step.args, result.status
                        )));
                    }
                }
            }
            if process::interrupted() {
                return Err(invalid("verification interrupted"));
            }
            if let Some(before) = &before {
                let after = identity(root, metadata, check)?;
                if before.get("fingerprint") != after.get("fingerprint") {
                    return Err(invalid(format!(
                        "inputs changed during {}; result not reusable; {}",
                        check.id,
                        difference(before, &after)
                    )));
                }
                if !preflight::uses_runtime_model(check) {
                    let mut value = before
                        .as_object()
                        .ok_or_else(|| invalid("invalid input identity"))?
                        .clone();
                    value.insert("schema".into(), 2.into());
                    value.insert("success".into(), true.into());
                    value.insert("seconds".into(), check_start.elapsed().as_secs_f64().into());
                    let tmp = receipt.with_extension("tmp");
                    let mut f = File::create(&tmp)?;
                    f.write_all(&serde_json::to_vec_pretty(&value)?)?;
                    f.sync_all()?;
                    fs::rename(tmp, &receipt)?;
                }
            }
            println!(
                "PASS {}: {:.3}s measured",
                check.id,
                check_start.elapsed().as_secs_f64()
            );
            Ok(before)
        })();
        match result {
            Ok(before) => {
                completed.insert(check.id.clone(), before);
            }
            Err(error) => {
                // Even an error after receipt publication must not expose success.
                if receipt.exists() {
                    fs::remove_file(&receipt)?;
                }
                summary(
                    root,
                    metadata,
                    opts,
                    plan,
                    &completed,
                    Some(&check.id),
                    start.elapsed().as_secs_f64(),
                )?;
                return Err(error);
            }
        }
    }
    if !summary(
        root,
        metadata,
        opts,
        plan,
        &completed,
        None,
        start.elapsed().as_secs_f64(),
    )? {
        return Err(invalid("earlier successes no longer verify current inputs"));
    }
    println!("All selected checks passed. This does not certify independent review or remote CI.");
    Ok(())
}
