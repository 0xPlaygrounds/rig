//! `cargo xtask cassette record`: re-record fixtures by owning test.
//!
//! Each fixture resolves to the test that records it; each test runs once in
//! record mode, however many of the given fixtures it owns. Every run is
//! logged in the attempt ledger (`recordings.tsv` under the attempt root)
//! before it starts and again when it ends, and a test at the attempt cap
//! (at most six) is not run again. The test binaries are built first, so a
//! compile error spends no attempt. The owner's fixture directory is
//! snapshotted before each run; after a failed run every fixture it changed
//! is copied to the attempt root and the snapshot restored, so a failure
//! never leaves a half-written fixture behind. The created-resource cleanup
//! pass runs at the end whatever happened, and the command fails when any
//! run did.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use super::owner::{Owner, owners};

/// The attempt ledger's column header.
pub(crate) const LEDGER_HEADER: &str = "provider\tfixtures\ttest\tattempt\texit\tnote";

/// The most attempts any test may spend.
pub(crate) const HARD_CAP: usize = 6;

/// The ledger's `exit` value for a run that has started and not yet ended.
pub(crate) const STARTED: &str = "started";

/// How many attempts `test` already spent, per the ledger text: the highest
/// attempt number any of its rows names, so a run interrupted after its
/// `started` row still counts.
pub(crate) fn prior_attempts(ledger: &str, test: &str) -> usize {
    ledger
        .lines()
        .skip(1)
        .filter_map(|line| {
            let columns: Vec<&str> = line.split('\t').collect();
            (columns.get(2) == Some(&test) && columns.get(4) != Some(&"skipped"))
                .then(|| columns.get(3).and_then(|attempt| attempt.parse().ok()))
                .flatten()
        })
        .max()
        .unwrap_or(0)
}

/// The attempt number `test`'s next run gets, or `None` when its attempts
/// already reach `cap`. Failed, passed and interrupted runs all count.
pub(crate) fn next_attempt(ledger: &str, test: &str, cap: usize) -> Option<usize> {
    let prior = prior_attempts(ledger, test);
    (prior < cap).then_some(prior + 1)
}

/// A fixture argument: `provider/dir/name.yaml`, optionally under the
/// fixture root. Returns the provider and the scenario.
pub(crate) fn parse_fixture(argument: &str) -> Option<(String, String)> {
    let relative = argument
        .split_once("fixtures/cassettes/")
        .map_or(argument, |(_, rest)| rest);
    let relative = relative.strip_suffix(".yaml")?;
    let (provider, scenario) = relative.split_once('/')?;
    (!provider.is_empty() && !scenario.is_empty())
        .then(|| (provider.to_owned(), scenario.to_owned()))
}

/// Every file under `dir`, relative to it, with its bytes.
pub(crate) type Snapshot = BTreeMap<PathBuf, Vec<u8>>;

pub(crate) fn snapshot(dir: &Path) -> Result<Snapshot, String> {
    let mut files = Snapshot::new();
    let mut pending = vec![dir.to_path_buf()];
    while let Some(current) = pending.pop() {
        let Ok(entries) = std::fs::read_dir(&current) else {
            continue;
        };
        for entry in entries {
            let path = entry.map_err(|error| error.to_string())?.path();
            if path.is_dir() {
                pending.push(path);
            } else {
                let bytes = std::fs::read(&path).map_err(|error| error.to_string())?;
                let relative = path.strip_prefix(dir).unwrap_or(&path).to_path_buf();
                files.insert(relative, bytes);
            }
        }
    }
    Ok(files)
}

/// The files under `dir` that differ from `before`: changed, added or
/// removed, relative to `dir`.
pub(crate) fn changed_since(dir: &Path, before: &Snapshot) -> Result<Vec<PathBuf>, String> {
    let after = snapshot(dir)?;
    let mut changed: Vec<PathBuf> = after
        .iter()
        .filter(|(path, bytes)| before.get(*path) != Some(*bytes))
        .map(|(path, _)| path.clone())
        .collect();
    changed.extend(
        before
            .keys()
            .filter(|path| !after.contains_key(*path))
            .cloned(),
    );
    changed.sort();
    Ok(changed)
}

/// After a failed run: copy each file under `dir` that differs from
/// `before` to `kept` (keeping its relative path), then put `before`'s bytes
/// back and remove files the run created. Every file is attempted even when
/// one fails, and the error names each file that could not be restored.
/// Returns what was restored.
pub(crate) fn restore_snapshot(
    dir: &Path,
    before: &Snapshot,
    kept: &Path,
) -> Result<Vec<PathBuf>, String> {
    let changed = changed_since(dir, before)?;
    let write = |path: &Path, bytes: &[u8]| -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, bytes)
    };
    let mut errors = Vec::new();
    for relative in &changed {
        let path = dir.join(relative);
        // Keeping a copy is best effort; putting the original back is not
        // skipped when keeping fails.
        if let Ok(current) = std::fs::read(&path)
            && let Err(error) = write(&kept.join(relative), &current)
        {
            errors.push(format!("keeping {}: {error}", path.display()));
        }
        let restored = match before.get(relative) {
            Some(bytes) => write(&path, bytes),
            None => std::fs::remove_file(&path),
        };
        if let Err(error) = restored {
            errors.push(format!("restoring {}: {error}", path.display()));
        }
    }
    if errors.is_empty() {
        Ok(changed)
    } else {
        Err(format!("could not restore: {}", errors.join("; ")))
    }
}

/// Whether a passing nextest log ran no test at all.
pub(crate) fn ran_nothing(log: &str) -> bool {
    log.contains("Starting 0 tests") || log.contains(" 0 tests run")
}

pub(crate) struct Options {
    pub(crate) cap: usize,
    pub(crate) pause_seconds: u64,
    pub(crate) dry_run: bool,
    pub(crate) cleanup: bool,
    pub(crate) fixtures: Vec<String>,
}

pub(crate) fn parse_options(args: &[String]) -> Result<Options, String> {
    let mut options = Options {
        cap: HARD_CAP,
        pause_seconds: 0,
        dry_run: false,
        cleanup: true,
        fixtures: Vec::new(),
    };
    let mut args = args.iter();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--cap" => {
                options.cap = args
                    .next()
                    .and_then(|value| value.parse().ok())
                    .ok_or("--cap needs a number")?;
                if options.cap > HARD_CAP {
                    return Err(format!("--cap may not exceed {HARD_CAP}"));
                }
            }
            "--pause" => {
                options.pause_seconds = args
                    .next()
                    .and_then(|value| value.parse().ok())
                    .ok_or("--pause needs seconds")?;
            }
            "--dry-run" => options.dry_run = true,
            "--no-cleanup" => options.cleanup = false,
            fixture => options.fixtures.push(fixture.to_owned()),
        }
    }
    if options.fixtures.is_empty() {
        return Err("give at least one fixture (provider/scenario.yaml)".into());
    }
    Ok(options)
}

pub(crate) fn run(root: &Path, args: &[String]) -> Result<(), String> {
    let options = parse_options(args)?;
    let attempts = super::attempt_root(root);
    let ledger_path = attempts.join("recordings.tsv");

    // One run per owning test, however many of its fixtures were named.
    let mut by_test: BTreeMap<(String, String), Vec<String>> = BTreeMap::new();
    let mut unowned = Vec::new();
    for argument in &options.fixtures {
        let (provider, scenario) =
            parse_fixture(argument).ok_or_else(|| format!("not a fixture path: {argument}"))?;
        let fixture = format!("{provider}/{scenario}.yaml");
        let found = owners(root, &provider, &scenario)?;
        match found.first() {
            Some(Owner::Test(test)) => {
                if found.len() > 1 {
                    println!(
                        "{fixture}: {} producers, recording with {test}",
                        found.len()
                    );
                }
                by_test
                    .entry((provider.clone(), test.clone()))
                    .or_default()
                    .push(fixture);
            }
            other => unowned.push((provider, fixture, other.cloned())),
        }
    }
    for (provider, fixture, reason) in &unowned {
        println!("skip {fixture}: {reason:?}");
        if !options.dry_run {
            append_row(
                &ledger_path,
                &[
                    provider,
                    fixture,
                    "-",
                    "0",
                    "skipped",
                    &format!("{reason:?}"),
                ],
            )?;
        }
    }

    if !options.dry_run {
        let providers: std::collections::BTreeSet<&str> = by_test
            .keys()
            .map(|(provider, _)| provider.as_str())
            .collect();
        for provider in providers {
            build_tests(root, provider)?;
        }
    }
    let session = Session {
        root,
        options: &options,
        attempts: &attempts,
        ledger_path: &ledger_path,
        fixture_root: &root.join("crates/rig-cassette/fixtures/cassettes"),
    };
    let mut failures = Vec::new();
    for ((provider, test), fixtures) in &by_test {
        if let Err(error) = session.record_one(provider, test, fixtures) {
            println!("  {test}: {error}");
            failures.push(format!("{test}: {error}"));
        }
    }

    // Clean up whatever happened above, then report.
    let cleaned = if options.cleanup && !options.dry_run {
        super::cleanup(root, &[])
    } else {
        Ok(())
    };
    if !failures.is_empty() {
        return Err(format!(
            "{} test(s) did not record:\n{}",
            failures.len(),
            failures.join("\n")
        ));
    }
    cleaned
}

/// What every attempt of one `record` invocation shares.
struct Session<'a> {
    root: &'a Path,
    options: &'a Options,
    attempts: &'a Path,
    ledger_path: &'a Path,
    fixture_root: &'a Path,
}

impl Session<'_> {
    /// One attempt of one owning test; `Err` when it did not record.
    fn record_one(&self, provider: &str, test: &str, fixtures: &[String]) -> Result<(), String> {
        let ledger = std::fs::read_to_string(self.ledger_path).unwrap_or_default();
        let Some(attempt) = next_attempt(&ledger, test, self.options.cap) else {
            return Err(format!(
                "its attempts reach the cap of {}",
                self.options.cap
            ));
        };
        println!("record {test} (attempt {attempt}): {}", fixtures.join(", "));
        if self.options.dry_run {
            return Ok(());
        }
        let provider_dir = self.fixture_root.join(provider);
        let before = snapshot(&provider_dir)?;
        let attempt_text = attempt.to_string();
        append_row(
            self.ledger_path,
            &[
                provider,
                &fixtures.join(";"),
                test,
                &attempt_text,
                STARTED,
                "-",
            ],
        )?;

        let log = self
            .attempts
            .join(provider)
            .join(format!("{}.attempt{attempt}.log", test.replace("::", "__")));
        // A run that could not even start is a failed run: it still restores
        // and writes its result row, with the reason.
        let (mut passed, spawn_error) =
            match run_test(self.root, self.attempts, provider, test, &log) {
                Ok(passed) => (passed, None),
                Err(error) => (false, Some(error)),
            };
        let mut note = format!("log {}", log.display());
        if let Some(error) = &spawn_error {
            note.push_str(&format!("; could not run: {error}"));
        }
        if passed && ran_nothing(&std::fs::read_to_string(&log).unwrap_or_default()) {
            passed = false;
            note.push_str("; no test ran");
        }
        // Price and report every fixture the run touched, named or not.
        let changed = changed_since(&provider_dir, &before).unwrap_or_default();
        let mut touched: Vec<String> = changed
            .iter()
            .map(|relative| format!("{provider}/{}", relative.display()))
            .collect();
        for fixture in fixtures {
            if !touched.contains(fixture) {
                touched.push(fixture.clone());
            }
        }
        let mut restore_error = None;
        if passed {
            if changed.is_empty() {
                note.push_str("; no fixture changed");
            }
        } else {
            let kept = self
                .attempts
                .join(provider)
                .join(format!("{}.attempt{attempt}", test.replace("::", "__")));
            match restore_snapshot(&provider_dir, &before, &kept) {
                Ok(restored) if !restored.is_empty() => note.push_str(&format!(
                    "; restored {} fixture(s), kept {}",
                    restored.len(),
                    kept.display()
                )),
                Ok(_) => {}
                Err(error) => {
                    note.push_str(&format!("; {error}"));
                    restore_error = Some(error);
                }
            }
        }
        append_row(
            self.ledger_path,
            &[
                provider,
                &touched.join(";"),
                test,
                &attempt_text,
                if passed { "0" } else { "1" },
                &note,
            ],
        )?;
        println!("  {}: {note}", if passed { "recorded" } else { "failed" });
        if self.options.pause_seconds > 0 {
            std::thread::sleep(std::time::Duration::from_secs(self.options.pause_seconds));
        }
        match (passed, restore_error) {
            (_, Some(error)) => Err(error),
            (true, None) => Ok(()),
            (false, None) => {
                Err(spawn_error.unwrap_or_else(|| format!("failed; see {}", log.display())))
            }
        }
    }
}

/// Build `provider`'s test binary without running it, so a compile error
/// spends no attempt.
fn build_tests(root: &Path, provider: &str) -> Result<(), String> {
    let status = Command::new("cargo")
        .args([
            "nextest",
            "run",
            "--no-run",
            "--locked",
            "-p",
            "rig-cassette",
            "--features",
            "http,agent,ecs,bedrock",
            "--test",
            provider,
        ])
        .current_dir(root)
        .status()
        .map_err(|error| format!("cargo nextest: {error}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!(
            "the {provider} test binary does not build; no attempt was spent"
        ))
    }
}

fn run_test(
    root: &Path,
    attempts: &Path,
    provider: &str,
    test: &str,
    log: &Path,
) -> Result<bool, String> {
    if let Some(parent) = log.parent() {
        std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let output = std::fs::File::create(log).map_err(|error| error.to_string())?;
    let errors = output.try_clone().map_err(|error| error.to_string())?;
    let status = Command::new("cargo")
        .args([
            "nextest",
            "run",
            "--locked",
            "-p",
            "rig-cassette",
            "--features",
            "http,agent,ecs,bedrock",
            "--test",
            provider,
            "--run-ignored",
            "all",
            "--retries",
            "0",
            "-E",
            &format!("test(={test})"),
        ])
        .current_dir(root)
        .env("RIG_PROVIDER_TEST_MODE", "record")
        .env("RIG_CASSETTE_ATTEMPT_DIR", attempts)
        .env_remove("RIG_REGENERATE_GOLDEN")
        .stdout(output)
        .stderr(errors)
        .status()
        .map_err(|error| format!("cargo nextest: {error}"))?;
    Ok(status.success())
}

fn append_row(path: &Path, columns: &[&str]) -> Result<(), String> {
    use std::io::Write as _;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let fresh = !path.exists();
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|error| error.to_string())?;
    if fresh {
        writeln!(file, "{LEDGER_HEADER}").map_err(|error| error.to_string())?;
    }
    let line: Vec<String> = columns
        .iter()
        .map(|column| column.replace(['\t', '\n'], " "))
        .collect();
    writeln!(file, "{}", line.join("\t")).map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests;
