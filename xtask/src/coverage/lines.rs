//! Line and branch coverage of production code, from `cargo llvm-cov
//! nextest` over the workspace with all features under the `local` nextest
//! profile (no Docker suites, no nested Cargo builds).
//!
//! Branch coverage is unstable in rustc, so the instrumented crates, and
//! only those, are built with `RUSTC_BOOTSTRAP` naming them. Production code
//! is every `src/` file of the facade and `crates/*`, except test modules,
//! test helpers and proc-macro crates, whose code runs in the compiler and
//! not in any test. The baseline keeps, per file, a hash of its source, its
//! covered and instrumented line and branch counts, and its covered lines
//! and branches. A baseline written from several runs keeps only
//! what every run covered, so a branch that only some runs reach is not
//! held against a later one. A file whose source is unchanged fails the
//! check when a line or branch it covered is no longer covered; a changed
//! file fails when its line or branch ratio falls. The regions in
//! [`unstable`](super::unstable) are left out of every measurement.
//!
//! `--per-test` reruns the suites with a nextest wrapper script that gives
//! every test process its own profile, converts it to covered lines and
//! branches as soon as the test exits, and deletes the raw profile.

#[cfg(test)]
mod tests;

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;

use super::shapes::hash;
use super::{Result, invalid, target_dir};
use crate::support::{files_under, output};

/// The baseline file's column header.
pub(crate) const HEADER: &str = "file\tsource\tlines\tbranches\tcovered lines\tcovered branches";

/// Packages left out of the measurement: the nested minimal runner repeats
/// sources `rig-cassette` already runs, and xtask is not production code.
const EXCLUDED_PACKAGES: &[&str] = &["rig-cassette-minimal", "xtask"];

/// Environment the per-test wrapper reads.
const WRAP_OUT: &str = "RIG_COVERAGE_OUT";
const WRAP_LLVM: &str = "RIG_COVERAGE_LLVM_BIN";

/// A branch outcome: its line, block and branch numbers in the LCOV report.
pub(crate) type Branch = (u32, u32, u32);

/// One file's coverage: every instrumented line and branch, and whether it
/// was covered.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct FileCoverage {
    pub(crate) lines: BTreeMap<u32, bool>,
    pub(crate) branches: BTreeMap<Branch, bool>,
}

impl FileCoverage {
    fn covered_lines(&self) -> impl Iterator<Item = u32> + '_ {
        self.lines
            .iter()
            .filter(|(_, covered)| **covered)
            .map(|(line, _)| *line)
    }

    fn counts(&self) -> Counts {
        Counts {
            lines: (self.covered_lines().count(), self.lines.len()),
            branches: (
                self.branches.values().filter(|covered| **covered).count(),
                self.branches.len(),
            ),
        }
    }

    /// Count as covered what either covered.
    pub(crate) fn unite(&mut self, other: &Self) {
        for (line, covered) in &other.lines {
            *self.lines.entry(*line).or_insert(false) |= *covered;
        }
        for (branch, covered) in &other.branches {
            *self.branches.entry(*branch).or_insert(false) |= *covered;
        }
    }

    /// Keep as covered only what `other` covered too.
    pub(crate) fn intersect(&mut self, other: &Self) {
        for (line, covered) in &mut self.lines {
            *covered &= other.lines.get(line).copied().unwrap_or(false);
        }
        for (branch, covered) in &mut self.branches {
            *covered &= other.branches.get(branch).copied().unwrap_or(false);
        }
    }
}

/// Covered and instrumented counts of one file.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Counts {
    pub(crate) lines: (usize, usize),
    pub(crate) branches: (usize, usize),
}

/// A baseline row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Row {
    pub(crate) source: String,
    pub(crate) counts: Counts,
    pub(crate) covered_lines: BTreeSet<u32>,
    pub(crate) covered_branches: BTreeSet<Branch>,
}

/// Whether `path` (relative to the workspace root) is production source.
pub(crate) fn is_production(path: &str) -> bool {
    let in_src = path.starts_with("src/")
        || path
            .strip_prefix("crates/")
            .and_then(|rest| rest.split_once('/'))
            .is_some_and(|(_, rest)| rest.starts_with("src/"));
    in_src
        && path.ends_with(".rs")
        && !path.split('/').any(|component| {
            let stem = component.strip_suffix(".rs").unwrap_or(component);
            matches!(stem, "tests" | "test_utils" | "test_fixtures") || stem.ends_with("_tests")
        })
}

/// Parse an LCOV report, keeping production files under `root`.
pub(crate) fn parse_lcov(text: &str, root: &Path) -> BTreeMap<String, FileCoverage> {
    let mut files: BTreeMap<String, FileCoverage> = BTreeMap::new();
    let mut current: Option<String> = None;
    for line in text.lines() {
        if let Some(path) = line.strip_prefix("SF:") {
            current = Path::new(path)
                .strip_prefix(root)
                .ok()
                .map(|relative| relative.to_string_lossy().replace('\\', "/"))
                .filter(|relative| is_production(relative));
        } else if line == "end_of_record" {
            current = None;
        } else if let Some(file) = &current {
            if let Some(rest) = line.strip_prefix("DA:") {
                let mut fields = rest.split(',');
                if let (Some(Ok(number)), Some(count)) =
                    (fields.next().map(str::parse), fields.next())
                {
                    let entry = files.entry(file.clone()).or_default();
                    let covered = entry.lines.entry(number).or_insert(false);
                    *covered |= executed(count);
                }
            } else if let Some(rest) = line.strip_prefix("BRDA:") {
                let fields: Vec<&str> = rest.split(',').collect();
                if let [number, block, branch, taken] = fields.as_slice()
                    && let (Ok(number), Ok(block), Ok(branch)) =
                        (number.parse(), block.parse(), branch.parse())
                {
                    let entry = files.entry(file.clone()).or_default();
                    let covered = entry
                        .branches
                        .entry((number, block, branch))
                        .or_insert(false);
                    *covered |= *taken != "-" && executed(taken);
                }
            }
        }
    }
    files
}

/// Whether an LCOV execution count shows the code ran. llvm-cov derives
/// some counts by subtracting counters, so a panic that unwinds out of a
/// function between two increments can drive one below zero. It prints
/// such a count wrapped, as `u64::MAX` for a line or `u32::MAX` for a
/// branch, and a wrapped count is no evidence of execution.
pub(crate) fn executed(count: &str) -> bool {
    count
        .parse::<u64>()
        .is_ok_and(|count| count != 0 && count != u64::from(u32::MAX) && count < 1 << 63)
}

/// `1-3,7,9-10` for the sorted `lines`.
pub(crate) fn ranges(lines: impl IntoIterator<Item = u32>) -> String {
    let mut out = String::new();
    let mut run: Option<(u32, u32)> = None;
    let flush = |out: &mut String, (start, end): (u32, u32)| {
        if !out.is_empty() {
            out.push(',');
        }
        if start == end {
            let _ = write!(out, "{start}");
        } else {
            let _ = write!(out, "{start}-{end}");
        }
    };
    for line in lines {
        run = match run {
            Some((start, end)) if line == end + 1 => Some((start, line)),
            Some(previous) => {
                flush(&mut out, previous);
                Some((line, line))
            }
            None => Some((line, line)),
        };
    }
    if let Some(last) = run {
        flush(&mut out, last);
    }
    out
}

/// The lines in a [`ranges`] string.
pub(crate) fn parse_ranges(text: &str) -> Option<BTreeSet<u32>> {
    let mut lines = BTreeSet::new();
    for part in text.split(',').filter(|part| !part.is_empty()) {
        let (start, end): (u32, u32) = match part.split_once('-') {
            Some((start, end)) => (start.parse().ok()?, end.parse().ok()?),
            None => {
                let line = part.parse().ok()?;
                (line, line)
            }
        };
        lines.extend(start..=end);
    }
    Some(lines)
}

fn format_branch((line, block, branch): Branch) -> String {
    format!("{line}.{block}.{branch}")
}

pub(crate) fn parse_branch(text: &str) -> Option<Branch> {
    let mut parts = text.split('.').map(str::parse);
    match (parts.next(), parts.next(), parts.next(), parts.next()) {
        (Some(Ok(line)), Some(Ok(block)), Some(Ok(branch)), None) => Some((line, block, branch)),
        _ => None,
    }
}

/// The baseline text for `files`, with each file's source hash from `source`.
pub(crate) fn render(
    files: &BTreeMap<String, FileCoverage>,
    source: impl Fn(&str) -> String,
) -> String {
    let mut out = format!("{HEADER}\n");
    for (file, coverage) in files {
        let counts = coverage.counts();
        let branches: Vec<String> = coverage
            .branches
            .iter()
            .filter(|(_, covered)| **covered)
            .map(|(branch, _)| format_branch(*branch))
            .collect();
        let _ = writeln!(
            out,
            "{file}\t{}\t{}/{}\t{}/{}\t{}\t{}",
            source(file),
            counts.lines.0,
            counts.lines.1,
            counts.branches.0,
            counts.branches.1,
            ranges(coverage.covered_lines()),
            branches.join(",")
        );
    }
    out
}

fn ratio(text: &str) -> Option<(usize, usize)> {
    let (covered, total) = text.split_once('/')?;
    Some((covered.parse().ok()?, total.parse().ok()?))
}

fn parse_row(columns: &[&str]) -> Option<(String, Row)> {
    let [file, source, lines, branches, covered, covered_branches] = columns else {
        return None;
    };
    let covered_branches = covered_branches
        .split(',')
        .filter(|item| !item.is_empty())
        .map(parse_branch)
        .collect::<Option<_>>()?;
    Some((
        (*file).to_owned(),
        Row {
            source: (*source).to_owned(),
            counts: Counts {
                lines: ratio(lines)?,
                branches: ratio(branches)?,
            },
            covered_lines: parse_ranges(covered)?,
            covered_branches,
        },
    ))
}

/// Parse a baseline written by [`render`].
pub(crate) fn parse(text: &str) -> Result<BTreeMap<String, Row>> {
    let mut rows = BTreeMap::new();
    for (number, line) in text.lines().enumerate().skip(1) {
        let columns: Vec<&str> = line.split('\t').collect();
        let (file, row) = parse_row(&columns)
            .ok_or_else(|| invalid(format!("lines line {}: {line:?}", number + 1)))?;
        rows.insert(file, row);
    }
    Ok(rows)
}

/// Whether `now` covers a smaller share than `before`.
pub(crate) fn dropped(before: (usize, usize), now: (usize, usize)) -> bool {
    let (before_covered, before_total) = before;
    let (now_covered, now_total) = now;
    if now_total == 0 {
        return before_covered > 0;
    }
    // Cross-multiplied, so equal ratios over different totals compare equal.
    (now_covered as u128) * (before_total as u128) < (before_covered as u128) * (now_total as u128)
}

/// Every baseline file whose coverage regressed. `source` gives a file's
/// current hash, `None` when the file is gone, which is not a regression.
/// `unstable` gives the line and branch rows of each file's
/// [`unstable`](super::unstable) regions: a changed file still measures them,
/// while its baseline counts leave them out, so they come off its totals.
pub(crate) fn regressions(
    baseline: &BTreeMap<String, Row>,
    current: &BTreeMap<String, FileCoverage>,
    source: impl Fn(&str) -> Option<String>,
    unstable: &BTreeMap<String, (usize, usize)>,
) -> Vec<String> {
    let mut found = Vec::new();
    for (file, row) in baseline {
        let Some(hash) = source(file) else {
            continue;
        };
        let Some(coverage) = current.get(file) else {
            if row.counts.lines.0 > 0 {
                found.push(format!("{file}: no longer measured"));
            }
            continue;
        };
        if hash == row.source {
            // Unchanged source: the same line numbers, so compare the sets.
            // Only what is instrumented now counts, so code another
            // platform compiles out is not a loss.
            let lines: Vec<u32> = row
                .covered_lines
                .iter()
                .copied()
                .filter(|line| coverage.lines.get(line) == Some(&false))
                .collect();
            if !lines.is_empty() {
                found.push(format!("{file}: lines {} no longer covered", ranges(lines)));
            }
            let branches: Vec<String> = row
                .covered_branches
                .iter()
                .filter(|branch| coverage.branches.get(branch) == Some(&false))
                .map(|branch| format_branch(*branch))
                .collect();
            if !branches.is_empty() {
                found.push(format!(
                    "{file}: branches {} no longer covered",
                    branches.join(",")
                ));
            }
            continue;
        }
        let mut now = coverage.counts();
        if let Some((lines, branches)) = unstable.get(file) {
            now.lines.1 = now.lines.1.saturating_sub(*lines);
            now.branches.1 = now.branches.1.saturating_sub(*branches);
        }
        for (what, before, now) in [
            ("lines", row.counts.lines, now.lines),
            ("branches", row.counts.branches, now.branches),
        ] {
            if dropped(before, now) {
                found.push(format!(
                    "{file}: {what} {}/{} -> {}/{}",
                    before.0, before.1, now.0, now.1
                ));
            }
        }
    }
    found
}

/// The lines and branches an unchanged file's baseline covered that `current`
/// instruments and no longer covers.
pub(crate) fn lost_regions(
    baseline: &BTreeMap<String, Row>,
    current: &BTreeMap<String, FileCoverage>,
    source: impl Fn(&str) -> Option<String>,
) -> Vec<(String, super::unstable::Region)> {
    use super::unstable::Region;
    let mut found = Vec::new();
    for (file, row) in baseline {
        let (Some(coverage), Some(hash)) = (current.get(file), source(file)) else {
            continue;
        };
        if hash != row.source {
            continue;
        }
        for line in &row.covered_lines {
            if coverage.lines.get(line) == Some(&false) {
                found.push((file.clone(), Region::Line(*line)));
            }
        }
        for branch in &row.covered_branches {
            if coverage.branches.get(branch) == Some(&false) {
                found.push((file.clone(), Region::Branch(*branch)));
            }
        }
    }
    found
}

/// Covered and instrumented totals per crate.
pub(crate) fn per_crate(files: &BTreeMap<String, FileCoverage>) -> BTreeMap<String, Counts> {
    let mut totals: BTreeMap<String, Counts> = BTreeMap::new();
    for (file, coverage) in files {
        let name = file
            .strip_prefix("crates/")
            .and_then(|rest| rest.split('/').next())
            .unwrap_or("rig");
        let counts = coverage.counts();
        let total = totals.entry(name.to_owned()).or_default();
        total.lines.0 += counts.lines.0;
        total.lines.1 += counts.lines.1;
        total.branches.0 += counts.branches.0;
        total.branches.1 += counts.branches.1;
    }
    totals
}

fn percent((covered, total): (usize, usize)) -> f64 {
    if total == 0 {
        0.0
    } else {
        (covered as f64) * 100.0 / (total as f64)
    }
}

fn print_totals(files: &BTreeMap<String, FileCoverage>) {
    for (name, counts) in per_crate(files) {
        println!(
            "lines {name}: lines {}/{} ({:.1}%), branches {}/{} ({:.1}%)",
            counts.lines.0,
            counts.lines.1,
            percent(counts.lines),
            counts.branches.0,
            counts.branches.1,
            percent(counts.branches)
        );
    }
}

/// The arguments and environment of the instrumented nextest run.
fn llvm_cov(root: &Path, extra: &[&str]) -> Result<Command> {
    let mut command = Command::new("cargo");
    command.args([
        "llvm-cov",
        "nextest",
        "--branch",
        "--locked",
        "--workspace",
        "--all-features",
        "--no-fail-fast",
    ]);
    for package in EXCLUDED_PACKAGES {
        command.args(["--exclude", package]);
    }
    command
        .args(extra)
        .current_dir(root)
        .env("RUSTC_BOOTSTRAP", instrumented_crates(root)?)
        .env("NEXTEST_PROFILE", "local")
        .env("RIG_PROVIDER_TEST_MODE", "replay")
        .env_remove("RIG_REGENERATE_GOLDEN")
        .env_remove("NEXTEST_RETRIES");
    Ok(command)
}

/// The crates cargo-llvm-cov instruments, comma-separated. Naming them in
/// `RUSTC_BOOTSTRAP` admits `-Zcoverage-options=branch` for exactly those
/// crates and leaves every dependency on the stable compiler.
fn instrumented_crates(root: &Path) -> Result<String> {
    let env = output(root, "cargo", &["llvm-cov", "show-env"])?;
    env.lines()
        .find_map(|line| line.strip_prefix("__CARGO_LLVM_COV_RUSTC_WRAPPER_CRATE_NAMES="))
        .map(|names| names.trim_matches('\'').to_owned())
        .ok_or_else(|| invalid("cargo llvm-cov show-env named no instrumented crates"))
}

/// The 64-bit hash of a file's bytes, `None` when it cannot be read.
fn source_hash(root: &Path, file: &str) -> Option<String> {
    let bytes = std::fs::read(root.join(file)).ok()?;
    Some(hash(&String::from_utf8_lossy(&bytes)))
}

/// A file's text and its hash, `None` when it cannot be read.
fn source_text(root: &Path, file: &str) -> Option<(String, String)> {
    let bytes = std::fs::read(root.join(file)).ok()?;
    let text = String::from_utf8_lossy(&bytes).into_owned();
    let source = hash(&text);
    Some((text, source))
}

/// One instrumented run's report. `clean` rebuilds the instrumented crates
/// first, as cargo-llvm-cov does by default.
fn run_once(root: &Path, clean: bool) -> Result<BTreeMap<String, FileCoverage>> {
    let report = target_dir(root).join("coverage/lcov.info");
    if let Some(dir) = report.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let report_arg = report.to_string_lossy().into_owned();
    let mut extra = vec!["--lcov", "--output-path", &report_arg];
    if !clean {
        extra.push("--no-clean");
    }
    let status = llvm_cov(root, &extra)?.status()?;
    // The raw profiles are no longer needed once the report is written.
    let _ = Command::new("cargo")
        .args(["llvm-cov", "clean", "--profraw-only", "--workspace"])
        .current_dir(root)
        .status();
    remove_stray_profiles(root)?;
    if !status.success() {
        return Err(invalid("the instrumented test run failed"));
    }
    let text = std::fs::read_to_string(&report)?;
    let mut files = parse_lcov(&text, &root.canonicalize()?);
    let macros = proc_macro_dirs(root)?;
    files.retain(|file, _| !macros.iter().any(|dir| file.starts_with(dir.as_str())));
    Ok(files)
}

/// The source directories (`crates/x/`) of the workspace's proc-macro
/// crates. Their code runs inside the compiler while the tests build, so
/// its coverage depends on what the build recompiled, not on any test.
fn proc_macro_dirs(root: &Path) -> Result<Vec<String>> {
    let metadata: Value = serde_json::from_str(&output(
        root,
        "cargo",
        &["metadata", "--locked", "--no-deps", "--format-version", "1"],
    )?)?;
    let root = root.canonicalize()?;
    let mut dirs = Vec::new();
    let packages = metadata.get("packages").and_then(Value::as_array);
    for package in packages.into_iter().flatten() {
        let targets = package.get("targets").and_then(Value::as_array);
        let is_macro = targets.into_iter().flatten().any(|target| {
            target
                .get("kind")
                .and_then(Value::as_array)
                .is_some_and(|kinds| kinds.iter().any(|kind| kind == "proc-macro"))
        });
        let dir = package
            .get("manifest_path")
            .and_then(Value::as_str)
            .and_then(|manifest| Path::new(manifest).parent())
            .and_then(|dir| dir.strip_prefix(&root).ok())
            .map(|dir| format!("{}/", dir.to_string_lossy().replace('\\', "/")));
        if let (true, Some(dir)) = (is_macro, dir) {
            dirs.push(dir);
        }
    }
    Ok(dirs)
}

/// What [`measure`] found: the rendered measurement, its regressions, and
/// when writing a baseline, the [`unstable`](super::unstable) rows to write
/// beside it.
pub(crate) struct Measured {
    pub(crate) text: String,
    pub(crate) lost: Vec<String>,
    pub(crate) unstable: Option<String>,
}

/// Measure over `runs` runs, keeping what every run covered, print per-crate
/// totals, and compare with `baseline` when given. The regions `unstable`
/// names are left out of the measurement. Without a baseline, the regions
/// the runs disagree on join `unstable`, and the rows to write come back.
pub(crate) fn measure(
    root: &Path,
    baseline: Option<&str>,
    unstable: &str,
    runs: usize,
) -> Result<Measured> {
    let rows = super::unstable::parse(unstable)?;
    let mut files = run_once(root, true)?;
    let mut union = files.clone();
    for run in 1..runs {
        let next = run_once(root, false)?;
        let mut flaky = 0;
        for (file, coverage) in &mut files {
            if let Some(other) = next.get(file) {
                let before = coverage.counts();
                coverage.intersect(other);
                if coverage.counts() != before {
                    flaky += 1;
                }
            }
        }
        for (file, coverage) in next {
            union.entry(file).or_default().unite(&coverage);
        }
        println!(
            "lines: run {} of {runs}; {flaky} files covered less than before",
            run + 1
        );
    }
    let source = |file: &str| source_hash(root, file);
    let (rows, written) = match baseline {
        Some(_) => (rows, None),
        None => {
            let found = super::unstable::disagreements(&union, &files);
            let (rows, notes) =
                super::unstable::refresh(&rows, &found, |file| source_text(root, file));
            for note in notes {
                println!("{note}");
            }
            let text = super::unstable::render(&rows);
            (rows, Some(text))
        }
    };
    super::unstable::exclude(&mut files, &rows, source);
    println!(
        "lines: {} unstable regions held out ({})",
        rows.len(),
        super::unstable::FILE
    );
    print_totals(&files);
    let lost = match baseline {
        Some(text) => {
            let baseline = parse(text)?;
            let mut lost = super::unstable::problems(&rows, |file| {
                baseline.get(file).map(|row| row.source.clone())
            });
            lost.extend(regressions(
                &baseline,
                &files,
                source,
                &super::unstable::per_file(&rows),
            ));
            let candidates = lost_regions(&baseline, &files, source);
            if !candidates.is_empty() {
                // A region a race reaches only sometimes is adopted by adding
                // these rows, each with its reason, to the committed file.
                let (rows, _) =
                    super::unstable::refresh(&[], &candidates, |file| source_text(root, file));
                println!(
                    "lines: if a race, not a test change, lost these regions, add these rows \
                     with a reason to {}:\n{}",
                    super::unstable::FILE,
                    super::unstable::render(&rows)
                        .lines()
                        .skip(1)
                        .collect::<Vec<_>>()
                        .join("\n")
                );
            }
            lost
        }
        None => Vec::new(),
    };
    Ok(Measured {
        text: render(&files, |file| source_hash(root, file).unwrap_or_default()),
        lost,
        unstable: written,
    })
}

/// Delete the `default_*.profraw` files that test children started with a
/// cleared environment leave in their working directory.
fn remove_stray_profiles(root: &Path) -> Result<()> {
    let stray = output(
        root,
        "git",
        &[
            "ls-files",
            "-z",
            "--others",
            "--exclude-standard",
            "--",
            "*.profraw",
        ],
    )?;
    for path in stray.split('\0').filter(|path| !path.is_empty()) {
        let is_default = Path::new(path)
            .file_name()
            .is_some_and(|name| name.to_string_lossy().starts_with("default_"));
        if is_default {
            std::fs::remove_file(root.join(path))?;
        }
    }
    Ok(())
}

/// The directory of the toolchain's `llvm-profdata` and `llvm-cov`.
fn llvm_bin(root: &Path) -> Result<PathBuf> {
    let sysroot = output(root, "rustc", &["--print", "sysroot"])?;
    let version = output(root, "rustc", &["-vV"])?;
    let host = version
        .lines()
        .find_map(|line| line.strip_prefix("host: "))
        .ok_or_else(|| invalid("rustc -vV printed no host"))?;
    let bin = Path::new(sysroot.trim())
        .join("lib/rustlib")
        .join(host.trim())
        .join("bin");
    if bin.join("llvm-profdata").exists() {
        Ok(bin)
    } else {
        Err(invalid(format!(
            "{} has no llvm-profdata; run `rustup component add llvm-tools`",
            bin.display()
        )))
    }
}

/// Run every test once under the wrapper, then gather each test's covered
/// lines and branches into `target/coverage/per-test.tsv`.
pub(crate) fn per_test(root: &Path) -> Result<()> {
    let out = target_dir(root).join("coverage/per-test");
    if out.exists() {
        std::fs::remove_dir_all(&out)?;
    }
    std::fs::create_dir_all(out.join("raw"))?;
    let config = out.join("nextest.toml");
    let wrapper = std::env::current_exe()?;
    std::fs::write(
        &config,
        wrapper_config(
            &std::fs::read_to_string(root.join(".config/nextest.toml"))?,
            &wrapper,
        ),
    )?;
    let config_arg = config.to_string_lossy().into_owned();
    let status = llvm_cov(root, &["--no-report", "--config-file", &config_arg])?
        .env(WRAP_OUT, &out)
        .env(WRAP_LLVM, llvm_bin(root)?)
        .status()?;
    remove_stray_profiles(root)?;
    let merged = gather(&out)?;
    let target = target_dir(root).join("coverage/per-test.tsv");
    std::fs::write(&target, &merged.text)?;
    println!(
        "per-test coverage of {} tests in {} ({} bytes)",
        merged.tests,
        target.display(),
        merged.text.len()
    );
    if status.success() {
        Ok(())
    } else {
        Err(invalid("the instrumented test run failed"))
    }
}

/// The repository's nextest configuration with the coverage wrapper
/// applied to every test of the `local` profile.
pub(crate) fn wrapper_config(repository: &str, wrapper: &Path) -> String {
    format!(
        "experimental = [\"wrapper-scripts\"]\n{repository}\n\
         [scripts.wrapper.rig-coverage]\n\
         command = {{ command-line = {:?}, relative-to = \"none\" }}\n\n\
         [[profile.local.scripts]]\n\
         filter = 'all()'\n\
         run-wrapper = 'rig-coverage'\n",
        format!("{} coverage --wrap", wrapper.display())
    )
}

struct Gathered {
    tests: usize,
    text: String,
}

/// Concatenate the per-test files in path order.
fn gather(out: &Path) -> Result<Gathered> {
    let mut text = String::from("binary\ttest\tfile\tcovered lines\tcovered branches\n");
    let mut tests = 0;
    let dir = out.join("tests");
    if dir.exists() {
        for file in files_under(&dir, Some("tsv"))? {
            text.push_str(&std::fs::read_to_string(&file)?);
            tests += 1;
        }
    }
    Ok(Gathered { tests, text })
}

/// The nextest wrapper: run one test process with its own profile directory,
/// convert the profiles to covered lines and branches, delete them, and exit
/// as the test did.
pub(crate) fn wrap(args: &[String]) -> Result<i32> {
    let (binary, rest) = args
        .split_first()
        .ok_or_else(|| invalid("coverage --wrap requires a test binary"))?;
    let out = PathBuf::from(
        std::env::var_os(WRAP_OUT).ok_or_else(|| invalid(format!("{WRAP_OUT} is unset")))?,
    );
    let llvm = PathBuf::from(
        std::env::var_os(WRAP_LLVM).ok_or_else(|| invalid(format!("{WRAP_LLVM} is unset")))?,
    );
    let binary_id = std::env::var("NEXTEST_BINARY_ID").unwrap_or_default();
    let test = std::env::var("NEXTEST_TEST_NAME").unwrap_or_default();
    let raw = out.join("raw").join(std::process::id().to_string());
    std::fs::create_dir_all(&raw)?;
    let status = Command::new(binary)
        .args(rest)
        .env("LLVM_PROFILE_FILE", raw.join("%p-%m.profraw"))
        .status()?;
    let profiles = files_under(&raw, Some("profraw"))?;
    if !profiles.is_empty() {
        let merged = raw.join("merged.profdata");
        let mut merge = Command::new(llvm.join("llvm-profdata"));
        merge
            .args(["merge", "-sparse", "-o"])
            .arg(&merged)
            .args(&profiles);
        if merge.status()?.success() {
            let export = Command::new(llvm.join("llvm-cov"))
                .args([
                    "export",
                    "-format=lcov",
                    "-skip-functions",
                    "-instr-profile",
                ])
                .arg(&merged)
                .arg(binary)
                .output()?;
            let root = std::env::var_os("NEXTEST_WORKSPACE_ROOT")
                .map(PathBuf::from)
                .unwrap_or_default();
            let files = parse_lcov(&String::from_utf8_lossy(&export.stdout), &root);
            let record = per_test_record(&binary_id, &test, &files);
            let dir = out.join("tests").join(sanitize(&binary_id));
            std::fs::create_dir_all(&dir)?;
            // Hashed: test names can be longer than a file name may be.
            std::fs::write(dir.join(format!("{}.tsv", hash(&test))), record)?;
        }
    }
    std::fs::remove_dir_all(&raw)?;
    Ok(exit_code(&status))
}

#[cfg(unix)]
fn exit_code(status: &std::process::ExitStatus) -> i32 {
    use std::os::unix::process::ExitStatusExt;
    status
        .code()
        .or_else(|| status.signal().map(|signal| 128 + signal))
        .unwrap_or(1)
}

#[cfg(not(unix))]
fn exit_code(status: &std::process::ExitStatus) -> i32 {
    status.code().unwrap_or(1)
}

/// One test's rows: a line per production file it covers.
pub(crate) fn per_test_record(
    binary_id: &str,
    test: &str,
    files: &BTreeMap<String, FileCoverage>,
) -> String {
    let mut out = String::new();
    for (file, coverage) in files {
        let lines: BTreeSet<u32> = coverage.covered_lines().collect();
        if lines.is_empty() {
            continue;
        }
        let branches: Vec<String> = coverage
            .branches
            .iter()
            .filter(|(_, covered)| **covered)
            .map(|((line, block, branch), _)| format!("{line}.{block}.{branch}"))
            .collect();
        let _ = writeln!(
            out,
            "{binary_id}\t{test}\t{file}\t{}\t{}",
            ranges(lines),
            branches.join(",")
        );
    }
    out
}

/// A directory name for a binary id.
pub(crate) fn sanitize(name: &str) -> String {
    name.chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.') {
                ch
            } else {
                '~'
            }
        })
        .collect()
}
