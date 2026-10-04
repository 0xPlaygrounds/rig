//! The mutation kill set of the replay core, from `cargo mutants` with
//! nextest, each mutant tested against its crate's fast suites only: the
//! crate's unit tests and its conformance targets.
//!
//! The replay core has about four thousand mutants, and one rig-core mutant
//! rebuilds rig-core and its test binaries, so the gate samples them. A
//! mutant is selected when the stable hash of its identity is divisible by
//! the sample size, so the selection is deterministic and an edit elsewhere
//! never changes which mutants a file contributes. An identity is the
//! mutant's file and description without its line and column, numbered when
//! a function has several identical mutants. The baseline records the sample
//! size and each selected mutant's outcome, with how many tests failed on it
//! and the first three of them.

#[cfg(test)]
mod tests;

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::Path;
use std::process::Command;
use std::time::Instant;

use serde_json::Value;

use super::shapes::hash;
use super::{Options, Result, invalid, target_dir};
use crate::support::output;

/// The sample size when neither the options nor a baseline give one.
const DEFAULT_SAMPLE: u64 = 4;

/// The baseline file's column header, after its `sample` line.
pub(crate) const HEADER: &str = "mutant\toutcome\tfailed tests\tfirst failed";

/// How many failed tests a baseline row names; the rest are only counted.
const NAMED_KILLERS: usize = 3;

/// Test files under the mutated globs that are not mutated.
const EXCLUDE: &[&str] = &["**/tests.rs", "**/*_tests.rs", "**/tests/**"];

/// One mutated package: the files it mutates and the suites it runs.
pub(crate) struct Group {
    pub(crate) package: &'static str,
    pub(crate) files: &'static [&'static str],
    pub(crate) test_args: &'static [&'static str],
}

/// The replay core, per package.
pub(crate) const GROUPS: &[Group] = &[
    Group {
        package: "rig-core",
        files: &[
            "crates/rig-core/src/completion/history.rs",
            "crates/rig-core/src/completion/history/**",
            "crates/rig-core/src/operation/completion.rs",
            "crates/rig-core/src/operation/completion/**",
            "crates/rig-core/src/streaming/**",
            "crates/rig-core/src/providers/**",
        ],
        test_args: &[
            "--lib",
            "--test",
            "driver_adoption",
            "--test",
            "streaming_conformance_websocket",
        ],
    },
    Group {
        package: "rig-bedrock",
        files: &["crates/rig-bedrock/src/**"],
        test_args: &["--lib", "--test", "history_conformance"],
    },
    Group {
        package: "rig-agent",
        files: &["crates/rig-agent/src/run/**"],
        test_args: &["--lib"],
    },
    Group {
        package: "rig-ecs",
        files: &["crates/rig-ecs/src/systems/**"],
        test_args: &["--lib"],
    },
];

/// How a mutant fared.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Outcome {
    Caught,
    Missed,
    Timeout,
    Unviable,
}

impl Outcome {
    fn as_str(self) -> &'static str {
        match self {
            Self::Caught => "caught",
            Self::Missed => "missed",
            Self::Timeout => "timeout",
            Self::Unviable => "unviable",
        }
    }

    fn parse(text: &str) -> Option<Self> {
        match text {
            "caught" | "CaughtMutant" => Some(Self::Caught),
            "missed" | "MissedMutant" => Some(Self::Missed),
            "timeout" | "Timeout" => Some(Self::Timeout),
            "unviable" | "Unviable" => Some(Self::Unviable),
            _ => None,
        }
    }

    /// A timeout counts as killed: the mutant did not pass the suites.
    fn killed(self) -> bool {
        matches!(self, Self::Caught | Self::Timeout)
    }
}

/// A tested mutant.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Tested {
    pub(crate) outcome: Outcome,
    /// How many tests failed on the mutant.
    pub(crate) failed: usize,
    /// The first of them in name order, at most [`NAMED_KILLERS`].
    pub(crate) killers: BTreeSet<String>,
}

impl Tested {
    pub(crate) fn new(outcome: Outcome, mut killers: BTreeSet<String>) -> Self {
        let failed = killers.len();
        while killers.len() > NAMED_KILLERS {
            killers.pop_last();
        }
        Self {
            outcome,
            failed,
            killers,
        }
    }
}

/// A baseline or a measurement: the sample size and every selected mutant.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct KillSet {
    pub(crate) sample: u64,
    pub(crate) mutants: BTreeMap<String, Tested>,
}

/// `name` without the `:line:col` after its file.
pub(crate) fn strip_position(name: &str) -> String {
    let Some((file, rest)) = name.split_once(".rs:") else {
        return name.to_owned();
    };
    let mut parts = rest.splitn(3, ':');
    match (parts.next(), parts.next(), parts.next()) {
        (Some(line), Some(col), Some(description))
            if line.chars().all(|ch| ch.is_ascii_digit())
                && col.chars().all(|ch| ch.is_ascii_digit()) =>
        {
            format!("{file}.rs:{description}")
        }
        _ => name.to_owned(),
    }
}

/// The identity of every listed mutant name, in list order: the name
/// without position, with `#n` for the n-th repeat of the same one.
pub(crate) fn identities(names: &[String]) -> Vec<(String, String)> {
    let mut seen: BTreeMap<String, usize> = BTreeMap::new();
    names
        .iter()
        .map(|name| {
            let base = strip_position(name);
            let count = seen.entry(base.clone()).or_default();
            *count += 1;
            let identity = if *count == 1 {
                base
            } else {
                format!("{base} #{count}")
            };
            (name.clone(), identity)
        })
        .collect()
}

/// Whether the sample of size `sample` selects `identity`.
pub(crate) fn selected(identity: &str, sample: u64) -> bool {
    u64::from_str_radix(&hash(identity), 16).is_ok_and(|value| value % sample == 0)
}

/// The tests a nextest log reports as failed, timed out or killed by a
/// signal, as `binary-id test-name`.
pub(crate) fn killers(log: &str) -> BTreeSet<String> {
    let mut found = BTreeSet::new();
    for line in log.lines() {
        let line = line.trim_start();
        let Some((status, rest)) = line.split_once(" [") else {
            continue;
        };
        if !(status == "FAIL" || status == "TIMEOUT" || status.starts_with("SIG")) {
            continue;
        }
        let Some((_, rest)) = rest.split_once("] ") else {
            continue;
        };
        let rest = rest.trim_start();
        // Progress counters such as `(3/120)` precede the test.
        let rest = match rest.strip_prefix('(').and_then(|r| r.split_once(") ")) {
            Some((_, after)) => after,
            None => rest,
        };
        if !rest.is_empty() {
            found.insert(rest.trim().to_owned());
        }
    }
    found
}

/// The baseline text for `set`.
pub(crate) fn render(set: &KillSet) -> String {
    let mut out = format!("sample\t{}\n{HEADER}\n", set.sample);
    for (identity, tested) in &set.mutants {
        let killers: Vec<&str> = tested.killers.iter().map(String::as_str).collect();
        let _ = writeln!(
            out,
            "{identity}\t{}\t{}\t{}",
            tested.outcome.as_str(),
            tested.failed,
            killers.join(", ")
        );
    }
    out
}

/// Parse a baseline written by [`render`].
pub(crate) fn parse(text: &str) -> Result<KillSet> {
    let mut lines = text.lines();
    let sample = lines
        .next()
        .and_then(|line| line.strip_prefix("sample\t"))
        .and_then(|value| value.parse().ok())
        .ok_or_else(|| invalid("mutants baseline: missing `sample` line"))?;
    let mut mutants = BTreeMap::new();
    for (number, line) in lines.enumerate().skip(1) {
        let columns: Vec<&str> = line.split('\t').collect();
        let row = match columns.as_slice() {
            [identity, outcome, failed, killed_by] => failed
                .parse()
                .ok()
                .map(|failed: usize| (identity, outcome, failed, killed_by)),
            _ => None,
        };
        let (identity, outcome, failed, killed_by) =
            row.ok_or_else(|| invalid(format!("mutants line {}: {line:?}", number + 3)))?;
        let outcome = Outcome::parse(outcome)
            .ok_or_else(|| invalid(format!("mutants line {}: {outcome:?}", number + 3)))?;
        let killers = killed_by
            .split(", ")
            .filter(|killer| !killer.is_empty())
            .map(str::to_owned)
            .collect();
        mutants.insert(
            (*identity).to_owned(),
            Tested {
                outcome,
                failed,
                killers,
            },
        );
    }
    Ok(KillSet { sample, mutants })
}

/// Every baseline-killed mutant that the current run let survive. A mutant
/// the current source no longer has is gone, not a regression.
pub(crate) fn survivors(baseline: &KillSet, current: &KillSet) -> Vec<String> {
    baseline
        .mutants
        .iter()
        .filter(|(_, tested)| tested.outcome.killed())
        .filter_map(|(identity, _)| {
            let now = current.mutants.get(identity)?;
            (!now.outcome.killed())
                .then(|| format!("mutant survives: {identity} ({})", now.outcome.as_str()))
        })
        .collect()
}

fn mutants_command(root: &Path, group: &Group) -> Command {
    let mut command = Command::new("cargo");
    command
        .args(["mutants", "--package", group.package, "--all-features"])
        .current_dir(root)
        .env("RIG_PROVIDER_TEST_MODE", "replay")
        .env_remove("RIG_REGENERATE_GOLDEN")
        .env_remove("NEXTEST_RETRIES");
    for file in group.files {
        command.args(["-f", file]);
    }
    for file in EXCLUDE {
        command.args(["-e", file]);
    }
    command
}

/// The listed mutants of `group`, as `(name, identity)` in list order.
fn list(root: &Path, group: &Group) -> Result<Vec<(String, String)>> {
    let listed = mutants_command(root, group).arg("--list").output()?;
    if !listed.status.success() {
        return Err(invalid(format!(
            "cargo mutants --list failed for {}:\n{}",
            group.package,
            String::from_utf8_lossy(&listed.stderr)
        )));
    }
    let names: Vec<String> = String::from_utf8_lossy(&listed.stdout)
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(str::to_owned)
        .collect();
    Ok(identities(&names))
}

/// Run the selected mutants of `group` and read their outcomes.
///
/// cargo-mutants' `--re` does not filter field-deletion mutants, so the
/// sample is applied through `--iterate` instead: every listed mutant outside
/// it is written to the output's `previously_caught.txt`, which cargo-mutants
/// skips by exact name.
fn test_group(
    root: &Path,
    group: &Group,
    listed: &[(String, String)],
    chosen: &[(String, String)],
    jobs: usize,
) -> Result<BTreeMap<String, Tested>> {
    if chosen.is_empty() {
        return Ok(BTreeMap::new());
    }
    let out = target_dir(root)
        .join("coverage/mutants")
        .join(group.package);
    if out.exists() {
        std::fs::remove_dir_all(&out)?;
    }
    let dir = out.join("mutants.out");
    std::fs::create_dir_all(&dir)?;
    let skipped: String = listed
        .iter()
        .filter(|entry| !chosen.contains(entry))
        .map(|(name, _)| format!("{name}\n"))
        .collect();
    std::fs::write(dir.join("previously_caught.txt"), skipped)?;
    let mut command = mutants_command(root, group);
    command
        .args([
            "--test-tool",
            "nextest",
            "--no-shuffle",
            "--iterate",
            "--jobs",
        ])
        .arg(jobs.to_string())
        .arg("--output")
        .arg(&out);
    for arg in group.test_args {
        command.arg(format!("--cargo-test-arg={arg}"));
    }
    // cargo-mutants exits non-zero whenever a mutant is missed; the outcomes
    // file is the result.
    let _ = command.status()?;
    let outcomes: Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("outcomes.json"))?)?;
    let by_name: BTreeMap<&str, &str> = chosen
        .iter()
        .map(|(name, identity)| (name.as_str(), identity.as_str()))
        .collect();
    let mut tested = BTreeMap::new();
    let entries = outcomes.get("outcomes").and_then(Value::as_array);
    for entry in entries.into_iter().flatten() {
        let field = |pointer: &str| entry.pointer(pointer).and_then(Value::as_str);
        let Some(name) = field("/scenario/Mutant/name") else {
            if field("/scenario") == Some("Baseline") && field("/summary") != Some("Success") {
                return Err(invalid(format!(
                    "{}: the unmutated suites fail; see {}",
                    group.package,
                    dir.display()
                )));
            }
            continue;
        };
        let Some(identity) = by_name.get(name) else {
            continue;
        };
        let Some(outcome) = field("/summary").and_then(Outcome::parse) else {
            continue;
        };
        let killers = field("/log_path")
            .and_then(|log| std::fs::read_to_string(dir.join(log)).ok())
            .map(|log| killers(&log))
            .unwrap_or_default();
        tested.insert((*identity).to_owned(), Tested::new(outcome, killers));
    }
    Ok(tested)
}

/// Measure the kill set, print its totals, and compare with `baseline`.
pub(crate) fn measure(
    root: &Path,
    baseline: Option<&str>,
    options: &Options,
) -> Result<(String, Vec<String>)> {
    output(root, "cargo", &["mutants", "--version"])?;
    let baseline = baseline.map(parse).transpose()?;
    let sample = options
        .sample
        .or(baseline.as_ref().map(|set| set.sample))
        .unwrap_or(DEFAULT_SAMPLE);
    if let Some(baseline) = &baseline
        && baseline.sample != sample
    {
        return Err(invalid(format!(
            "--sample {sample} differs from the baseline's {}",
            baseline.sample
        )));
    }
    let mut current = KillSet {
        sample,
        mutants: BTreeMap::new(),
    };
    let start = Instant::now();
    for group in GROUPS {
        let listed = list(root, group)?;
        let chosen: Vec<(String, String)> = listed
            .iter()
            .filter(|(_, identity)| selected(identity, sample))
            .cloned()
            .collect();
        let group_start = Instant::now();
        let tested = test_group(root, group, &listed, &chosen, options.jobs)?;
        let killed = tested.values().filter(|t| t.outcome.killed()).count();
        println!(
            "mutants {}: {} listed, {} sampled, {} tested, {killed} killed, in {:.0}s",
            group.package,
            listed.len(),
            chosen.len(),
            tested.len(),
            group_start.elapsed().as_secs_f64()
        );
        current.mutants.extend(tested);
    }
    let count = |outcome: Outcome| {
        current
            .mutants
            .values()
            .filter(|tested| tested.outcome == outcome)
            .count()
    };
    println!(
        "mutants: {} tested (one in {sample}): {} caught, {} timeout, {} missed, {} unviable, in {:.0}s",
        current.mutants.len(),
        count(Outcome::Caught),
        count(Outcome::Timeout),
        count(Outcome::Missed),
        count(Outcome::Unviable),
        start.elapsed().as_secs_f64()
    );
    let lost = baseline
        .map(|baseline| survivors(&baseline, &current))
        .unwrap_or_default();
    Ok((render(&current), lost))
}
