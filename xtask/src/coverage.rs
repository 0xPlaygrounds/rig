//! `cargo xtask coverage`: the coverage gate. It measures three things and
//! keeps a compact baseline of each under `crates/rig-cassette/coverage/`:
//! line and branch coverage of production code per file (`lines.tsv`), the
//! mutants the fast suites kill in the replay core (`mutants.tsv`), and the
//! request skeletons and reply shapes the cassette corpus records
//! (`shapes.tsv`).
//!
//! Without `--check` the measured parts overwrite their baseline files. With
//! `--check` they are compared instead, and the command fails when a file's
//! line or branch coverage drops, a baseline-killed mutant survives, or a
//! shape loses its last recording. Line coverage and shapes run by default;
//! mutation is opt-in with `--mutants` because it takes hours.
//!
//! ```console
//! cargo xtask coverage --check             # lines and shapes, as CI runs it
//! cargo xtask coverage --check --mutants   # also mutation, for test deletions
//! cargo xtask coverage --only shapes       # refresh one baseline file
//! cargo xtask coverage --per-test          # per-test coverage under target/
//! ```

mod lines;
mod mutants;
mod shapes;
#[cfg(test)]
mod tests;

use std::path::{Path, PathBuf};
use std::time::Instant;

pub(crate) const USAGE: &str = "\
  coverage [--check] [--mutants] [--only lines,shapes,mutants] [--runs N]
           [--sample N] [--jobs N]
                              measure line/branch coverage, mutants and cassette
                              shapes; write the baseline, or compare with --check
  coverage --per-test         line/branch coverage of every test, under target/coverage
";

/// Where the committed baseline lives, relative to the workspace root.
pub(crate) const BASELINE_DIR: &str = "crates/rig-cassette/coverage";

#[derive(Debug, thiserror::Error)]
pub(crate) enum Error {
    #[error("{0}")]
    Invalid(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

impl From<String> for Error {
    fn from(message: String) -> Self {
        Self::Invalid(message)
    }
}

pub(crate) type Result<T> = std::result::Result<T, Error>;

pub(crate) fn invalid(message: impl Into<String>) -> Error {
    Error::Invalid(message.into())
}

/// One measured part of the gate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Part {
    Lines,
    Shapes,
    Mutants,
}

impl Part {
    fn parse(text: &str) -> Result<Self> {
        match text {
            "lines" => Ok(Self::Lines),
            "shapes" => Ok(Self::Shapes),
            "mutants" => Ok(Self::Mutants),
            other => Err(invalid(format!(
                "unknown coverage part {other:?}; expected lines, shapes or mutants"
            ))),
        }
    }

    fn file(self) -> &'static str {
        match self {
            Self::Lines => "lines.tsv",
            Self::Shapes => "shapes.tsv",
            Self::Mutants => "mutants.tsv",
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Options {
    pub(crate) check: bool,
    pub(crate) parts: Vec<Part>,
    pub(crate) per_test: bool,
    /// Mutation samples one mutant in `sample`; `None` keeps the baseline's.
    pub(crate) sample: Option<u64>,
    /// Parallel cargo-mutants jobs.
    pub(crate) jobs: usize,
    /// Instrumented runs whose common coverage is kept: three when writing
    /// the baseline, one when checking, unless `--runs` says otherwise.
    pub(crate) runs: usize,
}

fn positive<T: std::str::FromStr + PartialOrd + Default>(name: &str, text: &str) -> Result<T> {
    text.parse()
        .ok()
        .filter(|value| *value > T::default())
        .ok_or_else(|| invalid(format!("{name} needs a positive number, not {text:?}")))
}

impl Options {
    pub(crate) fn parse(args: &[String]) -> Result<Self> {
        let mut check = false;
        let mut mutants = false;
        let mut only: Option<Vec<Part>> = None;
        let mut per_test = false;
        let mut sample = None;
        let mut jobs = 2;
        let mut runs = None;
        let mut args = args.iter();
        while let Some(arg) = args.next() {
            let mut value = |name: &str| {
                args.next()
                    .ok_or_else(|| invalid(format!("{name} requires a value")))
            };
            match arg.as_str() {
                "--check" => check = true,
                "--mutants" => mutants = true,
                "--per-test" => per_test = true,
                "--only" => {
                    only = Some(
                        value("--only")?
                            .split(',')
                            .map(Part::parse)
                            .collect::<Result<_>>()?,
                    );
                }
                "--sample" => sample = Some(positive("--sample", value("--sample")?)?),
                "--jobs" => jobs = positive("--jobs", value("--jobs")?)?,
                "--runs" => runs = Some(positive("--runs", value("--runs")?)?),
                other => return Err(invalid(format!("unknown coverage option {other}\n{USAGE}"))),
            }
        }
        if per_test && (check || only.is_some() || mutants) {
            return Err(invalid("--per-test takes no other option"));
        }
        let mut parts = only.unwrap_or_else(|| vec![Part::Lines, Part::Shapes]);
        if mutants && !parts.contains(&Part::Mutants) {
            parts.push(Part::Mutants);
        }
        Ok(Self {
            check,
            parts,
            per_test,
            sample,
            jobs,
            runs: runs.unwrap_or(if check { 1 } else { 3 }),
        })
    }
}

pub(crate) fn run(root: &Path, args: &[String]) -> Result<()> {
    let options = Options::parse(args)?;
    if options.per_test {
        return lines::per_test(root);
    }
    let baseline_dir = root.join(BASELINE_DIR);
    let mut failures = Vec::new();
    for part in &options.parts {
        let start = Instant::now();
        let path = baseline_dir.join(part.file());
        let baseline = if options.check {
            Some(std::fs::read_to_string(&path).map_err(|error| {
                invalid(format!(
                    "{}: {error}; run `cargo xtask coverage` to write it",
                    path.display()
                ))
            })?)
        } else {
            None
        };
        let (measured, lost) = match part {
            Part::Lines => lines::measure(root, baseline.as_deref(), options.runs)?,
            Part::Shapes => {
                let current =
                    shapes::collect(&root.join("crates/rig-cassette/fixtures/cassettes"))?;
                for (provider, (requests, replies)) in shapes::summary(&current) {
                    println!(
                        "shapes {provider}: {requests} request skeletons, {replies} reply shapes"
                    );
                }
                let lost = match &baseline {
                    Some(text) => shapes::lost(&shapes::parse(text)?, &current),
                    None => Vec::new(),
                };
                (shapes::render(&current), lost)
            }
            Part::Mutants => mutants::measure(root, baseline.as_deref(), &options)?,
        };
        println!(
            "coverage {:?}: measured in {:.0}s",
            part,
            start.elapsed().as_secs_f64()
        );
        if options.check {
            let scratch = measured_path(root, *part);
            write(&scratch, &measured)?;
            if !lost.is_empty() {
                println!(
                    "coverage {part:?}: {} regressions; the measurement is in {}",
                    lost.len(),
                    scratch.display()
                );
            }
            failures.extend(lost);
        } else {
            write(&path, &measured)?;
            println!("wrote {}", path.display());
        }
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(invalid(format!(
            "the coverage gate failed:\n  {}",
            failures.join("\n  ")
        )))
    }
}

/// The nextest wrapper `--per-test` installs; returns the test's exit code.
pub(crate) fn wrap(args: &[String]) -> Result<i32> {
    lines::wrap(args)
}

/// Where `--check` leaves a part's measurement, beside the build output.
fn measured_path(root: &Path, part: Part) -> PathBuf {
    target_dir(root).join("coverage").join(part.file())
}

/// `CARGO_TARGET_DIR`, else `target/` under the workspace root.
pub(crate) fn target_dir(root: &Path) -> PathBuf {
    std::env::var_os("CARGO_TARGET_DIR").map_or_else(|| root.join("target"), PathBuf::from)
}

fn write(path: &Path, text: &str) -> Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(path, text)?;
    Ok(())
}
