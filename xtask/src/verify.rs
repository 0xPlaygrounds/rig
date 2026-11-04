//! Verification: one list of check definitions (`checks.rs`) that CI runs one
//! per job and that the local planner selects from by what changed. Every
//! selected check executes; nothing is reused or certified across runs.
mod checks;
mod execute;
mod extras;
pub(crate) mod guards;
mod parts;
mod preflight;
mod selection;
#[cfg(test)]
mod tests;
use crate::support::output;
use serde_json::Value;
use std::{collections::BTreeMap, path::Path, process::Command};

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
type Result<T> = std::result::Result<T, Error>;
fn invalid(message: impl Into<String>) -> Error {
    Error::Invalid(message.into())
}
use checks::{Check, Step};
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Changed,
    Pr,
    Full,
    Check,
    /// Print which slow lanes the PR diff selects; CI's plan job reads it.
    Lanes,
    /// Run every test the diff can break: the tests of each package it
    /// changes and of every package that depends on one.
    Affected,
}
#[derive(Debug)]
struct Options {
    mode: Mode,
    base: Option<String>,
    dry_run: bool,
    check: Option<String>,
    /// With `--check`, run only the steps of this part (one CI job).
    part: Option<String>,
}
impl Options {
    fn parse(args: Vec<String>) -> Result<Self> {
        let mut mode = None;
        let mut base = None;
        let mut dry_run = false;
        let mut check = None;
        let mut part = None;
        let mut args = args.into_iter();
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--changed" | "--pr" | "--full" | "--check" | "--lanes" | "--affected" => {
                    if mode.is_some() {
                        return Err(invalid(
                            "select exactly one of --changed, --pr, --full, --lanes, --affected, \
                             --check ID",
                        ));
                    }
                    mode = Some(match arg.as_str() {
                        "--changed" => Mode::Changed,
                        "--pr" => Mode::Pr,
                        "--full" => Mode::Full,
                        "--lanes" => Mode::Lanes,
                        "--affected" => Mode::Affected,
                        _ => Mode::Check,
                    });
                    if arg == "--check" {
                        check = Some(
                            args.next()
                                .ok_or_else(|| invalid("--check requires an ID"))?,
                        );
                    }
                }
                "--base" => {
                    base = Some(
                        args.next()
                            .ok_or_else(|| invalid("--base requires a revision"))?,
                    )
                }
                "--part" => {
                    part = Some(
                        args.next()
                            .ok_or_else(|| invalid("--part requires a name"))?,
                    )
                }
                "--dry-run" => dry_run = true,
                _ => return Err(invalid(format!("unknown verify option {arg}"))),
            }
        }
        let mode = mode.ok_or_else(|| {
            invalid(
                "verify requires --changed, --pr --base REF, --full, --lanes --base REF, \
                 --affected --base REF, or --check ID",
            )
        })?;
        if matches!(mode, Mode::Pr | Mode::Lanes | Mode::Affected) && base.is_none() {
            return Err(invalid(
                "--pr, --lanes and --affected require --base REF (for example origin/main); no \
                 implicit main",
            ));
        }
        if part.is_some() && mode != Mode::Check {
            return Err(invalid("--part requires --check ID"));
        }
        Ok(Self {
            mode,
            base,
            dry_run,
            check,
            part,
        })
    }
}
pub(crate) fn run(root: &Path, args: Vec<String>) -> Result<()> {
    let opts = Options::parse(args)?;
    let metadata: Value = serde_json::from_str(&output(
        root,
        "cargo",
        &["metadata", "--locked", "--no-deps", "--format-version", "1"],
    )?)?;
    let changes = selection::changes(root, &opts)?;
    let mut all = checks::all();
    for check in &mut all {
        if check.steps.iter().any(|s| s.program == "@full-tests") {
            check.steps = parts::steps(&metadata)?;
        }
    }
    let mut plan = selection::plan(root, &metadata, &opts, &changes, &all)?;
    if let Some(part) = &opts.part {
        for check in &mut plan {
            check.steps.retain(|s| s.part.as_ref() == Some(part));
            if check.steps.is_empty() {
                return Err(invalid(format!("check {} has no part {part}", check.id)));
            }
        }
    }
    if opts.mode == Mode::Lanes {
        let has = |id: &str| plan.iter().any(|c| c.id == id);
        println!(
            "full={} floors={}",
            has("full-tests"),
            has("dependency-floors")
        );
        return Ok(());
    }
    println!(
        "Verification {:?}: {} changed paths; {} checks. Replay only; no live recording.",
        opts.mode,
        changes.len(),
        plan.len()
    );
    for check in &plan {
        println!("SELECT {}: {}", check.id, check.reason);
        for step in &check.steps {
            let part = step
                .part
                .as_ref()
                .map(|p| format!("[{p}] "))
                .unwrap_or_default();
            println!(
                "  {part}{} {} {:?}",
                step.program,
                step.args.join(" "),
                step.env
            );
        }
    }
    for check in &all {
        if !plan.iter().any(|c| c.id == check.id) {
            println!("SKIP {}: outside {:?} selection", check.id, opts.mode);
        }
    }
    if opts.dry_run {
        println!("Dry run only; nothing executed.");
        return Ok(());
    }
    execute::run(root, &metadata, &plan)
}
