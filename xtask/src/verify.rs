//! Conservative verification planning with disposable, local successful-result reuse.
//! This is not a historical evidence archive or a substitute for independent review.
mod checks;
mod execute;
mod selection;
#[cfg(test)]
mod tests;
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
}
#[derive(Debug)]
struct Options {
    mode: Mode,
    base: Option<String>,
    dry_run: bool,
    reuse: bool,
    check: Option<String>,
}
impl Options {
    fn parse(args: Vec<String>) -> Result<Self> {
        let mut mode = None;
        let mut base = None;
        let mut dry_run = false;
        let mut reuse = None;
        let mut check = None;
        let mut args = args.into_iter();
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--changed" | "--pr" | "--full" | "--check" => {
                    if mode.is_some() {
                        return Err(invalid(
                            "select exactly one of --changed, --pr, --full, --check ID",
                        ));
                    }
                    mode = Some(match arg.as_str() {
                        "--changed" => Mode::Changed,
                        "--pr" => Mode::Pr,
                        "--full" => Mode::Full,
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
                "--dry-run" => dry_run = true,
                "--reuse" => reuse = Some(true),
                "--no-reuse" => reuse = Some(false),
                _ => return Err(invalid(format!("unknown verify option {arg}"))),
            }
        }
        let mode=mode.ok_or_else(||invalid("verify requires --changed, --pr --base REF, --full, or --check ID; optional --dry-run and --no-reuse"))?;
        if mode == Mode::Pr && base.is_none() {
            return Err(invalid(
                "--pr requires --base REF (for example origin/feat/effect-bus); no implicit main",
            ));
        }
        if mode == Mode::Full && reuse == Some(true) {
            return Err(invalid("--full always executes; omit --reuse"));
        }
        Ok(Self {
            mode,
            base,
            dry_run,
            reuse: reuse.unwrap_or(mode == Mode::Changed),
            check,
        })
    }
}
fn output(root: &Path, program: &str, args: &[&str]) -> Result<String> {
    let result = Command::new(program)
        .args(args)
        .current_dir(root)
        .output()?;
    if !result.status.success() {
        return Err(invalid(format!(
            "{program} {args:?} failed: {}",
            String::from_utf8_lossy(&result.stderr)
        )));
    }
    String::from_utf8(result.stdout)
        .map_err(|_| invalid("non-UTF-8 command output; cannot safely plan"))
}
pub(crate) fn run(root: &Path, args: Vec<String>) -> Result<()> {
    let opts = Options::parse(args)?;
    let metadata: Value = serde_json::from_str(&output(
        root,
        "cargo",
        &["metadata", "--locked", "--no-deps", "--format-version", "1"],
    )?)?;
    let changes = selection::changes(root, &opts)?;
    let all = checks::all();
    let plan = selection::plan(root, &metadata, &opts, &changes, &all)?;
    println!(
        "Verification {:?}: {} changed paths; {} checks. No live recording or recapture.",
        opts.mode,
        changes.len(),
        plan.len()
    );
    for check in &plan {
        println!("SELECT {}: {}", check.id, check.reason);
        for step in &check.steps {
            println!("  {} {} {:?}", step.program, step.args.join(" "), step.env);
        }
    }
    for check in &all {
        if !plan.iter().any(|c| c.id == check.id) {
            println!(
                "SKIP {}: outside {:?} selection; not certified or reused",
                check.id, opts.mode
            );
        }
    }
    if opts.dry_run {
        return Ok(());
    }
    execute::run(root, &metadata, &opts, &plan)
}
