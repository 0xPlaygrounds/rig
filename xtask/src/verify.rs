//! Conservative verification planning with disposable, local successful-result reuse.
//! This is not a historical evidence archive or a substitute for independent review.
mod checks;
mod execute;
mod preflight;
mod process;
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
    let result = process::capture(root, program, args)?;
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
    process::install_interrupt_handler()?;
    println!("Planning: cargo metadata --locked --no-deps (dependency resolution may take time)");
    let metadata: Value = serde_json::from_str(&output(
        root,
        "cargo",
        &["metadata", "--locked", "--no-deps", "--format-version", "1"],
    )?)?;
    let changes = selection::changes(
        root,
        &opts,
        metadata
            .get("target_directory")
            .and_then(Value::as_str)
            .map(Path::new),
    )?;
    let all = checks::all();
    let mut plan = selection::plan(root, &metadata, &opts, &changes, &all)?;
    preflight::configure_model_cache(&metadata, &mut plan)?;
    println!(
        "Verification {:?}: {} changed paths; {} checks. No live recording or recapture.",
        opts.mode,
        changes.len(),
        plan.len()
    );
    println!("Changed inputs: {changes:?}");
    println!(
        "Target directory: {}",
        metadata.get("target_directory").unwrap_or(&Value::Null)
    );
    println!(
        "Environment: RIG_PROVIDER_TEST_MODE=replay; unset RIG_REGENERATE_GOLDEN and NEXTEST_RETRIES; other Cargo/environment configuration inherited and fingerprinted."
    );
    for check in &plan {
        println!(
            "SELECT {}: {}; {}",
            check.id,
            check.reason,
            execute::policy(&opts, check)
        );
        for step in &check.steps {
            println!("  {} {} {:?}", step.program, step.args.join(" "), step.env);
        }
    }
    for manifest in preflight::fixture_manifests(&plan) {
        println!(
            "PREPARATION before check fingerprints: cargo metadata --format-version 1 --manifest-path {manifest}; retain or resolve ignored fixture lockfile"
        );
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
        return execute::preview(root, &metadata, &opts, &plan);
    }
    execute::run(root, &metadata, &opts, &plan)
}
