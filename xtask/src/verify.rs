//! Verification: one list of check definitions (`checks.rs`), grouped into
//! CI jobs, and one planner (`plan.rs`) that selects from them by what a
//! change touches. CI's plan job prints the plan and each CI job runs its
//! group's selected checks; `--affected` runs the same plan locally. Every
//! selected check executes; nothing is reused or certified across runs.
mod checks;
mod execute;
mod extras;
mod graph;
pub(crate) mod guards;
mod plan;
mod preflight;
mod selection;
#[cfg(test)]
mod tests;
use crate::support::output;
use serde_json::Value;
use std::{path::Path, process::Command};

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

/// The jobs only CI runs: the Docker suites and the macOS runner.
const CI_ONLY: [&str; 2] = ["services", "macos"];

#[derive(Clone, Debug, PartialEq, Eq)]
enum Mode {
    /// Print the plan for CI: `plan=<json>` and `jobs=<json>`.
    Plan,
    /// Run the checks of one CI job that the plan in `RIG_CI_PLAN` selects.
    Job(String),
    /// Run every check the diff selects, but the CI-only jobs.
    Affected,
    Full,
    Check(String),
}
#[derive(Debug)]
struct Options {
    mode: Mode,
    base: Option<String>,
    full: bool,
    dry_run: bool,
}
impl Options {
    fn parse(args: Vec<String>) -> Result<Self> {
        let mut mode = None;
        let mut base = None;
        let mut full = false;
        let mut dry_run = false;
        let mut args = args.into_iter();
        while let Some(arg) = args.next() {
            let next = match arg.as_str() {
                "--plan" => Some(Mode::Plan),
                "--affected" => Some(Mode::Affected),
                "--full" if mode == Some(Mode::Plan) => {
                    full = true;
                    None
                }
                "--full" => Some(Mode::Full),
                "--job" | "--check" => {
                    let id = args
                        .next()
                        .ok_or_else(|| invalid(format!("{arg} requires an ID")))?;
                    Some(if arg == "--job" {
                        Mode::Job(id)
                    } else {
                        Mode::Check(id)
                    })
                }
                "--base" => {
                    base = Some(
                        args.next()
                            .ok_or_else(|| invalid("--base requires a revision"))?,
                    );
                    None
                }
                "--dry-run" => {
                    dry_run = true;
                    None
                }
                _ => return Err(invalid(format!("unknown verify option {arg}"))),
            };
            if let Some(next) = next {
                if mode.is_some() {
                    return Err(invalid(
                        "select exactly one of --plan, --job ID, --affected, --full, --check ID",
                    ));
                }
                mode = Some(next);
            }
        }
        let mode = mode.ok_or_else(|| {
            invalid(
                "verify requires --plan (--base REF | --full), --job ID, --affected --base REF, \
                 --full or --check ID",
            )
        })?;
        let needs_base = matches!(mode, Mode::Affected) || (mode == Mode::Plan && !full);
        if needs_base && base.is_none() {
            return Err(invalid(
                "--affected and --plan require --base REF (for example origin/main); no implicit \
                 main",
            ));
        }
        Ok(Self {
            mode,
            base,
            full,
            dry_run,
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
    let graph = graph::Graph::from_metadata(&metadata)?;
    let all = checks::all();
    let (selected, keep_going) = match &opts.mode {
        Mode::Plan => {
            let (plan, why) = if opts.full {
                (plan::full(&all), vec!["full plan".to_owned()])
            } else {
                diff_plan(root, &graph, &all, &opts)?
            };
            for line in why {
                eprintln!("{line}");
            }
            let mut jobs: Vec<&str> = plan.keys().map(|id| checks::job(id)).collect();
            if !opts.full {
                jobs.push("release-docs");
            }
            jobs.sort_unstable();
            jobs.dedup();
            println!("plan={}", plan::encode(&plan));
            println!("jobs={}", serde_json::to_string(&jobs)?);
            return Ok(());
        }
        Mode::Job(job) => {
            let text = std::env::var("RIG_CI_PLAN")
                .map_err(|_| invalid("--job reads the plan from RIG_CI_PLAN, which is unset"))?;
            let plan = plan::decode(&text)?;
            let checks = resolve(&graph, &all, &plan, |id| checks::job(id) == job)?;
            if checks.is_empty() {
                return Err(invalid(format!("the plan selects no check of job {job}")));
            }
            (checks, true)
        }
        Mode::Affected => {
            let (plan, why) = diff_plan(root, &graph, &all, &opts)?;
            for line in why {
                println!("{line}");
            }
            let checks = resolve(&graph, &all, &plan, |id| {
                !CI_ONLY.contains(&checks::job(id))
            })?;
            (checks, false)
        }
        Mode::Full => (all.clone(), false),
        Mode::Check(id) => {
            let check = all
                .iter()
                .find(|c| c.id == *id)
                .cloned()
                .ok_or_else(|| invalid(format!("missing required check {id}")))?;
            (vec![check], false)
        }
    };
    for check in &selected {
        println!("SELECT {}", check.id);
        for step in &check.steps {
            println!("  {} {} {:?}", step.program, step.args.join(" "), step.env);
        }
    }
    if opts.dry_run {
        println!("Dry run only; nothing executed.");
        return Ok(());
    }
    execute::run(root, &metadata, &selected, keep_going)
}

/// The plan for the diff against `--base`.
fn diff_plan(
    root: &Path,
    graph: &graph::Graph,
    all: &[Check],
    opts: &Options,
) -> Result<(plan::Plan, Vec<String>)> {
    let paths = selection::changes(root, opts)?;
    let files: Vec<String> = execute::tracked_inputs(root)?;
    plan::plan(root, graph, all, &files, &paths)
}

/// The checks of `plan` that `keep` admits, in definition order, with the
/// package-scoped ones' steps narrowed to their scope.
fn resolve(
    graph: &graph::Graph,
    all: &[Check],
    plan: &plan::Plan,
    keep: impl Fn(&str) -> bool,
) -> Result<Vec<Check>> {
    let mut out = Vec::new();
    for check in all.iter().filter(|c| keep(&c.id)) {
        let Some(scope) = plan.get(&check.id) else {
            println!("SKIP {}: not selected by the plan", check.id);
            continue;
        };
        let mut check = check.clone();
        if let plan::Scope::Packages(packages) = scope {
            let args = plan::package_args(graph, packages)?;
            let names: Vec<&str> = packages.keys().map(String::as_str).collect();
            check.steps = checks::scoped(&check.id, &args, &names);
            check.reason = format!("{} packages", packages.len());
        }
        out.push(check);
    }
    for id in plan.keys() {
        if !all.iter().any(|c| c.id == *id) {
            return Err(invalid(format!("the plan names unknown check {id}")));
        }
    }
    Ok(out)
}
