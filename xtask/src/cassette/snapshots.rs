//! `cargo xtask cassette snapshots`: rewrite or check the request snapshots
//! beside the provider cassettes by replaying them offline.
//!
//! A rewrite deletes every `<fixture>.requests.json`, then replays the
//! cassette suites with `RIG_CASSETTE_SNAPSHOTS=write`, so each session
//! writes the snapshot of a fixture whose requests differ from its recording
//! and a snapshot nothing replays disappears. `--check` replays with
//! `RIG_CASSETTE_SNAPSHOTS=check` and fails when a request differs from its
//! snapshot. Both replay with `RIG_CASSETTE_MATCHING=shape`, so a request
//! that keeps its recording's coarse shape is served and its difference
//! lands in the snapshot. `--test TARGET` limits either to those targets,
//! and a rewrite then deletes nothing first.

use std::path::{Path, PathBuf};
use std::process::Command;

const CASSETTES: &str = "crates/rig-cassette/fixtures/cassettes";
const SUFFIX: &str = ".requests.json";

/// What the command was asked to do.
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct Options {
    pub(crate) check: bool,
    pub(crate) targets: Vec<String>,
}

pub(crate) fn parse(args: &[String]) -> Result<Options, String> {
    let mut options = Options::default();
    let mut args = args.iter();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--check" => options.check = true,
            "--test" => options
                .targets
                .push(args.next().cloned().ok_or("--test needs a target")?),
            other => return Err(format!("unknown argument {other}")),
        }
    }
    Ok(options)
}

/// Every request snapshot under `cassettes`, in path order.
pub(crate) fn snapshot_files(cassettes: &Path) -> Result<Vec<PathBuf>, String> {
    if !cassettes.is_dir() {
        return Ok(Vec::new());
    }
    Ok(crate::support::files_under(cassettes, Some("json"))?
        .into_iter()
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.ends_with(SUFFIX))
        })
        .collect())
}

pub(crate) fn run(root: &Path, args: &[String]) -> Result<(), String> {
    let options = parse(args)?;
    let cassettes = root.join(CASSETTES);
    let mode = if options.check { "check" } else { "write" };
    if !options.check && options.targets.is_empty() {
        for path in snapshot_files(&cassettes)? {
            std::fs::remove_file(&path).map_err(|error| format!("{}: {error}", path.display()))?;
        }
    }
    let mut command = Command::new("cargo");
    command.args([
        "nextest",
        "run",
        "--locked",
        "-p",
        "rig-cassette",
        "--features",
        "http,agent,ecs,bedrock",
        "--retries",
        "0",
        "--no-fail-fast",
    ]);
    for target in &options.targets {
        command.args(["--test", target]);
    }
    let status = command
        .current_dir(root)
        .env("RIG_PROVIDER_TEST_MODE", "replay")
        .env("RIG_CASSETTE_SNAPSHOTS", mode)
        .env("RIG_CASSETTE_MATCHING", "shape")
        .env_remove("RIG_REGENERATE_GOLDEN")
        .status()
        .map_err(|error| format!("cargo nextest: {error}"))?;
    let files = snapshot_files(&cassettes)?;
    let bytes: u64 = files
        .iter()
        .filter_map(|path| std::fs::metadata(path).ok())
        .map(|metadata| metadata.len())
        .sum();
    println!("{} request snapshot(s), {bytes} bytes", files.len());
    if status.success() {
        Ok(())
    } else if options.check {
        Err("replay failed or a request differs from its snapshot".into())
    } else {
        Err("the replay failed, so some snapshots may be missing".into())
    }
}

#[cfg(test)]
mod tests;
