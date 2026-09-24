//! `cargo xtask cassette …`: recording, cleanup and review tools for the
//! provider cassette corpus.
//!
//! Every command shares one attempt root with the recorder itself:
//! `RIG_CASSETTE_ATTEMPT_DIR`, else `cassette-attempts` under
//! `CARGO_TARGET_DIR` or `target/`, made absolute. It holds the attempt ledger
//! (`recordings.tsv`), failed recordings, fixtures kept from failed runs and the
//! created-resource ledger (`ledger.jsonl`).

mod goldens;
mod owner;
mod record;
mod scan;
mod spend;

use std::path::{Path, PathBuf};
use std::process::Command;

pub(crate) const USAGE: &str = "\
  cassette owner <provider/scenario.yaml>...
                              print the tests that record each fixture
  cassette record [--cap N] [--pause SECS] [--dry-run] [--no-cleanup] <fixture>...
                              re-record fixtures by owning test, with an attempt
                              ledger and cap, then clean up created state
  cassette spend [--cap-per-wire USD] [--cap-total USD]
                              price every attempt in the ledger
  cassette scan [--base REF] [<fixture>...]
                              scan changed fixtures for credentials and account data
  cassette goldens [--test TARGET]...
                              regenerate effect goldens from replay and revert
                              delivery-only churn
  cassette cleanup [ledger.jsonl]
                              delete provider state the ledger still holds";

/// The attempt root the recorder and these commands share, absolute so a
/// test process running in its crate directory resolves the same place.
pub(crate) fn attempt_root(root: &Path) -> PathBuf {
    let current = std::env::current_dir().unwrap_or_else(|_| root.to_path_buf());
    if let Some(dir) = std::env::var_os("RIG_CASSETTE_ATTEMPT_DIR") {
        return current.join(dir);
    }
    std::env::var_os("CARGO_TARGET_DIR")
        .map_or_else(|| root.join("target"), |dir| current.join(dir))
        .join("cassette-attempts")
}

pub(crate) fn run(root: &Path, args: Vec<String>) -> Result<(), String> {
    let (command, rest) = args
        .split_first()
        .ok_or_else(|| format!("no cassette command given\n{USAGE}"))?;
    match command.as_str() {
        "owner" => owners(root, rest),
        "record" => record::run(root, rest),
        "spend" => spend::run(root, rest),
        "scan" => scan::run(root, rest),
        "goldens" => goldens::run(root, rest),
        "cleanup" => cleanup(root, rest),
        other => Err(format!("unknown cassette command {other:?}\n{USAGE}")),
    }
}

fn owners(root: &Path, fixtures: &[String]) -> Result<(), String> {
    for fixture in fixtures {
        let (provider, scenario) = record::parse_fixture(fixture)
            .ok_or_else(|| format!("not a fixture path: {fixture}"))?;
        let found: Vec<String> = owner::owners(root, &provider, &scenario)?
            .iter()
            .map(|owner| format!("{owner:?}"))
            .collect();
        println!("{fixture}\t{}", found.join("\t"));
    }
    Ok(())
}

/// Run the recorder's cleanup pass over the created-resource ledger.
pub(crate) fn cleanup(root: &Path, args: &[String]) -> Result<(), String> {
    let attempts = attempt_root(root);
    let ledger = args
        .first()
        .map_or_else(|| attempts.join("ledger.jsonl"), PathBuf::from);
    let status = Command::new("cargo")
        .args([
            "run",
            "--quiet",
            "--locked",
            "-p",
            "rig-cassette",
            "--features",
            "http",
            "--example",
            "cassette_tool",
            "--",
            "cleanup",
        ])
        .arg(&ledger)
        .current_dir(root)
        .env("RIG_CASSETTE_ATTEMPT_DIR", &attempts)
        .status()
        .map_err(|error| format!("cargo run: {error}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("cleanup left resources in {}", ledger.display()))
    }
}
