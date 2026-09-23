//! Recorder utilities for the `cargo xtask cassette` commands.
//!
//! `account-failures <cassette.yaml>...` prints each reply the recorder would
//! refuse as an undeclared account failure and exits 1 when there is one.
//! `stored-state <provider> <cassette.yaml>...` prints each cassette that
//! stores Responses state without deleting it and exits 1 when there is one.
//! `cleanup [ledger.jsonl]` deletes every resource the created-resource
//! ledger still holds, with each provider's credential from its usual
//! environment variable, and prints what it did.
//!
//! ```console
//! cargo run -p rig-cassette --features http --example cassette_tool -- cleanup
//! ```

use std::path::PathBuf;
use std::process::ExitCode;

use rig_cassette::http::{cassette_account_failures, cassette_stored_state, ledger};

#[tokio::main]
async fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("account-failures") => account_failures(args.map(PathBuf::from).collect()),
        Some("stored-state") => {
            let provider = args.next().unwrap_or_default();
            stored_state(&provider, args.map(PathBuf::from).collect())
        }
        Some("cleanup") => {
            let path = args.next().map_or_else(ledger::ledger_path, PathBuf::from);
            cleanup(path).await
        }
        _ => {
            eprintln!(
                "usage: cassette_tool account-failures <cassette.yaml>... | \
                 stored-state <provider> <cassette.yaml>... | cleanup [ledger.jsonl]"
            );
            ExitCode::FAILURE
        }
    }
}

fn account_failures(paths: Vec<PathBuf>) -> ExitCode {
    let mut found_any = false;
    for path in paths {
        let contents = match std::fs::read_to_string(&path) {
            Ok(contents) => contents,
            Err(error) => {
                eprintln!("{}: {error}", path.display());
                return ExitCode::FAILURE;
            }
        };
        for found in cassette_account_failures(&contents) {
            found_any = true;
            println!(
                "{}\tinteraction {}\t{}\tstatus {}\t{:?}",
                path.display(),
                found.index,
                found.request,
                found.status,
                found.failure
            );
        }
    }
    if found_any {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}

fn stored_state(provider: &str, paths: Vec<PathBuf>) -> ExitCode {
    let mut found_any = false;
    for path in paths {
        let Ok(contents) = std::fs::read_to_string(&path) else {
            eprintln!("{}: unreadable", path.display());
            return ExitCode::FAILURE;
        };
        let stored = cassette_stored_state(provider, &contents);
        if !stored.is_empty() {
            found_any = true;
            println!("{}\t{}", path.display(), stored.join(","));
        }
    }
    if found_any {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}

async fn cleanup(path: PathBuf) -> ExitCode {
    let pending = ledger::outstanding(&path);
    println!("{} outstanding in {}", pending.len(), path.display());
    let report = ledger::clean_up(&path, ledger::credential_from_env).await;
    println!(
        "deleted {}, already gone {}, failed {}, no credential {}",
        report.deleted, report.gone, report.remaining, report.no_credential
    );
    if report.remaining == 0 {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
