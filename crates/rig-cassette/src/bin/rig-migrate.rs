//! `rig-migrate FILE...`: rewrite effect logs and rig-ecs checkpoints in
//! place to the format this rig reads. A file already current is left
//! untouched.

use std::process::ExitCode;

use rig_cassette::migrate::{Migration, migrate};

fn main() -> ExitCode {
    let paths: Vec<String> = std::env::args().skip(1).collect();
    if paths.is_empty() {
        eprintln!("usage: rig-migrate FILE...");
        return ExitCode::FAILURE;
    }
    let mut failed = false;
    for path in &paths {
        match migrate_file(path) {
            Ok(Migration::Current) => println!("{path}: current"),
            Ok(Migration::EffectLog { from }) => {
                println!("{path}: effect log migrated from format {from}");
            }
            Ok(Migration::Checkpoint { from }) => {
                println!("{path}: rig-ecs checkpoint migrated from format {from}");
            }
            Err(error) => {
                eprintln!("{path}: {error}");
                failed = true;
            }
        }
    }
    if failed {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}

fn migrate_file(path: &str) -> Result<Migration, Box<dyn std::error::Error>> {
    let text = std::fs::read_to_string(path)?;
    let (document, migration) = migrate(serde_json::from_str(&text)?)?;
    if migration != Migration::Current {
        std::fs::write(
            path,
            format!("{}\n", serde_json::to_string_pretty(&document)?),
        )?;
    }
    Ok(migration)
}
