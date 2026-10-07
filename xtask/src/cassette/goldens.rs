//! `cargo xtask cassette goldens`: regenerate effect goldens from replay.
//!
//! Regenerating replays every selected target with
//! `RIG_REGENERATE_GOLDEN=1`, so each golden holds exactly what its producer
//! logs over its replayed cassette.

use std::path::Path;
use std::process::Command;

pub(crate) fn run(root: &Path, args: &[String]) -> Result<(), String> {
    let mut targets = Vec::new();
    let mut args = args.iter();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--test" => targets.push(args.next().cloned().ok_or("--test needs a target")?),
            other => return Err(format!("unknown argument {other}")),
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
        "http,agent,bedrock",
        "--retries",
        "0",
        "--no-fail-fast",
    ]);
    for target in &targets {
        command.args(["--test", target]);
    }
    let status = command
        .current_dir(root)
        .env("RIG_PROVIDER_TEST_MODE", "replay")
        .env("RIG_REGENERATE_GOLDEN", "1")
        .status()
        .map_err(|error| format!("cargo nextest: {error}"))?;
    if status.success() {
        Ok(())
    } else {
        Err("the regeneration run failed".into())
    }
}
