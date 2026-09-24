//! `cargo xtask cassette goldens`: regenerate effect goldens from replay and
//! revert the ones whose only change is delivery-order churn.
//!
//! Regenerating replays every selected target with
//! `RIG_REGENERATE_GOLDEN=1`. A world golden's `header.deliveries` records
//! racy delivery order that replay excludes from comparison, so a golden
//! whose only difference from `HEAD` is that field is restored rather than
//! committed as noise.

use std::path::Path;
use std::process::Command;

use serde_json::Value;

/// Whether `before` and `after` (golden JSON) differ only in
/// `header.deliveries`.
pub(crate) fn delivery_only(before: &str, after: &str) -> bool {
    let strip = |text: &str| -> Option<Value> {
        let mut value: Value = serde_json::from_str(text).ok()?;
        if let Some(header) = value.get_mut("header").and_then(Value::as_object_mut) {
            header.remove("deliveries");
        }
        Some(value)
    };
    before != after && strip(before).is_some_and(|before| Some(before) == strip(after))
}

fn git(root: &Path, args: &[&str]) -> Result<String, String> {
    let output = Command::new("git")
        .args(args)
        .current_dir(root)
        .output()
        .map_err(|error| format!("git: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "git {}: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// Restore every changed golden under the effects tree whose only change
/// from `HEAD` is `header.deliveries`. Returns (reverted, kept).
pub(crate) fn revert_delivery_churn(root: &Path) -> Result<(usize, usize), String> {
    let changed = git(
        root,
        &[
            "diff",
            "--name-only",
            "HEAD",
            "--",
            "crates/rig-cassette/fixtures/effects",
        ],
    )?;
    let (mut reverted, mut kept) = (0, 0);
    for path in changed.lines().filter(|path| path.ends_with(".json")) {
        let before = git(root, &["show", &format!("HEAD:{path}")])?;
        let after = std::fs::read_to_string(root.join(path)).unwrap_or_default();
        if delivery_only(&before, &after) {
            git(root, &["checkout", "HEAD", "--", path])?;
            reverted += 1;
        } else {
            kept += 1;
        }
    }
    Ok((reverted, kept))
}

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
        "http,agent,ecs,bedrock",
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
    let (reverted, kept) = revert_delivery_churn(root)?;
    println!("reverted {reverted} delivery-only golden(s), kept {kept}");
    if status.success() {
        Ok(())
    } else {
        Err("the regeneration run failed".into())
    }
}

#[cfg(test)]
mod tests;
