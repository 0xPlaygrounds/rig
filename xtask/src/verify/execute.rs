//! Serial execution of the selected checks with live output. A failed step
//! stops the run; nothing records success anywhere.
use super::*;
use std::{fs, time::Instant};

/// The paths a `git … -z` command prints, NUL-separated.
pub(super) fn git_paths(root: &Path, args: &[&str]) -> Result<Vec<String>> {
    Ok(output(root, "git", args)?
        .split('\0')
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .collect())
}
pub(super) fn tracked_inputs(root: &Path) -> Result<Vec<String>> {
    git_paths(root, &["ls-files", "-z"])
}
pub(super) fn untracked_inputs(root: &Path) -> Result<Vec<String>> {
    git_paths(root, &["ls-files", "--others", "--exclude-standard", "-z"])
}

/// The command for one step. Verification is always replay, matches request
/// bodies by shape, checks the request snapshots and never rewrites them, and CLI or environment retry
/// overrides must not defeat the no-retry contract of the guards profile.
/// Model downloads go under the target directory.
pub(super) fn command(root: &Path, step: &Step, target: &Path) -> Command {
    let mut cmd = Command::new(&step.program);
    let cache = target.join("verify/fastembed-cache");
    cmd.args(&step.args)
        .current_dir(root)
        .envs(&step.env)
        .env("RIG_PROVIDER_TEST_MODE", "replay")
        .env("RIG_CASSETTE_SNAPSHOTS", "check")
        .env("RIG_CASSETTE_MATCHING", "shape")
        .env("FASTEMBED_CACHE_DIR", &cache)
        .env("HF_HOME", &cache)
        .env_remove("RIG_REGENERATE_GOLDEN")
        .env_remove("NEXTEST_RETRIES");
    cmd
}
fn internal(root: &Path, target: &Path, step: &Step) -> Result<()> {
    match step.program.as_str() {
        "@bevy-sources" => {
            let metadata = output(
                root,
                "cargo",
                &["metadata", "--locked", "--format-version", "1"],
            )?;
            crate::bevy::check(&serde_json::from_str(&metadata)?)
                .map_err(|error| invalid(error.to_string()))
        }
        "@layout" => crate::test_layout::check(root).map_err(invalid),
        "@packaging" => crate::packaging::check(root).map_err(invalid),
        "@wires" => crate::wires::check(root).map_err(invalid),
        "@options-guards" => super::guards::check(root).map_err(invalid),
        "@fixture-paths" => {
            // `cargo test` runs from the crate root and nextest from the
            // workspace root, so a CWD-relative fixture path passes under one
            // and fails under the other. Doc comments are exempt.
            for path in tracked_inputs(root)?
                .into_iter()
                .chain(untracked_inputs(root)?)
                .filter(|p| {
                    p.starts_with("crates/")
                        && p.ends_with(".rs")
                        && (p.contains("/src/") || p.contains("/tests/"))
                })
            {
                let Ok(text) = fs::read_to_string(root.join(&path)) else {
                    continue;
                };
                for line in text.lines().filter(|l| {
                    !l.trim_start().starts_with("///") && !l.trim_start().starts_with("//!")
                }) {
                    if line.contains("\"tests/data") || line.contains("\"./tests/data") {
                        return Err(invalid(format!(
                            "CWD-relative fixture path in {path}; anchor it to CARGO_MANIFEST_DIR"
                        )));
                    }
                }
            }
            Ok(())
        }
        "@ecs-boundary" => {
            // rig-ecs is a library runtime: with every feature on, nothing
            // in its normal dependency tree is a view, the `rig` facade
            // (and its launcher protocol), the app or an HTTP stack, and
            // it touches no files: stores are the app's.
            const FORBIDDEN: [&str; 8] = [
                "rig",
                "rig-harness",
                "rig-tools",
                "rig-reqwest",
                "reqwest",
                "ratatui",
                "crossterm",
                "ignore",
            ];
            let tree = output(
                root,
                "cargo",
                &[
                    "tree",
                    "--locked",
                    "-p",
                    "rig-ecs",
                    "--all-features",
                    "-e",
                    "normal",
                    "--prefix",
                    "none",
                    "--format",
                    "{p}",
                ],
            )?;
            if let Some(name) = tree
                .lines()
                .filter_map(|line| line.split_whitespace().next())
                .find(|name| FORBIDDEN.contains(name))
            {
                return Err(invalid(format!(
                    "rig-ecs depends on `{name}`; the runtime must not depend on a view, the \
                     `rig` facade, the app or an HTTP stack"
                )));
            }
            for path in tracked_inputs(root)?
                .into_iter()
                .chain(untracked_inputs(root)?)
                .filter(|p| p.starts_with("crates/rig-ecs/src/") && p.ends_with(".rs"))
            {
                let Ok(text) = fs::read_to_string(root.join(&path)) else {
                    continue;
                };
                for (number, line) in text.lines().enumerate() {
                    let code = line.split("//").next().unwrap_or_default();
                    if paths(code, "std::fs").next().is_some() {
                        return Err(invalid(format!(
                            "{path}:{}: rig-ecs names `std::fs`; the app's store touches files",
                            number + 1
                        )));
                    }
                }
            }
            Ok(())
        }
        "@plugin-boundary" => plugin_boundary(root),
        "@native-only" => {
            // A native-only crate on wasm must fail with exactly its one
            // `compile_error!` sentence; an item outside the `not(wasm)` gate
            // leaks follow-on errors.
            let package = step
                .args
                .first()
                .ok_or_else(|| invalid("native-only package missing"))?;
            let expected = step
                .args
                .get(1)
                .ok_or_else(|| invalid("native-only diagnostic missing"))?;
            let cargo = Step::new(
                "cargo",
                &[
                    "check",
                    "--locked",
                    "--package",
                    package,
                    "--target",
                    "wasm32-unknown-unknown",
                ],
            )
            .env("CARGO_TERM_COLOR", "never");
            let result = command(root, &cargo, target).output()?;
            let stderr = String::from_utf8_lossy(&result.stderr);
            let count = stderr
                .lines()
                .filter(|l| {
                    (l.starts_with("error:") || l.starts_with("error["))
                        && !l.contains("could not compile")
                })
                .count();
            if result.status.success() || !stderr.contains(expected) || count != 1 {
                print!("{stderr}");
                return Err(invalid(format!(
                    "{package}: expected exactly one native-only diagnostic, got {count}"
                )));
            }
            Ok(())
        }
        _ => Err(invalid(format!(
            "unknown internal command {}",
            step.program
        ))),
    }
}
pub(super) fn run(root: &Path, metadata: &Value, plan: &[Check]) -> Result<()> {
    if plan.is_empty() {
        println!("No executable changes; no verification result claimed.");
        return Ok(());
    }
    let target = Path::new(
        metadata["target_directory"]
            .as_str()
            .ok_or_else(|| invalid("Cargo metadata missing target directory"))?,
    );
    preflight::run(root, plan)?;
    let start = Instant::now();
    for (index, check) in plan.iter().enumerate() {
        println!(
            "RUN {}/{} {} ({:.0}s elapsed)",
            index + 1,
            plan.len(),
            check.id,
            start.elapsed().as_secs_f64()
        );
        let check_start = Instant::now();
        for step in &check.steps {
            println!("STEP {} {}", step.program, step.args.join(" "));
            let outcome = if step.program.starts_with('@') {
                internal(root, target, step)
            } else if command(root, step, target).status()?.success() {
                Ok(())
            } else {
                Err(invalid(format!(
                    "required check {} failed: {} {:?}",
                    check.id, step.program, step.args
                )))
            };
            if let Err(error) = outcome {
                for remaining in plan.iter().skip(index + 1) {
                    println!("NOT RUN {}", remaining.id);
                }
                return Err(error);
            }
        }
        println!(
            "PASS {}: {:.1}s",
            check.id,
            check_start.elapsed().as_secs_f64()
        );
    }
    println!(
        "All {} checks passed in {:.0}s. This does not certify independent review or remote CI.",
        plan.len(),
        start.elapsed().as_secs_f64()
    );
    Ok(())
}

/// A default plugin, and `front`, use rig-harness as a plugin crate does:
/// its prelude (which can re-export only `pub` items), `front` (which has
/// no `pub(crate)` items) and their own modules. Tests are exempt.
fn plugin_boundary(root: &Path) -> Result<()> {
    let ident = |c: char| c.is_alphanumeric() || c == '_';
    for path in tracked_inputs(root)?
        .into_iter()
        .chain(untracked_inputs(root)?)
    {
        let module = path
            .strip_prefix("crates/rig-harness/src/")
            .unwrap_or_default();
        let parts: Vec<&str> = module.split('/').collect();
        let own = match parts.as_slice() {
            _ if !module.ends_with(".rs") || super::guards::is_test_file(module) => continue,
            ["front.rs"] => "front".to_owned(),
            ["plugins", plugin, ..] => format!("plugins::{}", plugin.trim_end_matches(".rs")),
            ["tui", ..] => "tui".to_owned(),
            _ => continue,
        };
        // How many `super::` stay in the plugin's module.
        let depth = (parts.len() + usize::from(own == "tui"))
            .saturating_sub(2 + usize::from(module.ends_with("mod.rs")));
        let allowed = |rest: &str| {
            ["prelude", "front", &own].iter().any(|ok| {
                rest.strip_prefix(ok)
                    .is_some_and(|after| !after.starts_with(ident))
            })
        };
        let text = fs::read_to_string(root.join(&path))?;
        for (number, line) in text.lines().enumerate() {
            let code = line.split("//").next().unwrap_or_default();
            let supers = |rest: &str| rest.split("super::").take_while(|s| s.is_empty()).count();
            if paths(code, "crate::").any(|rest| !allowed(rest))
                || paths(code, "super::").any(|rest| supers(rest) >= depth)
                || (own == "front" && code.contains("pub(crate)"))
            {
                return Err(invalid(format!(
                    "{path}:{}: past rig-harness's public API; a default plugin uses only the \
                     prelude, `front` and its own modules",
                    number + 1
                )));
            }
        }
    }
    Ok(())
}
/// What follows each `prefix` (such as `rig::`) in `code` that starts a
/// path, not the end of a longer name such as `my_rig::`.
fn paths<'a>(code: &'a str, prefix: &'a str) -> impl Iterator<Item = &'a str> {
    code.match_indices(prefix).filter_map(move |(at, _)| {
        let before = code.get(..at)?.chars().next_back();
        let starts = !before.is_some_and(|c| c.is_alphanumeric() || c == '_');
        code.get(at + prefix.len()..).filter(|_| starts)
    })
}
