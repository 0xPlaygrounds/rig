//! Serial execution of the selected checks with live output. A failed step
//! stops the run; nothing records success anywhere.
use super::*;
use std::{fs, process::Stdio, time::Instant};

pub(super) fn tracked_inputs(root: &Path) -> Result<Vec<String>> {
    Ok(output(root, "git", &["ls-files", "-z"])?
        .split('\0')
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .collect())
}
pub(super) fn untracked_inputs(root: &Path) -> Result<Vec<String>> {
    Ok(output(
        root,
        "git",
        &["ls-files", "--others", "--exclude-standard", "-z"],
    )?
    .split('\0')
    .filter(|s| !s.is_empty())
    .map(str::to_owned)
    .collect())
}

/// The command for one step. Verification is always replay, and CLI or
/// environment retry overrides must not defeat the no-retry contract of the
/// guards profile. Model downloads go under the target directory.
pub(super) fn command(root: &Path, step: &Step, target: &Path) -> Command {
    let mut cmd = Command::new(&step.program);
    let cache = target.join("verify/fastembed-cache");
    cmd.args(&step.args)
        .current_dir(root)
        .envs(&step.env)
        .env("RIG_PROVIDER_TEST_MODE", "replay")
        .env("FASTEMBED_CACHE_DIR", &cache)
        .env("HF_HOME", &cache)
        .env_remove("RIG_REGENERATE_GOLDEN")
        .env_remove("NEXTEST_RETRIES");
    cmd
}
fn internal(root: &Path, target: &Path, step: &Step) -> Result<()> {
    match step.program.as_str() {
        "@layout" => crate::test_layout::check(root).map_err(invalid),
        "@scenarios" => crate::scenarios::run(root, Vec::new()).map_err(|e| invalid(e.to_string())),
        "@registrations" => {
            let cargo = Step {
                program: "cargo".into(),
                args: step.args.clone(),
                env: step.env.clone(),
            };
            let result = command(root, &cargo, target)
                .stderr(Stdio::inherit())
                .output()?;
            if !result.status.success() {
                return Err(invalid("registration discovery failed"));
            }
            let file = target.join("verify/registrations.json");
            fs::create_dir_all(file.parent().unwrap_or(target))?;
            fs::write(&file, result.stdout)?;
            crate::scenarios::run(root, vec![file.to_string_lossy().into_owned()])
                .map_err(|e| invalid(e.to_string()))
        }
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
