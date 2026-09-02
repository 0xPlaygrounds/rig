#![cfg_attr(
    test,
    allow(
        clippy::expect_used,
        clippy::indexing_slicing,
        clippy::panic,
        clippy::unwrap_used,
        clippy::unreachable
    )
)]
//! Workspace maintenance tasks for rig.
//!
//! Not part of the build and not published: this is the home for things that
//! *produce* checked-in artifacts, so that a generated file can say which
//! command generates it and CI can assert the answer has not drifted.
//!
//! ```console
//! cargo xtask generate-provider-aliases          # rewrite the file
//! cargo xtask generate-provider-aliases --check  # fail if it would change
//! cargo xtask check-test-layout                  # fail on inline `mod tests { }`
//! cargo xtask check-sorted-blocks                # fail on an out-of-order marked list
//! ```

mod aliases;
mod reachable;
mod rustdoc;
mod sorted_blocks;
mod test_layout;

use std::path::{Path, PathBuf};
use std::process::ExitCode;

/// The file `generate-provider-aliases` owns.
const ALIAS_TREE: &str = "crates/rig-reqwest/src/providers.rs";

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let task = args.next();
    let check = args.any(|arg| arg == "--check");

    let result = match task.as_deref() {
        Some("generate-provider-aliases") => generate_provider_aliases(check),
        Some("check-test-layout") => test_layout::check(&workspace_root()),
        Some("check-sorted-blocks") => sorted_blocks::check(&workspace_root()),
        Some(other) => Err(format!("unknown task {other:?}\n{USAGE}")),
        None => Err(format!("no task given\n{USAGE}")),
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

const USAGE: &str = "\
usage: cargo xtask <task> [--check]

tasks:
  generate-provider-aliases   regenerate crates/rig-reqwest/src/providers.rs
                              from rig-core's rustdoc output
  check-test-layout           fail if any crates/*/src file has an inline
                              test-gated `mod x { }` instead of `mod x;`
  check-sorted-blocks         fail if a list between `sorted: start` and
                              `sorted: end` markers is not in byte order
";

fn generate_provider_aliases(check: bool) -> Result<(), String> {
    let workspace = workspace_root();
    let path = workspace.join(ALIAS_TREE);
    let generated = aliases::render(&workspace)?;
    let formatted = rustfmt(&workspace, &generated)?;

    if !check {
        std::fs::write(&path, &formatted)
            .map_err(|error| format!("could not write {}: {error}", path.display()))?;
        println!("wrote {ALIAS_TREE}");
        return Ok(());
    }

    let committed = std::fs::read_to_string(&path)
        .map_err(|error| format!("could not read {}: {error}", path.display()))?;
    if committed == formatted {
        println!("ok: {ALIAS_TREE} is up to date");
        return Ok(());
    }

    Err(format!(
        "{ALIAS_TREE} is out of date.\n\n{}\nRun `cargo xtask generate-provider-aliases`.",
        diff_summary(&committed, &formatted)
    ))
}

/// A line-level summary of what regeneration would change — enough for a CI log
/// to name the missing alias without a diff tool.
fn diff_summary(committed: &str, generated: &str) -> String {
    let old: Vec<&str> = committed.lines().collect();
    let new: Vec<&str> = generated.lines().collect();
    let mut out = String::new();
    for line in &new {
        if !old.contains(line) {
            out.push_str("  + ");
            out.push_str(line.trim());
            out.push('\n');
        }
    }
    for line in &old {
        if !new.contains(line) {
            out.push_str("  - ");
            out.push_str(line.trim());
            out.push('\n');
        }
    }
    out
}

/// What to say when rustfmt is not there at all, which is a setup problem
/// rather than anything wrong with the generated source.
const MISSING_RUSTFMT: &str = "\
this task formats its output with rustfmt, which is not available.
Install it with `rustup component add rustfmt`; in CI, ask the rust-setup
action for it with `components: rustfmt`.";

/// Format the generated source the way the repo formats everything else, so the
/// committed file is `cargo fmt --check` clean and reviewable.
fn rustfmt(workspace: &Path, source: &str) -> Result<String, String> {
    use std::io::Write as _;
    use std::process::{Command, Stdio};

    let mut child = Command::new("rustfmt")
        .current_dir(workspace)
        .args(["--edition", "2024", "--emit", "stdout"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("{MISSING_RUSTFMT}\ncould not start rustfmt: {error}"))?;
    child
        .stdin
        .take()
        .ok_or("rustfmt stdin was not piped")?
        .write_all(source.as_bytes())
        .map_err(|error| format!("could not write to rustfmt: {error}"))?;
    let output = child
        .wait_with_output()
        .map_err(|error| format!("rustfmt failed: {error}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        // rustup answers a missing component on stderr with a zero-length
        // format, which reads like a rejection unless it is called out.
        let hint = if stderr.contains("is not installed") {
            MISSING_RUSTFMT
        } else {
            "rustfmt rejected the generated source:"
        };
        return Err(format!("{hint}\n{stderr}"));
    }
    String::from_utf8(output.stdout)
        .map_err(|error| format!("rustfmt output is not UTF-8: {error}"))
}

fn workspace_root() -> PathBuf {
    // `xtask/` sits directly under the workspace root by construction, so its
    // parent is the root. A manifest dir with no parent is not a situation this
    // tool can be in, but it is not worth a panic either.
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    manifest.parent().unwrap_or(manifest).to_path_buf()
}
