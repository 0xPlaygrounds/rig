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
//! Not part of the build and not published: the home for source-tree checks
//! that CI runs and that need a real parser rather than a grep.
//!
//! ```console
//! cargo xtask check-test-layout   # fail on inline `mod tests { }`
//! ```

mod test_layout;

use std::path::{Path, PathBuf};
use std::process::ExitCode;

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let task = args.next();

    let result = match task.as_deref() {
        Some("check-test-layout") => test_layout::check(&workspace_root()),
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
usage: cargo xtask <task>

tasks:
  check-test-layout           fail if any crates/*/src file has an inline
                              test-gated `mod x { }` instead of `mod x;`
";

fn workspace_root() -> PathBuf {
    // `xtask/` sits directly under the workspace root by construction, so its
    // parent is the root. A manifest dir with no parent is not a situation this
    // tool can be in, but it is not worth a panic either.
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    manifest.parent().unwrap_or(manifest).to_path_buf()
}
