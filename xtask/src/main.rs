#![cfg_attr(
    test,
    allow(
        clippy::expect_used,
        clippy::indexing_slicing,
        clippy::panic,
        clippy::unwrap_used
    )
)]
//! Workspace maintenance tasks for rig.
//!
//! Unpublished helper for the source-tree checks CI runs, which need a real
//! parser rather than a text search.
//!
//! ```console
//! cargo xtask check-packaging     # fail on stowaways, bloat and unused deps
//! cargo xtask check-test-layout   # fail on inline `mod tests { }`
//! cargo xtask check-wires         # fail if a provider is not a wire
//! cargo xtask cassette record …   # re-record fixtures by owning test
//! ```

mod bevy;
mod cassette;
mod packaging;
mod test_layout;
mod verify;
mod wires;

use std::path::{Path, PathBuf};
use std::process::ExitCode;

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let task = args.next();

    let result = match task.as_deref() {
        Some("verify") => verify::run(&workspace_root(), args.collect()).map_err(|e| e.to_string()),
        Some("check-packaging") => packaging::check(&workspace_root()),
        Some("check-test-layout") => test_layout::check(&workspace_root()),
        Some("check-wires") => wires::check(&workspace_root()),
        Some("cassette") => cassette::run(&workspace_root(), args.collect()),
        Some(other) => Err(format!(
            "unknown task {other:?}\n{USAGE}{}",
            cassette::USAGE
        )),
        None => Err(format!("no task given\n{USAGE}{}", cassette::USAGE)),
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
  verify --changed|--pr|--full|--lanes [--base REF] [--dry-run]  plan and run verification
  verify --check ID           run one check by id (CI runs one per job)
  check-packaging             fail if the published facade carries files that
                              are not its source, if a publishable crate grows
                              past the size ceiling, if a manifest names a
                              dependency its sources never use, or if a facade
                              feature is outside the additivity guard
  check-test-layout           fail if any crates/*/src file has an inline
                              test-gated `mod x { }` instead of `mod x;`
  check-wires                 fail if anything under rig-core's providers/ is
                              not a wire: `.await`, `async`, a transport type
                              parameter, or a consumer-trait impl
";

fn workspace_root() -> PathBuf {
    // `xtask/` sits directly under the workspace root, so its parent is the root.
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    manifest.parent().unwrap_or(manifest).to_path_buf()
}
