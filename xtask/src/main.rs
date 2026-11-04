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
//! cargo xtask check-stage-order   # fail if a corpus crate's stages go unchecked
//! cargo xtask check-wires         # fail if a provider is not a wire
//! cargo xtask cassette record …   # re-record fixtures by owning test
//! cargo xtask catalog sync         # regenerate rig-core's model catalog
//! ```

mod bevy;
mod cassette;
mod catalog;
mod driver_adoption;
mod packaging;
mod random_ids;
mod stage_order;
mod support;
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
        Some("check-stage-order") => stage_order::check(&workspace_root()),
        Some("check-wires") => wires::check(&workspace_root()),
        Some("cassette") => cassette::run(&workspace_root(), args.collect()),
        Some("catalog") => catalog::run(&workspace_root(), args.collect()),
        Some(other) => Err(format!(
            "unknown task {other:?}\n{USAGE}{}\n{}",
            cassette::USAGE,
            catalog::USAGE
        )),
        None => Err(format!(
            "no task given\n{USAGE}{}\n{}",
            cassette::USAGE,
            catalog::USAGE
        )),
    };
    finish(result)
}

fn finish(result: Result<(), String>) -> ExitCode {
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
  verify --affected --base REF
                              run every check the change selects: the tests of
                              each package that can observe it (before every push)
  verify --plan --base REF|--full
                              print the plan CI's jobs run (`plan=` and `jobs=`)
  verify --job ID             run one CI job's checks from the plan in RIG_CI_PLAN
  verify --full | --check ID  run every check, or one by id
  (each takes --dry-run to print the selection only)
  check-packaging             fail if the published facade carries files that
                              are not its source, if a publishable crate grows
                              past the size ceiling, if a manifest names a
                              dependency its sources never use, or if a facade
                              feature is outside the additivity guard
  check-test-layout           fail if any crates/*/src, plugins/*/src or
                              corpus/*/src file has an inline test-gated
                              `mod x { }` instead of `mod x;`
  check-stage-order           fail if a corpus plugin adds agent-loop systems
                              and no test of it calls assert_stages_ordered
  check-wires                 fail if anything under rig-core's providers/ is
                              not a wire: `.await`, `async`, a transport type
                              parameter, or a consumer-trait impl
";

fn workspace_root() -> PathBuf {
    // `xtask/` sits directly under the workspace root, so its parent is the root.
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    manifest.parent().unwrap_or(manifest).to_path_buf()
}
