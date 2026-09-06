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

mod parity;
mod scenario_inventory;
mod test_layout;

use std::path::{Path, PathBuf};
use std::process::ExitCode;

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let task = args.next();

    let result = match task.as_deref() {
        Some("check-test-layout") => test_layout::check(&workspace_root()),
        Some("parity-batch") => {
            parity::batch::run(&workspace_root(), args.collect()).map_err(|e| e.to_string())
        }
        Some("parity-manifest") => parity::manifest::run(args.collect()).map_err(|e| e.to_string()),
        Some("parity-queue") => parity::queue::run(args.collect()).map_err(|e| e.to_string()),
        Some("parity-review") => parity::review::run(args.collect()).map_err(|e| e.to_string()),
        Some("parity-targets") => parity::targets::run(args.collect()).map_err(|e| e.to_string()),
        Some("inventory-provider-tests") => match args.next() {
            Some(list) => {
                let root = args
                    .next()
                    .map(PathBuf::from)
                    .unwrap_or_else(workspace_root);
                let registrations = args.next().map(PathBuf::from);
                scenario_inventory::run(&root, Path::new(&list), registrations.as_deref())
            }
            None => Err("inventory-provider-tests requires a nextest JSON list path".into()),
        },
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
  parity-batch <batch.json> <baseline-root>  replay exact original/native pairs and retain evidence
  parity-manifest <init|report> [options]   validate provider inventory and mappings
  parity-queue <manifest.json> <requirements.json> <output.json>  generate remaining family and delivery work
  parity-review <manifest.json> <review.json> <source-root> <evidence-root>  apply explicit reviewed decisions after integrity checks
  parity-targets inventory <root> <output>  enumerate workspace targets independently of test listings
  parity-targets check <root> <package> <listing> <output>  require every package test target to be listed
  inventory-provider-tests <nextest.json> [source-root] [registration-map.json]  reconcile provider source with compiled tests
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
