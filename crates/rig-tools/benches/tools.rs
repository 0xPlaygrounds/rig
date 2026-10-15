//! The coding tools' hot paths, on generated files: a shell round trip, a
//! search that matches nothing and one that matches in every file, the diff
//! of a full rewrite of a Cargo.lock-sized file, and an edit of a 1 MB file.
//!
//! `cargo bench -p rig-tools --bench tools`. The tree `search` walks is
//! written once under Cargo's bench scratch directory (`CARGO_TARGET_TMPDIR`).

use std::fmt::Display;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use rig_tools::edit::apply;
use rig_tools::fs::file_change;
use rig_tools::workspace::Local;
use rig_tools::{Replacement, SearchArgs, Shell, ShellArgs, find};

#[global_allocator]
static ALLOC: divan::AllocProfiler = divan::AllocProfiler::system();

fn main() {
    divan::main();
}

/// `result`'s value, or the process ends saying what failed: a bench whose
/// input cannot be made measures nothing.
fn or_exit<T, E: Display>(result: Result<T, E>, what: &str) -> T {
    result.unwrap_or_else(|error| {
        eprintln!("{what}: {error}");
        std::process::exit(1)
    })
}

/// `sh -c true`: starting a command, waiting for it and reading its output.
#[divan::bench(sample_count = 50)]
fn shell_true_roundtrip() -> String {
    let shell = Shell::default();
    or_exit(
        futures::executor::block_on(shell.run(ShellArgs::new("true"))),
        "sh -c true",
    )
    .output
}

/// Files in the generated tree, and lines in each.
const FILES: usize = 1_000;
const LINES: usize = 200;

/// A source tree of [`FILES`] Rust-looking files of [`LINES`] lines (about
/// 8 MB), each holding `marker_<file>` once, in its middle.
fn tree() -> &'static Path {
    static TREE: OnceLock<PathBuf> = OnceLock::new();
    TREE.get_or_init(|| {
        let root = Path::new(env!("CARGO_TARGET_TMPDIR")).join("rig-tools-bench-tree");
        for file in 0..FILES {
            let dir = root.join(format!("module_{}", file % 20));
            let path = dir.join(format!("file_{file}.rs"));
            if path.exists() {
                continue;
            }
            or_exit(std::fs::create_dir_all(&dir), "the bench tree");
            let text: String = (0..LINES)
                .map(|line| match line == LINES / 2 {
                    true => format!("pub fn marker_{file}() -> usize {{ {file} }}\n"),
                    false => format!(
                        "    let value_{line} = compute(input_{line}, {line}).wrapping_add({file});\n"
                    ),
                })
                .collect();
            or_exit(std::fs::write(&path, text), "a bench tree file");
        }
        root
    })
}

/// `search` over the tree: a pattern no line has (every file read and
/// scanned), and one every file has once (stops at the match limit).
#[divan::bench(args = ["nomatch", "manymatch"], sample_count = 20)]
fn search(bencher: divan::Bencher, case: &str) {
    let pattern = match case {
        "nomatch" => "no_such_identifier_[0-9]+",
        _ => r"pub fn marker_\d+",
    };
    let args = SearchArgs::new(pattern).in_path(tree().to_string_lossy());
    let found = or_exit(find(&Local, &args), "search");
    if (case == "nomatch") != found.hits.is_empty() {
        eprintln!("search {case}: {} hits", found.hits.len());
        std::process::exit(1);
    }
    bencher.bench(|| find(&Local, &args).map(|found| found.hits.len()));
}

/// A Cargo.lock-shaped text of `packages` packages (6 lines each), at
/// version `minor` with checksums that depend on it.
fn lockfile(packages: usize, minor: usize) -> String {
    (0..packages)
        .map(|package| {
            format!(
                "[[package]]\nname = \"crate-{package}\"\nversion = \"1.{minor}.{package}\"\n\
                 source = \"registry+https://github.com/rust-lang/crates.io-index\"\n\
                 checksum = \"{:016x}{:016x}\"\n\n",
                package.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ minor,
                package
                    .wrapping_add(minor)
                    .wrapping_mul(0xff51_afd7_ed55_8ccd),
            )
        })
        .collect()
}

/// The change a `write` of a whole Cargo.lock-sized file (20,000 lines)
/// publishes, after an update that moved every version and checksum.
#[divan::bench(sample_count = 5, sample_size = 1)]
fn file_change_full_rewrite(bencher: divan::Bencher) {
    let before = lockfile(20_000 / 6, 1);
    let after = lockfile(20_000 / 6, 2);
    bencher.bench(|| file_change("Cargo.lock", Some(&before), Some(&after)));
}

/// `edit` of one line in the middle of a 1 MB file, as the tool makes it
/// without the disk: matching, splicing, hashing and the diff.
#[divan::bench(sample_count = 20)]
fn edit_apply_1mb(bencher: divan::Bencher) {
    let lines = 1024 * 1024 / 64;
    let text: String = (0..lines)
        .map(|line| {
            format!("    let value_{line:06} = compute(input, {line:06}).wrapping_add(1);\n")
        })
        .collect();
    let middle = lines / 2;
    let edit = [Replacement::new(
        format!("    let value_{middle:06} = compute(input, {middle:06}).wrapping_add(1);\n"),
        format!("    let value_{middle:06} = compute(input, {middle:06}).wrapping_sub(1);\n"),
    )];
    or_exit(apply("big.rs", &text, &edit), "the 1 MB edit");
    bencher.bench(|| apply("big.rs", &text, &edit));
}
