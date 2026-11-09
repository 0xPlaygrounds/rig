//! Sets `RIG_LONG_VERSION`: the package version, and the short commit when
//! the launcher is built in a rig checkout, as `0.46.0 (b1321662b)`. A
//! crates.io package has no checkout, so it gets the version alone.

use std::path::PathBuf;
use std::process::Command;

fn main() {
    let version = std::env::var("CARGO_PKG_VERSION").unwrap_or_default();
    let manifest = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap_or_default());
    // `crates/rig-cli` is two folders below the checkout's root.
    let checkout = manifest.join("../..");
    let git = |args: &[&str]| {
        let output = Command::new("git")
            .arg("-C")
            .arg(&checkout)
            .args(args)
            .output()
            .ok()?;
        let text = String::from_utf8(output.stdout).ok()?;
        output.status.success().then(|| text.trim().to_owned())
    };
    // Only a repository rooted at the checkout counts, not one the package
    // was unpacked inside, such as a home directory under git.
    let root = git(&["rev-parse", "--show-toplevel"]).map(PathBuf::from);
    let own = root.is_some_and(|root| root.canonicalize().ok() == checkout.canonicalize().ok());
    let commit = own
        .then(|| git(&["rev-parse", "--short=9", "HEAD"]))
        .flatten();
    println!("cargo::rerun-if-changed=build.rs");
    if commit.is_some() {
        // HEAD names the branch, or the commit when detached. A commit on a
        // branch rewrites its loose ref file, or creates one under
        // `refs/heads` when the branch was packed; a reftable repository
        // writes new tables under `reftable` instead. `--git-path` finds each
        // for a worktree too. Only paths that exist are watched: a missing
        // one would rerun this script on every build.
        let path = |file: &str| {
            let path = git(&["rev-parse", "--git-path", file]).map(|path| checkout.join(path));
            path.filter(|path| path.exists())
        };
        let branch = git(&["rev-parse", "--symbolic-full-name", "HEAD"])
            .filter(|branch| branch.starts_with("refs/"))
            .and_then(|branch| path(&branch))
            .filter(|branch| branch.is_file());
        let watched = [
            path("HEAD"),
            branch.or_else(|| path("refs/heads")),
            path("reftable"),
        ];
        for path in watched.into_iter().flatten() {
            println!("cargo::rerun-if-changed={}", path.display());
        }
    }
    let long = match commit {
        Some(commit) => format!("{version} ({commit})"),
        None => version,
    };
    println!("cargo::rustc-env=RIG_LONG_VERSION={long}");
}
