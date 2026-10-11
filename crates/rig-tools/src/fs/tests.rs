use std::path::PathBuf;

use rig_core::tool::{FilesRead, Tool, ToolContext, ToolErrorKind};

use super::resolve;
use crate::{Read, ReadArgs, Write, WriteArgs};

/// A scratch directory of its own for `name`, resolved, holding `real/`.
fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("rig-tools-{name}-{}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
    std::fs::create_dir_all(dir.join("real")).ok();
    std::fs::canonicalize(&dir).unwrap_or(dir)
}

/// A path resolves against the base, `..` folded, and through a symlinked
/// directory to where the file really is, whether it exists or not.
#[cfg(unix)]
#[test]
fn a_path_resolves_against_the_base_through_links() {
    let dir = scratch("resolve");
    std::os::unix::fs::symlink(dir.join("real"), dir.join("link")).ok();
    std::fs::write(dir.join("real").join("a.rs"), "").ok();

    assert_eq!(resolve(Some(&dir), "real/a.rs"), dir.join("real/a.rs"));
    assert_eq!(
        resolve(Some(&dir), "link/../real/a.rs"),
        dir.join("real/a.rs")
    );
    assert_eq!(resolve(Some(&dir), "link/a.rs"), dir.join("real/a.rs"));
    assert_eq!(
        resolve(Some(&dir), "link/new/../b.rs"),
        dir.join("real/b.rs")
    );
    assert_eq!(resolve(Some(&dir), "/"), PathBuf::from("/"));
    std::fs::remove_dir_all(dir).ok();
}

/// A tool with a base works in it; confined, it refuses a path outside the
/// base before touching it, however it is spelled.
#[test]
fn a_confined_tool_stays_in_its_base() {
    let dir = scratch("confine");
    let base = dir.join("real");
    std::fs::write(base.join("a.txt"), "one\n").ok();
    std::fs::write(dir.join("outside.txt"), "secret\n").ok();
    let read = Read {
        base: Some(base.clone()),
        confine: true,
    };

    let mut context = ToolContext::new();
    let shown = futures::executor::block_on(read.call(&mut context, ReadArgs::new("a.txt")));
    assert_eq!(shown.ok().as_deref(), Some("     1\tone\n"));
    let published = context.result::<FilesRead>().ok().flatten();
    assert_eq!(
        published.map(|read| read.0),
        Some(vec![base.join("a.txt").to_string_lossy().into_owned()])
    );

    let outside = dir.join("outside.txt").to_string_lossy().into_owned();
    for path in ["../outside.txt", outside.as_str()] {
        let refused =
            futures::executor::block_on(read.call(&mut ToolContext::new(), ReadArgs::new(path)));
        assert_eq!(
            refused.err().map(|error| error.kind()),
            Some(ToolErrorKind::PermissionDenied),
            "{path}"
        );
    }
    let write = Write {
        base: Some(base),
        confine: true,
    };
    let args = WriteArgs::new("../new.txt", "x");
    let refused = futures::executor::block_on(write.call(&mut ToolContext::new(), args));
    assert!(refused.is_err());
    assert!(!dir.join("new.txt").exists());
    std::fs::remove_dir_all(dir).ok();
}
