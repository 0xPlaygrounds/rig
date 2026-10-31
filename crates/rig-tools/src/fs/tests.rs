use std::path::PathBuf;

use rig_core::tool::{FilesRead, Tool, ToolContext, ToolErrorKind, ToolOutput};

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
        seen: crate::Seen::default(),
        ..Default::default()
    };

    let mut context = ToolContext::new();
    let shown = futures::executor::block_on(read.call(&mut context, ReadArgs::new("a.txt")));
    assert_eq!(
        shown.ok().as_ref().and_then(ToolOutput::as_text),
        Some("one\n")
    );
    let published = context.result::<FilesRead>().ok().flatten();
    assert_eq!(
        published.map(|read| read.paths().map(str::to_owned).collect::<Vec<_>>()),
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
        ..Default::default()
    };
    let args = WriteArgs::new("../new.txt", "x");
    let refused = futures::executor::block_on(write.call(&mut ToolContext::new(), args));
    assert!(refused.is_err());
    assert!(!dir.join("new.txt").exists());
    std::fs::remove_dir_all(dir).ok();
}

/// A change's hunks turn the text before into the text after, whatever
/// the line endings, a missing last newline, or hunks far apart; and its
/// unified diff shows the same lines.
#[test]
fn a_changes_hunks_make_the_text_after() {
    let far: String = (1..=20).map(|line| format!("{line}\n")).collect();
    let far_after = far
        .replacen("2\n", "two\n", 1)
        .replacen("19\n", "nineteen\n", 1);
    let cases = [
        (Some("one\r\ntwo\r\n"), Some("one\r\n2\r\n")),
        (Some("a\rb\rc"), Some("a\rB\rc")),
        (Some("one\ntwo"), Some("one\ntwo\nthree")),
        (Some(far.as_str()), Some(far_after.as_str())),
        (None, Some("new\nfile")),
        (Some("same\n"), Some("same\n")),
    ];
    for (before, after) in cases {
        let change = super::file_change("f", before, after);
        assert_eq!(change.after_text().as_deref(), after, "{before:?}");
        let unified = change.unified();
        let signed = unified.split(['\n', '\r']).filter(|line| {
            line.starts_with(['+', '-']) && !line.starts_with("+++") && !line.starts_with("---")
        });
        assert_eq!(signed.count(), change.added + change.removed, "{unified}");
    }
    let far_change = super::file_change("f", Some(&far), Some(&far_after));
    assert_eq!(far_change.hunks.len(), 2);
    assert_eq!(far_change.changed_lines().collect::<Vec<_>>(), [2, 19]);
}

/// A note on paths that could not be read stays bounded, its first path
/// too, however long the paths and reasons are and however many.
#[test]
fn the_unread_note_is_bounded() {
    let long = "x".repeat(50_000);
    let one = super::unread(
        [(long.clone(), long.clone())].into_iter(),
        "they were not searched",
    );
    assert!(one.len() < 3_000 && one.contains(" […]"), "{}", one.len());
    let many = (0..1000).map(|n| (format!("{n}{}", "y".repeat(400)), "denied".to_owned()));
    let note = super::unread(
        many.collect::<Vec<_>>().into_iter(),
        "they were not searched",
    );
    assert!(
        note.len() < 5_000 && note.ends_with(" more]"),
        "{}",
        note.len()
    );
}
