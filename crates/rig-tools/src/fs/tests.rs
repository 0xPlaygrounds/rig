use std::path::PathBuf;

use rig_core::tool::{FilesChanged, FilesRead, Tool, ToolContext, ToolErrorKind, ToolOutput};

use super::{Prefix, resolve};
use crate::workspace::Contents;
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
        ["it was not searched", "they were not searched"],
    );
    assert!(one.len() < 3_000 && one.contains(" […]"), "{}", one.len());
    let many = (0..1000).map(|n| (format!("{n}{}", "y".repeat(400)), "denied".to_owned()));
    let note = super::unread(
        many.collect::<Vec<_>>().into_iter(),
        ["it was not searched", "they were not searched"],
    );
    assert!(
        note.len() < 5_000 && note.ends_with(" more]"),
        "{}",
        note.len()
    );
}

/// A prefix is cut exactly when bytes of the file were left unread, however
/// decoding changed the text's length.
#[test]
fn a_prefix_is_cut_only_when_bytes_were_left_unread() {
    let cases: [(&[u8], u64, bool); 4] = [
        (b"hello", 5, false),
        (b"hell", 5, true),
        (b"\xef\xbb\xbfhello", 8, false),
        (b"\xffabc", 5, true),
    ];
    for (bytes, size, cut) in cases {
        let contents = Contents {
            bytes: bytes.to_vec(),
            size,
        };
        let prefix = Prefix::of(contents, bytes.len());
        assert_eq!(prefix.cut(), cut, "{:?} of {size} bytes", prefix.text);
    }
}

/// A confined write refuses a dangling link in its base that leads out,
/// however its target is spelled, and changes nothing outside; one that
/// stays in the base creates its target and keeps the link.
#[cfg(unix)]
#[test]
fn a_confined_write_never_follows_a_link_out_of_its_base() {
    use std::os::unix::fs::symlink;
    let dir = scratch("dangling");
    let base = dir.join("real");
    std::fs::create_dir(dir.join("out")).ok();
    std::fs::write(dir.join("out/victim.txt"), "kept\n").ok();
    symlink("../out/new.txt", base.join("relative")).ok();
    symlink(dir.join("out/new.txt"), base.join("absolute")).ok();
    symlink("../out/missing/../victim.txt", base.join("back")).ok();
    symlink("relative", base.join("chain")).ok();
    symlink(dir.join("out/missing"), base.join("folder")).ok();
    symlink("inside/new.txt", base.join("stays")).ok();
    let write = Write {
        base: Some(base.clone()),
        confine: true,
        ..Default::default()
    };
    let call = |path: &str, context: &mut ToolContext| {
        let args = WriteArgs::new(path, "escaped\n");
        futures::executor::block_on(write.call(context, args))
    };

    for path in ["relative", "absolute", "back", "chain", "folder/new.txt"] {
        let mut context = ToolContext::new();
        assert_eq!(
            call(path, &mut context).err().map(|error| error.kind()),
            Some(ToolErrorKind::PermissionDenied),
            "{path}"
        );
        assert!(context.result::<FilesChanged>().ok().flatten().is_none());
    }
    assert!(!dir.join("out/new.txt").exists());
    assert!(!dir.join("out/missing").exists());
    assert_eq!(
        std::fs::read_to_string(dir.join("out/victim.txt")).ok(),
        Some("kept\n".to_owned())
    );
    assert!(call("stays", &mut ToolContext::new()).is_ok());
    assert_eq!(
        std::fs::read_to_string(base.join("inside/new.txt")).ok(),
        Some("escaped\n".to_owned())
    );
    assert!(std::fs::symlink_metadata(base.join("stays")).is_ok_and(|meta| meta.is_symlink()));
    std::fs::remove_dir_all(dir).ok();
}

/// A dangling link resolves to the file a write creates, so the path a
/// tool checks and reports is the one it changes.
#[cfg(unix)]
#[test]
fn a_dangling_link_resolves_to_what_a_write_creates() {
    let dir = scratch("dangling-resolve");
    std::os::unix::fs::symlink("real/new/../made.txt", dir.join("link")).ok();
    let resolved = resolve(Some(&dir), "link");
    assert_eq!(resolved, dir.join("real/made.txt"));
    assert!(super::write_atomic(&dir.join("link"), b"x").is_ok());
    assert!(resolved.exists());
    std::fs::remove_dir_all(dir).ok();
}

/// A private file's temporary copy is never readable by others, watched
/// across many writes, and a new file gets the default mode, not a private
/// one.
#[cfg(unix)]
#[test]
fn a_private_file_is_never_staged_readable() {
    use std::os::unix::fs::PermissionsExt;
    use std::sync::atomic::{AtomicBool, Ordering};

    let mode = |path: &std::path::Path| {
        std::fs::metadata(path).map(|meta| meta.permissions().mode() & 0o777)
    };
    let dir = scratch("staged-mode").join("real");
    let private = dir.join("private");
    assert!(std::fs::write(&private, "x").is_ok());
    assert!(std::fs::set_permissions(&private, std::fs::Permissions::from_mode(0o600)).is_ok());

    let done = AtomicBool::new(false);
    let (open, writes) = std::thread::scope(|scope| {
        let watcher = scope.spawn(|| {
            while !done.load(Ordering::Relaxed) {
                for entry in std::fs::read_dir(&dir).into_iter().flatten().flatten() {
                    if !entry.file_name().to_string_lossy().ends_with(".tmp") {
                        continue;
                    }
                    let seen = entry
                        .metadata()
                        .map(|meta| meta.permissions().mode() & 0o777);
                    if let Some(open) = seen.ok().filter(|seen| seen & 0o077 != 0) {
                        return Some(open);
                    }
                }
            }
            None
        });
        let writes: Vec<_> = (0..300)
            .take_while(|_| !watcher.is_finished())
            .map(|_| super::write_atomic(&private, b"y").is_ok())
            .collect();
        done.store(true, Ordering::Relaxed);
        (watcher.join().ok().flatten(), writes)
    });
    assert!(writes.iter().all(|wrote| *wrote), "a write failed");
    assert_eq!(open, None, "staged readable by others");
    assert_eq!(mode(&private).ok(), Some(0o600));

    let default = dir.join("default");
    let new = dir.join("new");
    assert!(std::fs::write(&default, "x").is_ok() && super::write_atomic(&new, b"x").is_ok());
    assert_eq!(mode(&new).ok(), mode(&default).ok());
    std::fs::remove_dir_all(dir.parent().unwrap_or(&dir)).ok();
}
