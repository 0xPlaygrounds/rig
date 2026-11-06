use std::io::ErrorKind;
use std::sync::Arc;

use rig_core::tool::{ChangeKind, Tool, ToolContext, ToolExecutionError, ToolOutput, content_hash};

use super::Overlay;
use crate::process::Launcher;
use crate::remote::Remote;
use crate::workspace::{Listing, Local, Stat, Workspace};
use crate::{Edit, EditArgs, Read, ReadArgs, Replacement, Write, WriteArgs};

/// What `tool` answers to `args`.
fn run<T: Tool<Output = String, Error = ToolExecutionError>>(
    tool: &T,
    args: T::Args,
) -> Result<String, ToolExecutionError> {
    futures::executor::block_on(tool.call(&mut ToolContext::new(), args))
}

/// File tools run on an overlay leave the disk as it was: the next tool
/// reads what the last one wrote, and the overlay lists each change with
/// the content before and after.
#[test]
fn tools_on_an_overlay_leave_the_disk_and_list_their_changes() {
    let dir = std::env::temp_dir().join(format!("rig-tools-overlay-{}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
    std::fs::create_dir_all(&dir).ok();
    let dir = std::fs::canonicalize(&dir).unwrap_or(dir);
    std::fs::write(dir.join("a.rs"), "fn one() {}\n").ok();
    let overlay = Arc::new(Overlay::new(Arc::new(Local)));
    let workspace: Arc<dyn Workspace> = overlay.clone();
    let base = Some(dir.clone());
    let edit = Edit {
        workspace: workspace.clone(),
        base: base.clone(),
        confine: true,
        seen: crate::Seen::default(),
    };
    let write = Write {
        workspace: workspace.clone(),
        base: base.clone(),
        confine: true,
        seen: Default::default(),
    };
    let read = Read {
        workspace,
        base,
        confine: true,
        seen: crate::Seen::default(),
    };
    let edited = run(
        &edit,
        EditArgs::new("a.rs", [Replacement::new("one", "two")]),
    );
    assert!(edited.is_ok(), "{edited:?}");
    let written = run(&write, WriteArgs::new("new/b.rs", "b\n"));
    assert!(written.is_ok(), "{written:?}");
    let shown =
        futures::executor::block_on(read.call(&mut ToolContext::new(), ReadArgs::new("a.rs")));
    assert_eq!(
        shown.ok().as_ref().and_then(ToolOutput::as_text),
        Some("fn two() {}\n")
    );

    assert_eq!(
        std::fs::read_to_string(dir.join("a.rs")).ok().as_deref(),
        Some("fn one() {}\n")
    );
    assert!(!dir.join("new").exists());
    let changes: Vec<_> = overlay
        .changes()
        .into_iter()
        .map(|change| {
            (
                change.kind,
                change.added,
                change.removed,
                change.before.and_then(|it| it.hash),
                change.after.and_then(|it| it.hash),
            )
        })
        .collect();
    assert_eq!(
        changes,
        [
            (
                ChangeKind::Modified,
                1,
                1,
                Some(content_hash(b"fn one() {}\n")),
                Some(content_hash(b"fn two() {}\n"))
            ),
            (ChangeKind::Created, 1, 0, None, Some(content_hash(b"b\n"))),
        ]
    );
    let files = overlay
        .files(&dir, Some("*.rs"), Listing::Visible)
        .map(|listed| listed.files)
        .unwrap_or_default();
    assert_eq!(files, [dir.join("a.rs"), dir.join("new/b.rs")]);
    std::fs::remove_dir_all(dir).ok();
}

/// A previewed change says how large each side of the file is, and a
/// deletion carries its hunks and text, so a guard need not read the file
/// again.
#[test]
fn an_overlay_change_gives_each_side_and_a_deletions_text() {
    let dir = std::env::temp_dir().join(format!("rig-tools-overlay-states-{}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
    std::fs::create_dir_all(&dir).ok();
    let dir = std::fs::canonicalize(&dir).unwrap_or(dir);
    std::fs::write(dir.join("old.rs"), "a\nb\nc\n").ok();
    let overlay = Overlay::new(Arc::new(Local));
    assert!(overlay.remove(&dir.join("old.rs")).is_ok());
    assert!(overlay.write(&dir.join("new.rs"), b"0123456789").is_ok());
    let changes = overlay.changes();
    let kinds: Vec<_> = changes.iter().map(|change| change.kind).collect();
    assert_eq!(kinds, [ChangeKind::Deleted, ChangeKind::Created]);
    let deleted = changes.first();
    let lines = deleted.and_then(|it| it.before.as_ref()?.lines);
    assert_eq!(lines, Some(3));
    let hunks = deleted.map(|it| it.hunks.as_slice()).unwrap_or_default();
    let removed = hunks.iter().flat_map(|hunk| &hunk.lines);
    assert_eq!(removed.filter(|line| line.sign() == '-').count(), 3);
    let text = deleted.and_then(|it| it.before_text.as_deref());
    assert_eq!(text, Some("a\nb\nc\n"));
    let created = changes.get(1);
    assert_eq!(
        created.and_then(|it| Some(it.after.as_ref()?.bytes)),
        Some(10)
    );
    assert!(dir.join("old.rs").exists());
    std::fs::remove_dir_all(dir).ok();
}

/// An overlay says a path is on a host as the workspace under it does.
#[test]
fn an_overlay_tells_where_a_path_is_as_its_workspace_does() {
    let bench = crate::remote::Remote::new("bench", ["true"], crate::process::Launcher::default);
    let hosts = crate::remote::Hosts::new(Arc::new(Local), bench);
    let overlay = Overlay::new(Arc::new(hosts));
    let there = overlay.host_for(std::path::Path::new("bench:~/x"));
    let here = overlay.host_for(std::path::Path::new("/project/x"));
    assert_eq!((there, here), (Some("bench"), None));
}

/// A file is no directory, whether it is held here or under the overlay,
/// locally or on a remote whose `stat` calls such a path missing: a write
/// or a rename beneath it is refused and nothing changes. Once it is
/// removed the path is a directory to make, which a write cannot replace.
#[test]
fn an_overlay_refuses_a_path_beneath_a_file_it_holds() {
    let remote = ["sh", "-c", "exec sh -c \"$1\"", "transport"];
    let remote = Remote::new("bench", remote, Launcher::default);
    assert!(remote.is_ok());
    let Ok(remote) = remote else { return };
    let under: [(&str, Arc<dyn Workspace>); 2] =
        [("local", Arc::new(Local)), ("remote", Arc::new(remote))];
    for (name, inner) in under {
        let dir =
            std::env::temp_dir().join(format!("rig-overlay-parent-{name}-{}", std::process::id()));
        std::fs::remove_dir_all(&dir).ok();
        std::fs::create_dir_all(&dir).ok();
        let (held, disk, source) = (dir.join("held"), dir.join("disk"), dir.join("source"));
        std::fs::write(&disk, "on disk").ok();
        let overlay = Overlay::new(inner);
        assert!(overlay.write(&held, b"a file").is_ok());
        assert!(overlay.write(&source, b"source").is_ok());
        for child in [
            held.join("child"),
            held.join("deep/child"),
            disk.join("child"),
            disk.join("deep/child"),
        ] {
            let refused = Some(ErrorKind::NotADirectory);
            let written = overlay.write(&child, b"child");
            assert_eq!(written.err().map(|it| it.kind()), refused, "write beneath");
            let renamed = overlay.rename(&source, &child);
            assert_eq!(renamed.err().map(|it| it.kind()), refused, "rename beneath");
            let looked = overlay.stat(&child);
            assert_eq!(looked.err().map(|it| it.kind()), refused, "stat beneath");
        }
        assert_eq!(overlay.changes().len(), 2, "{name}");

        for parent in [&held, &disk] {
            let child = parent.join("child");
            assert!(overlay.remove(parent).is_ok());
            assert!(matches!(overlay.stat(&child), Ok(Stat::Missing)), "{name}");
            let written = overlay.write(&child, b"child");
            assert!(written.is_ok(), "{name}: {written:?}");
            assert!(matches!(overlay.stat(parent), Ok(Stat::Dir)), "{name}");
            assert!(overlay.write(parent, b"a file again").is_err());
        }
        std::fs::remove_dir_all(dir).ok();
    }
}

/// A rename to the same path leaves a file as it is, on disk or held here,
/// and of a missing file is still refused.
#[test]
fn renaming_a_file_to_itself_changes_nothing() {
    let dir = std::env::temp_dir().join(format!("rig-overlay-self-{}", std::process::id()));
    std::fs::create_dir_all(&dir).ok();
    let (existing, created) = (dir.join("existing"), dir.join("created"));
    std::fs::write(&existing, b"on disk").ok();
    let overlay = Overlay::new(Arc::new(Local));

    assert!(overlay.rename(&existing, &existing).is_ok());
    let read = overlay.read(&existing, 100).ok().map(|it| it.bytes);
    assert_eq!(read.as_deref(), Some(&b"on disk"[..]), "the file is gone");
    assert!(overlay.changes().is_empty());

    assert!(overlay.write(&created, b"held").is_ok());
    assert!(overlay.rename(&created, &created).is_ok());
    let read = overlay.read(&created, 100).ok().map(|it| it.bytes);
    assert_eq!(read.as_deref(), Some(&b"held"[..]));

    let missing = dir.join("missing");
    let renamed = overlay.rename(&missing, &missing);
    assert_eq!(renamed.err().map(|it| it.kind()), Some(ErrorKind::NotFound));

    std::fs::remove_dir_all(dir).ok();
}
