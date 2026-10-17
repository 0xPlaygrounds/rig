use std::sync::Arc;

use rig_core::tool::{ChangeKind, Tool, ToolContext, ToolExecutionError, content_hash};

use super::Overlay;
use crate::workspace::{Listing, Local, Workspace};
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
    };
    let write = Write {
        workspace: workspace.clone(),
        base: base.clone(),
        confine: true,
    };
    let read = Read {
        workspace,
        base,
        confine: true,
    };
    let edited = run(
        &edit,
        EditArgs::new("a.rs", [Replacement::new("one", "two")]),
    );
    assert!(edited.is_ok(), "{edited:?}");
    let written = run(&write, WriteArgs::new("new/b.rs", "b\n"));
    assert!(written.is_ok(), "{written:?}");
    let shown = run(&read, ReadArgs::new("a.rs"));
    assert_eq!(shown.ok().as_deref(), Some("fn two() {}\n"));

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
