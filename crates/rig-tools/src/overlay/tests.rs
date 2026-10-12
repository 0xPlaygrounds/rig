use std::sync::Arc;

use rig_core::tool::{ChangeKind, Tool, ToolContext, ToolExecutionError, content_hash};

use super::Overlay;
use crate::workspace::{Local, Workspace};
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
    assert_eq!(shown.ok().as_deref(), Some("     1\tfn two() {}\n"));

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
                change.before,
                change.after,
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
    let files = overlay.files(&dir, Some("*.rs")).unwrap_or_default();
    assert_eq!(files, [dir.join("a.rs"), dir.join("new/b.rs")]);
    std::fs::remove_dir_all(dir).ok();
}
