use rig_core::tool::{ChangeKind, FilesChanged, Tool, ToolContext, content_hash};

use super::{Move, MoveArgs};

/// A move refuses to replace an existing file unless told to; told to, it
/// replaces it, and publishes the old path deleted and the new one changed,
/// with the hash of the moved bytes.
#[test]
fn a_move_replaces_an_existing_file_only_when_told_to() {
    let dir = std::env::temp_dir().join(format!("rig-tools-move-{}", std::process::id()));
    let dir = {
        std::fs::create_dir_all(&dir).ok();
        std::fs::canonicalize(&dir).unwrap_or(dir)
    };
    std::fs::write(dir.join("a.txt"), "new\n").ok();
    std::fs::write(dir.join("b.txt"), "old\n").ok();
    let tool = Move {
        base: Some(dir.clone()),
        ..Move::default()
    };
    let run = |args: MoveArgs| {
        let mut context = ToolContext::new();
        let output = futures::executor::block_on(tool.call(&mut context, args));
        let changed: Option<FilesChanged> = context.result().ok().flatten();
        (output, changed)
    };

    let (refused, changed) = run(MoveArgs::new("a.txt", "b.txt"));
    assert!(refused.is_err_and(|error| error.to_string().contains("overwrite")));
    assert_eq!(changed, None);
    assert_eq!(
        std::fs::read_to_string(dir.join("b.txt")).ok().as_deref(),
        Some("old\n")
    );

    let (moved, changed) = run(MoveArgs::new("a.txt", "b.txt").overwrite());
    assert!(moved.is_ok(), "{moved:?}");
    let changes: Vec<_> = changed
        .map(|changed| changed.0)
        .unwrap_or_default()
        .into_iter()
        .map(|change| {
            let after = change.after.and_then(|it| it.hash);
            (change.kind, change.added, change.removed, after)
        })
        .collect();
    assert_eq!(
        changes,
        [
            (ChangeKind::Deleted, 0, 1, None),
            (ChangeKind::Modified, 1, 1, Some(content_hash(b"new\n")))
        ]
    );
    assert!(!dir.join("a.txt").exists());
    assert_eq!(
        std::fs::read_to_string(dir.join("b.txt")).ok().as_deref(),
        Some("new\n")
    );

    let (created, _) = run(MoveArgs::new("b.txt", "deeper/c.txt"));
    assert!(created.is_ok(), "{created:?}");
    assert!(dir.join("deeper").join("c.txt").is_file());
    std::fs::remove_dir_all(dir).ok();
}
