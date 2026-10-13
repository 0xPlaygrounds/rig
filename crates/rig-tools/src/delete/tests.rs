use rig_core::tool::{FileChange, FileState, FilesChanged, Tool, ToolContext, ToolErrorKind};

use super::{Delete, DeleteArgs};

/// A confined delete refuses a path outside its base and leaves the file;
/// inside it, it removes the file and publishes it deleted, with its lines.
/// A directory is refused.
#[test]
fn a_delete_stays_in_its_base_and_publishes_the_deleted_file() {
    let dir = std::env::temp_dir().join(format!("rig-tools-delete-{}", std::process::id()));
    let dir = {
        std::fs::create_dir_all(dir.join("base").join("sub")).ok();
        std::fs::canonicalize(&dir).unwrap_or(dir)
    };
    let base = dir.join("base");
    std::fs::write(dir.join("outside.txt"), "keep\n").ok();
    std::fs::write(base.join("a.txt"), "one\ntwo\n").ok();
    let delete = Delete {
        base: Some(base.clone()),
        confine: true,
        ..Delete::default()
    };
    let run = |path: &str| {
        let mut context = ToolContext::new();
        let output = futures::executor::block_on(delete.call(&mut context, DeleteArgs::new(path)));
        let changed: Option<FilesChanged> = context.result().ok().flatten();
        (output, changed)
    };

    let (refused, changed) = run("../outside.txt");
    assert_eq!(
        refused.err().map(|error| error.kind()),
        Some(ToolErrorKind::PermissionDenied)
    );
    assert_eq!(changed, None);
    assert!(dir.join("outside.txt").exists());

    let (deleted, changed) = run("a.txt");
    assert_eq!(deleted.ok().as_deref(), Some("Deleted a.txt."));
    let path = base.join("a.txt").to_string_lossy().into_owned();
    assert_eq!(
        changed.map(|changed| changed.0),
        Some(vec![
            FileChange::deleted(&path, 2)
                .with_diff(format!(
                    "--- {path}\n+++ /dev/null\n@@ -1,2 +0,0 @@\n-one\n-two\n"
                ))
                .with_before(FileState::of(b"one\ntwo\n"))
        ])
    );
    assert!(!base.join("a.txt").exists());
    assert!(run("sub").0.is_err());
    assert!(base.join("sub").is_dir());
    std::fs::remove_dir_all(dir).ok();
}
