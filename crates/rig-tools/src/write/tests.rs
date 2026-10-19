use rig_core::tool::{
    ChangeKind, FileChange, FileState, FilesChanged, Tool, ToolContext, content_hash,
};

use super::{Write, WriteArgs};
use crate::{Edit, EditArgs, Replacement};

/// Writing and editing a file publish what they did to it, by its absolute
/// path: created with its lines, then modified with the lines added and
/// removed, each with its hunks and the text it was made to. Each reply
/// says only what the model could not know: that the file was created or
/// replaced, and which lines an edit changed.
#[test]
fn writes_and_edits_publish_the_files_they_changed() {
    let write_tool = Write::default();
    let edit_tool = Edit::default();
    let dir = std::env::temp_dir().join(format!("rig-tools-changes-{}", std::process::id()));
    let path = crate::fs::resolve(Some(&dir), "a.txt")
        .to_string_lossy()
        .into_owned();
    let changed = |context: &ToolContext| {
        context
            .result::<FilesChanged>()
            .ok()
            .flatten()
            .unwrap_or_default()
    };

    let mut context = ToolContext::new();
    let written = Tool::call(
        &write_tool,
        &mut context,
        WriteArgs::new(&path, "one\ntwo\n"),
    );
    let reply = futures::executor::block_on(written).unwrap_or_default();
    assert_eq!(reply, format!("Created {path}."));
    let created = changed(&context).0;
    assert_eq!(
        created.iter().map(FileChange::unified).collect::<Vec<_>>(),
        [format!(
            "--- /dev/null\n+++ {path}\n@@ -0,0 +1,2 @@\n+one\n+two\n"
        )]
    );
    let without_hunks = |change: FileChange| FileChange {
        hunks: Vec::new(),
        ..change
    };
    assert_eq!(
        created.into_iter().map(without_hunks).collect::<Vec<_>>(),
        [FileChange::created(&path, 2).with_after(FileState::of(b"one\ntwo\n"))]
    );

    let mut context = ToolContext::new();
    let rewritten = Tool::call(
        &write_tool,
        &mut context,
        WriteArgs::new(&path, "one\n2\nthree\n"),
    );
    let reply = futures::executor::block_on(rewritten).unwrap_or_default();
    assert_eq!(reply, format!("Replaced {path}, which had 2 lines."));
    let change = changed(&context).0.into_iter().next();
    assert_eq!(
        change.map(|change| (change.kind, change.added, change.removed)),
        Some((ChangeKind::Modified, 2, 1))
    );

    let mut context = ToolContext::new();
    let edits = [
        Replacement::new("one", "1"),
        Replacement::new("three", "3").all(),
    ];
    let edited = Tool::call(&edit_tool, &mut context, EditArgs::new(&path, edits));
    let reply = futures::executor::block_on(edited).unwrap_or_default();
    let said = reply.lines().next();
    assert_eq!(said, Some(format!("Edited {path}: lines 1, 3.").as_str()));
    let change = changed(&context).0.into_iter().next();
    assert_eq!(
        change
            .as_ref()
            .map(|change| (change.path.as_str(), change.added, change.removed)),
        Some((path.as_str(), 2, 2))
    );
    assert_eq!(
        change
            .as_ref()
            .and_then(|change| change.after.clone()?.hash),
        Some(content_hash(b"1\n2\n3\n"))
    );
    assert_eq!(
        change.as_ref().map(FileChange::unified),
        Some(format!(
            "--- {path}\n+++ {path}\n@@ -1,3 +1,3 @@\n-one\n+1\n 2\n-three\n+3\n"
        ))
    );
    assert_eq!(
        change
            .as_ref()
            .and_then(|change| change.before_text.as_deref()),
        Some("one\n2\nthree\n")
    );
    assert_eq!(
        change.and_then(|change| change.after_text()).as_deref(),
        Some("1\n2\n3\n")
    );
    assert_eq!(
        std::fs::read_to_string(&path).ok().as_deref(),
        Some("1\n2\n3\n")
    );
    std::fs::remove_dir_all(dir).ok();
}
