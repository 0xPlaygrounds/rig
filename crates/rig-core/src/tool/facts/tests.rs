use super::*;
use crate::tool::ToolContext;

/// The facts read back from a published result as they were published, and
/// keep their wire shape, which effect logs store.
#[test]
fn published_facts_read_back_with_a_stable_shape() {
    let changed = FilesChanged(vec![
        FileChange::created("new.rs", 4),
        FileChange::modified("lib.rs", 2, 1)
            .with_hunks(vec![Hunk {
                old: Lines { start: 1, count: 1 },
                new: Lines { start: 1, count: 2 },
                lines: vec![
                    HunkLine::Removed("a\n".to_owned()),
                    HunkLine::Added("b\n".to_owned()),
                    HunkLine::Added("c\n".to_owned()),
                ],
            }])
            .with_before_text("a\n")
            .with_before(FileState {
                bytes: 2,
                lines: Some(1),
                hash: Some(ContentHash("beef".to_owned())),
            })
            .with_after(FileState::unread(4)),
        FileChange::deleted("old.rs", 9),
    ]);
    let lib = changed.0.get(1).and_then(FileChange::after_text);
    assert_eq!(lib.as_deref(), Some("b\nc\n"));
    let read = FilesRead(vec![
        FileRead::new("/work/a.rs", ContentHash("aa".to_owned())),
        FileRead::new("/work/b.rs", ContentHash("bb".to_owned())).with_lines(LinesShown {
            first: 3,
            count: 2,
            total: 10,
        }),
    ]);
    let mut context = ToolContext::new();
    context.insert_result(read.clone()).expect("encodes");
    context.insert_result(changed.clone()).expect("encodes");
    context
        .insert_result(ExitStatus::Exited(2))
        .expect("encodes");
    let cut = OutputCut {
        kept: KeptEnd::Ends,
        kept_lines: 2,
        total_lines: 5,
        spill: Some("/scratch/out.txt".to_owned()),
    };
    context.insert_result(cut.clone()).expect("encodes");
    let published = context.result_context();
    assert_eq!(published.get::<OutputCut>().ok().flatten(), Some(cut));
    assert_eq!(
        published.get::<FilesChanged>().ok().flatten(),
        Some(changed)
    );
    assert_eq!(published.get::<FilesRead>().ok().flatten(), Some(read));
    let status = published.get::<ExitStatus>().ok().flatten();
    assert_eq!(status.and_then(|status| status.code()), Some(2));
    assert!(!status.is_some_and(|status| status.success()));
    assert_eq!(
        serde_json::to_value(&published).ok(),
        Some(serde_json::json!({
            "exit_status": {"exited": 2},
            "output_cut": {"kept": "ends", "kept_lines": 2, "total_lines": 5,
                           "spill": "/scratch/out.txt"},
            "files_read": [
                {"path": "/work/a.rs", "hash": "aa"},
                {"path": "/work/b.rs", "hash": "bb",
                 "lines": {"first": 3, "count": 2, "total": 10}},
            ],
            "files_changed": [
                {"path": "new.rs", "kind": "created", "added": 4, "removed": 0},
                {"path": "lib.rs", "kind": "modified", "added": 2, "removed": 1,
                 "hunks": [{"old": {"start": 1, "count": 1}, "new": {"start": 1, "count": 2},
                            "lines": ["-a\n", "+b\n", "+c\n"]}],
                 "before_text": "a\n", "before": {"bytes": 2, "lines": 1, "hash": "beef"},
                 "after": {"bytes": 4}},
                {"path": "old.rs", "kind": "deleted", "added": 0, "removed": 9},
            ],
        }))
    );
    for ended in [
        ExitStatus::Signaled,
        ExitStatus::TimedOut,
        ExitStatus::Killed,
    ] {
        assert_eq!((ended.code(), ended.success()), (None, false));
    }
}
