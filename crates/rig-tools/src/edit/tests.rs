use rig_core::tool::{FilesChanged, HunkLine, Tool, ToolContext};

use super::{Edit, EditArgs, Replacement, apply, batch_failure};

/// A misspelled key inside an edit, such as `replaceAll`, is refused
/// rather than ignored, which would replace one match instead of all.
#[test]
fn a_misspelled_key_inside_an_edit_is_refused() {
    let edit = serde_json::json!({"old_text": "a", "new_text": "b", "replaceAll": true});
    let args = serde_json::json!({"path": "a.rs", "edits": [edit]});
    let parsed = serde_json::from_value::<EditArgs>(args).map(|_| ());
    assert!(parsed.is_err_and(|error| error.to_string().contains("replaceAll")));
}

/// A failed edit is sent again; in a batch, every failed edit is named and
/// the whole batch is sent again.
#[test]
fn a_failure_names_every_failed_edit() {
    let single = batch_failure("a.rs", 1, &["`old_text` was not found".to_owned()]);
    assert_eq!(
        single,
        "`old_text` was not found. `a.rs` was not changed; fix the edit and send it again."
    );
    let one = batch_failure(
        "a.rs",
        3,
        &["edits[1]: `old_text` was not found".to_owned()],
    );
    assert_eq!(
        one,
        "edits[1]: `old_text` was not found. No edit was applied, so `a.rs` was not changed. \
         Send all 3 edits again, with this one fixed."
    );
    let failed = ["edits[0]: one".to_owned(), "edits[2]: two".to_owned()];
    let two = batch_failure("a.rs", 3, &failed);
    assert_eq!(
        two,
        "2 of the 3 edits failed:\nedits[0]: one.\nedits[2]: two.\nNo edit was applied, so \
         `a.rs` was not changed. Send all 3 edits again, with these fixed."
    );
}

/// `apply` makes of a file's text what the tool writes to it: here a CRLF
/// file with a byte-order mark, edited by a loose match.
#[test]
fn apply_gives_the_text_the_tool_writes() {
    let edit_tool = Edit::default();
    let path = std::env::temp_dir().join(format!("rig-tools-apply-{}.txt", std::process::id()));
    let before = "\u{feff}let a = \u{201c}x\u{201d};  \r\nlet b = 2;\r\n";
    std::fs::write(&path, before).ok();
    let path = path.to_string_lossy().into_owned();
    let edits = [Replacement::new("let a = \"x\";\n", "let a = 1;\n")];

    let applied = apply(&path, before, &edits);
    let mut context = ToolContext::new();
    let edited = Tool::call(&edit_tool, &mut context, EditArgs::new(&path, edits));
    let output = futures::executor::block_on(edited).unwrap_or_default();
    let written = std::fs::read_to_string(&path).ok();
    std::fs::remove_file(&path).ok();

    let applied = applied.ok();
    assert_eq!(
        applied.as_ref().map(|applied| applied.text.as_str()),
        Some("\u{feff}let a = 1;\r\nlet b = 2;\r\n")
    );
    assert_eq!(
        applied.as_ref().map(|applied| applied.text.clone()),
        written
    );
    assert_eq!(applied.as_ref().map(|applied| applied.notes.len()), Some(1));
    assert!(
        output.contains("Note: `old_text` matched only after"),
        "{output}"
    );
}

/// An edit answers with the lines that now hold the changed text and how
/// far the file grew or shrank, not with the change itself, which it still
/// publishes whole for the terminal view.
#[test]
fn an_edit_answers_where_it_changed_the_file() {
    let edit_tool = Edit::default();
    let path = std::env::temp_dir().join(format!("rig-tools-where-{}.txt", std::process::id()));
    std::fs::write(&path, "a\nb\nc\nd\ne\nf\ng\nh\ni\nj\n").ok();
    let path = path.to_string_lossy().into_owned();
    let call = |edits: Vec<Replacement>| {
        let mut context = ToolContext::new();
        let edited = Tool::call(&edit_tool, &mut context, EditArgs::new(&path, edits));
        let reply = futures::executor::block_on(edited).unwrap_or_default();
        (reply, context.result::<FilesChanged>().ok().flatten())
    };

    // b becomes three lines; h goes.
    let (reply, published) = call(vec![
        Replacement::new("b\n", "b1\nb2\nb3\n"),
        Replacement::new("h\n", ""),
    ]);
    assert_eq!(
        reply,
        format!("Edited {path}: lines 2-4, line 10 (+1 line).")
    );
    assert!(!reply.contains("@@"), "{reply}");
    let hunks = published
        .map(|FilesChanged(changes)| changes)
        .unwrap_or_default();
    let lines = hunks
        .iter()
        .flat_map(|change| &change.hunks)
        .flat_map(|hunk| &hunk.lines);
    assert!(
        lines
            .clone()
            .any(|line| matches!(line, HunkLine::Same(text) if text == "a\n")),
        "the published change keeps its context: {hunks:?}"
    );
    assert!(
        lines
            .clone()
            .any(|line| matches!(line, HunkLine::Removed(text) if text == "h\n"))
    );

    let (reply, _) = call(vec![Replacement::new("j", "J").all()]);
    assert_eq!(reply, format!("Edited {path}: line 11."));
    std::fs::write(&path, "x\ny\nx\ny\nx\n").ok();
    let (reply, _) = call(vec![Replacement::new("x", "z").all()]);
    assert_eq!(
        reply,
        format!("Edited {path}: 3 replacements, lines 1, 3, 5.")
    );
    // The first place only removed a line, and the file shrank.
    std::fs::write(&path, "a\nb\nc\nd\ne\nf\n").ok();
    let (reply, _) = call(vec![
        Replacement::new("b\n", ""),
        Replacement::new("e\n", "E\n"),
    ]);
    std::fs::remove_file(&path).ok();
    assert_eq!(reply, format!("Edited {path}: line 2, 4 (-1 line)."));
}
