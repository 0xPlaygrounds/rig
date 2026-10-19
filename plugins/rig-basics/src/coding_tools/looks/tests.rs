use rig_harness::rig_core::message::{ToolCall, ToolFunction, ToolName};
use rig_harness::rig_core::serde_json::{Value, json};
use rig_harness::rig_core::tool::{
    ContentHash, FileChange, FileRead, FilesChanged, FilesRead, Hunk, HunkLine, Lines, LinesShown,
    ToolContext, ToolResult, ToolResultContext,
};
use rig_tui::ratatui::text::Line;
use rig_tui::{CallState, ToolCallView};

use super::{edit, read, shell, write};

/// What `look` draws for a `tool` call with `args` that answered `output`.
fn drawn(
    look: fn(&ToolCallView<'_>) -> Vec<Line<'static>>,
    tool: &str,
    args: Value,
    output: &str,
) -> String {
    drawn_with(look, tool, args, output, None)
}

/// [`drawn`], for a call that published `facts`.
fn drawn_with(
    look: fn(&ToolCallView<'_>) -> Vec<Line<'static>>,
    tool: &str,
    args: Value,
    output: &str,
    facts: Option<&ToolResultContext>,
) -> String {
    let Ok(name) = ToolName::new(tool) else {
        return String::new();
    };
    let call = ToolCall::from_wire("c1", ToolFunction::new(name, args));
    let result = call.answer(&ToolResult::success(output.to_owned().into()));
    let mut view = ToolCallView::new(&call, Some(&result), CallState::Done);
    view.facts = facts;
    let lines: Vec<String> = look(&view).iter().map(ToString::to_string).collect();
    lines.join("\n")
}

#[test]
fn one_line_is_counted_as_one_line() {
    let wrote = drawn(
        write,
        "write",
        json!({ "path": "a", "content": "hi\n" }),
        "ok",
    );
    assert!(wrote.contains("write a 1 line\n"), "{wrote}");
    let read_two = drawn(read, "read", json!({ "path": "a", "limit": 2 }), "hi\nho");
    assert!(read_two.contains("first 2 lines"), "{read_two}");
}

#[test]
fn a_read_says_which_lines_it_showed_when_it_published_them() {
    let shown = |first, count| {
        let lines = LinesShown {
            first,
            count,
            total: 7,
        };
        let file = FileRead::new("a", ContentHash("aa".to_owned())).with_lines(lines);
        let mut context = ToolContext::default();
        context.insert_result(FilesRead(vec![file])).ok();
        let facts = context.result_context();
        let output = "three\nfour\n[3 more lines; read on with offset 5]\n";
        drawn_with(read, "read", json!({ "path": "a" }), output, Some(&facts))
    };
    let part = shown(3, 2);
    assert!(part.contains("read a lines 3–4 of 7"), "{part}");
    assert!(part.contains("     3  three\n"), "{part}");
    assert!(part.contains("     4  four\n"), "{part}");
    assert!(
        part.ends_with("   [3 more lines; read on with offset 5]"),
        "{part}"
    );
    let whole = shown(1, 7);
    assert!(whole.contains("read a 7 lines"), "{whole}");
}

/// Every call draws all of its arguments and output: nothing is cut.
#[test]
fn a_call_draws_every_line() {
    let rows: Vec<String> = (1..=40).map(|row| format!("row-{row}")).collect();
    let output = rows.join("\n");
    let ran = drawn(shell, "shell", json!({ "command": "seq 1 40" }), &output);
    assert!(ran.contains("row-1\n") && ran.ends_with("row-40"), "{ran}");
    let read_all = drawn(read, "read", json!({ "path": "a" }), &output);
    assert!(
        read_all.contains("row-1\n") && read_all.ends_with("row-40"),
        "{read_all}"
    );
    let content = json!({ "path": "a", "content": output });
    let wrote = drawn(write, "write", content, "ok");
    assert!(
        wrote.contains("row-1\n") && wrote.ends_with("row-40"),
        "{wrote}"
    );
}

#[test]
fn a_read_file_is_drawn_in_its_languages_colors() {
    let Ok(name) = ToolName::new("read") else {
        return;
    };
    let call = ToolCall::from_wire("c1", ToolFunction::new(name, json!({ "path": "a.rs" })));
    let output = "fn main() {}\n";
    let result = call.answer(&ToolResult::success(output.to_owned().into()));
    let view = ToolCallView::new(&call, Some(&result), CallState::Done);
    let lines = read(&view);
    let code = lines
        .get(1)
        .map(|line| line.spans.clone())
        .unwrap_or_default();
    assert!(
        code.iter().skip(1).any(|span| span.style.fg.is_some()),
        "{code:?}"
    );
    assert_eq!(
        lines.get(1).map(ToString::to_string).as_deref(),
        Some("   fn main() {}")
    );
}

/// The terminal draws an edit's reply without the file's lines it shows
/// the model, then the whole change it published, the lines around it too.
#[test]
fn an_edit_is_drawn_as_the_diff_it_published() {
    let change = FileChange::modified("/p/a.rs", 1, 1).with_hunks(vec![Hunk {
        old: Lines { start: 1, count: 2 },
        new: Lines { start: 1, count: 2 },
        lines: vec![
            HunkLine::Same("kept_context();\n".into()),
            HunkLine::Removed("old_call();\n".into()),
            HunkLine::Added("new_call();\n".into()),
        ],
    }]);
    let mut context = ToolContext::default();
    context.insert_result(FilesChanged(vec![change])).ok();
    let facts = context.result_context();
    let args = json!({ "path": "a.rs", "edits": [] });
    let reply = "Edited a.rs: line 2.\n[lines 1-2]\nkept_context();\nnew_call();";
    let drew = drawn_with(edit, "edit", args.clone(), reply, Some(&facts));
    for text in [
        "Edited a.rs: line 2.",
        "kept_context();",
        "old_call();",
        "new_call();",
    ] {
        assert!(drew.contains(text), "{text}: {drew}");
    }
    assert!(!drew.contains("[lines 1-2]"), "{drew}");
    let without = drawn(edit, "edit", args, "Edited a.rs: line 2.");
    assert!(!without.contains("new_call"), "{without}");
    assert!(without.contains("Edited a.rs: line 2."), "{without}");
}
