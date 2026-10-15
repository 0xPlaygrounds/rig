use rig_harness::rig_core::message::{ToolCall, ToolFunction, ToolName};
use rig_harness::rig_core::serde_json::{Value, json};
use rig_harness::rig_core::tool::{
    ContentHash, FileRead, FilesRead, LinesShown, ToolContext, ToolResult, ToolResultContext,
};
use rig_tui::ratatui::text::Line;
use rig_tui::{CallState, ToolCallView};

use super::{read, shell, write};

/// What `look` draws for a `tool` call with `args` that answered `output`.
fn drawn(
    look: fn(&ToolCallView<'_>) -> Vec<Line<'static>>,
    tool: &str,
    args: Value,
    output: &str,
) -> String {
    drawn_with(look, tool, args, output, None, false)
}

/// [`drawn`], for a call that published `facts`, expanded or not.
fn drawn_with(
    look: fn(&ToolCallView<'_>) -> Vec<Line<'static>>,
    tool: &str,
    args: Value,
    output: &str,
    facts: Option<&ToolResultContext>,
    expanded: bool,
) -> String {
    let Ok(name) = ToolName::new(tool) else {
        return String::new();
    };
    let call = ToolCall::from_wire("c1", ToolFunction::new(name, args));
    let result = call.answer(&ToolResult::success(output.to_owned().into()));
    let mut view = ToolCallView::new(&call, Some(&result), CallState::Done);
    view.facts = facts;
    view.expanded = expanded;
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
    let read_one = drawn(read, "read", json!({ "path": "a" }), "hi");
    assert!(read_one.ends_with("⎿ 1 line"), "{read_one}");
    let read_two = drawn(read, "read", json!({ "path": "a", "limit": 2 }), "hi\nho");
    assert!(read_two.contains("first 2 lines"), "{read_two}");
    assert!(read_two.ends_with("⎿ 2 lines"), "{read_two}");
}

#[test]
fn a_read_with_a_limit_counts_the_lines_it_read_not_its_note() {
    let output = "     1\tone\n     2\ttwo\n[5 more lines; read on with offset 3]\n";
    let shown = drawn(read, "read", json!({ "path": "a", "limit": 2 }), output);
    assert!(shown.ends_with("⎿ 2 lines"), "{shown}");
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
        let output = "     3\tthree\n     4\tfour\n[3 more lines; read on with offset 5]\n";
        drawn_with(
            read,
            "read",
            json!({ "path": "a" }),
            output,
            Some(&facts),
            false,
        )
    };
    let part = shown(3, 2);
    assert!(part.ends_with("⎿ lines 3–4 of 7"), "{part}");
    let whole = shown(1, 7);
    assert!(whole.ends_with("⎿ 7 lines"), "{whole}");
}

/// Expanded (Ctrl+O, or a transcript search laying every call out whole),
/// a call draws all of its output, not only the lines a collapsed one
/// shows.
#[test]
fn an_expanded_call_draws_every_line() {
    let rows: Vec<String> = (1..=40).map(|row| format!("row-{row}")).collect();
    let output = rows.join("\n");
    let command = json!({ "command": "seq 1 40" });
    let collapsed = drawn(shell, "shell", command.clone(), &output);
    assert!(!collapsed.contains("row-12\n"), "{collapsed}");
    let expanded = drawn_with(shell, "shell", command, &output, None, true);
    assert!(
        expanded.contains("row-1\n") && expanded.contains("row-12\n"),
        "{expanded}"
    );
    let content = json!({ "path": "a", "content": output });
    let wrote = drawn_with(write, "write", content, "ok", None, true);
    assert!(wrote.ends_with("row-40"), "{wrote}");
}
