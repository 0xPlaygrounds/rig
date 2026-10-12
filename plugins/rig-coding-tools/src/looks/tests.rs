use rig_harness::rig_core::message::{ToolCall, ToolFunction, ToolName};
use rig_harness::rig_core::serde_json::{Value, json};
use rig_harness::rig_core::tool::ToolResult;
use rig_tui::ratatui::text::Line;
use rig_tui::{CallState, ToolCallView};

use super::{read, write};

/// What `look` draws for a `tool` call with `args` that answered `output`.
fn drawn(
    look: fn(&ToolCallView<'_>) -> Vec<Line<'static>>,
    tool: &str,
    args: Value,
    output: &str,
) -> String {
    let Ok(name) = ToolName::new(tool) else {
        return String::new();
    };
    let call = ToolCall::from_wire("c1", ToolFunction::new(name, args));
    let result = call.answer(&ToolResult::success(output.to_owned().into()));
    let view = ToolCallView::new(&call, Some(&result), CallState::Done);
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
