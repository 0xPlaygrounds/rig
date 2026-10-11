//! How the terminal view (rig-tui) draws the tools' calls: `read` and
//! `search` summarize what they found, `edit` shows its change as a diff,
//! `write` the start of the new file, and `shell` the command and the end
//! of its output.

use rig_tui::ratatui::style::{Style, Stylize};
use rig_tui::ratatui::text::Line;
use rig_tui::{RESULT_LINES, ToolCallView, excerpt};

pub(super) fn read(view: &ToolCallView<'_>) -> Vec<Line<'static>> {
    let path = view.argument("path").unwrap_or_default().to_owned();
    let number = |key: &str| view.call.function.arguments.get(key)?.as_u64();
    let range = match (number("offset"), number("limit")) {
        (Some(offset), Some(limit)) => format!("lines {offset}..{}", offset + limit),
        (Some(offset), None) => format!("from line {offset}"),
        (None, Some(limit)) => format!("first {limit} lines"),
        (None, None) => String::new(),
    };
    let mut lines = vec![view.header(format!("read {path}"), range)];
    match view.result_text() {
        Some(_) if view.failed() => lines.extend(view.result_lines(RESULT_LINES)),
        Some(text) => lines.push(Line::from(format!("  ⎿ {} lines", text.lines().count())).dim()),
        None => {}
    }
    lines
}

pub(super) fn edit(view: &ToolCallView<'_>) -> Vec<Line<'static>> {
    let path = view.argument("path").unwrap_or_default().to_owned();
    let mut lines = vec![view.header(format!("edit {path}"), "")];
    lines.extend(view.result_lines(RESULT_LINES));
    lines
}

pub(super) fn write(view: &ToolCallView<'_>) -> Vec<Line<'static>> {
    let path = view.argument("path").unwrap_or_default().to_owned();
    let content = view.argument("content").unwrap_or_default();
    let mut lines = vec![view.header(
        format!("write {path}"),
        format!("{} lines", content.lines().count()),
    )];
    if view.failed() {
        lines.extend(view.result_lines(RESULT_LINES));
    } else {
        lines.extend(excerpt(content, RESULT_LINES, Style::new().dim()));
    }
    lines
}

pub(super) fn shell(view: &ToolCallView<'_>) -> Vec<Line<'static>> {
    let command = view.argument("command").unwrap_or_default();
    let mut lines = Vec::new();
    for (index, line) in command.lines().enumerate() {
        if index == 0 {
            lines.push(view.header(format!("$ {line}"), ""));
        } else {
            lines.push(Line::from(format!("    {line}")).bold());
        }
    }
    if lines.is_empty() {
        lines.push(view.header("$", ""));
    }
    lines.extend(view.tail_lines(RESULT_LINES + 2));
    lines
}

pub(super) fn search(view: &ToolCallView<'_>) -> Vec<Line<'static>> {
    let pattern = view.argument("pattern").unwrap_or_default();
    let mut detail = String::new();
    if let Some(path) = view.argument("path") {
        detail.push_str(&format!("in {path}"));
    }
    if let Some(glob) = view.argument("glob") {
        detail.push_str(&format!(" ({glob})"));
    }
    let mut lines = vec![view.header(format!("search {pattern}"), detail.trim().to_owned())];
    lines.extend(view.result_lines(RESULT_LINES));
    lines
}
