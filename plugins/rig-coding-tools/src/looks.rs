//! How the terminal view (rig-tui) draws the tools' calls: `read` and
//! `search` summarize what they found, `edit` shows its change as a diff,
//! `write` the start of the new file, `delete` and `move` the paths, and
//! `shell` the command and the end of its output.

use rig_tui::ratatui::style::{Style, Stylize};
use rig_tui::ratatui::text::Line;
use rig_tui::{RESULT_LINES, ToolCallView, excerpt};

pub(super) fn read(view: &ToolCallView<'_>) -> Vec<Line<'static>> {
    let path = view.argument("path").unwrap_or_default().to_owned();
    let number = |key: &str| usize::try_from(view.call.function.arguments.get(key)?.as_u64()?).ok();
    let range = match (number("offset"), number("limit")) {
        (Some(offset), Some(limit)) => format!("lines {offset}..{}", offset.saturating_add(limit)),
        (Some(offset), None) => format!("from line {offset}"),
        (None, Some(limit)) => format!("first {}", lines(limit)),
        (None, None) => String::new(),
    };
    let mut rows = vec![view.header(format!("read {path}"), range)];
    match view.result_text() {
        Some(_) if view.failed() => rows.extend(view.result_lines(RESULT_LINES)),
        Some(text) => {
            // The file's lines, without the read's `[… more lines …]` note.
            let read = text.lines().filter(|line| !line.starts_with('['));
            rows.push(Line::from(format!("  ⎿ {}", lines(read.count()))).dim());
        }
        None => {}
    }
    rows
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
    let mut rows = vec![view.header(format!("write {path}"), lines(content.lines().count()))];
    if view.failed() {
        rows.extend(view.result_lines(RESULT_LINES));
    } else {
        rows.extend(excerpt(content, RESULT_LINES, Style::new().dim()));
    }
    rows
}

pub(super) fn delete(view: &ToolCallView<'_>) -> Vec<Line<'static>> {
    let path = view.argument("path").unwrap_or_default().to_owned();
    let mut lines = vec![view.header(format!("delete {path}"), "")];
    if view.failed() {
        lines.extend(view.result_lines(RESULT_LINES));
    }
    lines
}

pub(super) fn moved(view: &ToolCallView<'_>) -> Vec<Line<'static>> {
    let from = view.argument("from").unwrap_or_default();
    let to = view.argument("to").unwrap_or_default();
    let mut lines = vec![view.header(format!("move {from} → {to}"), "")];
    if view.failed() {
        lines.extend(view.result_lines(RESULT_LINES));
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

/// `n` lines, said as a count: "1 line", "3 lines".
fn lines(n: usize) -> String {
    match n {
        1 => "1 line".to_owned(),
        n => format!("{n} lines"),
    }
}

#[cfg(test)]
mod tests;
