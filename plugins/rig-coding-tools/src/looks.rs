//! How the terminal view (rig-tui) draws the tools' calls: `read` and
//! `search` summarize what they found, `edit` shows its change as a diff,
//! `write` the start of the new file, `delete` and `move` the paths, `list`
//! the start of the listing, and `shell` the command and the end of its
//! output.

use rig_harness::rig_core::tool::{FilesRead, LinesShown};
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
    let shown = view
        .facts
        .and_then(|facts| facts.get::<FilesRead>().ok().flatten());
    let shown = shown.and_then(|read| read.0.first().and_then(|file| file.lines));
    match (view.result_text(), shown) {
        (Some(_), _) if view.failed() => rows.extend(view.result_lines(view.limit(RESULT_LINES))),
        (Some(_), Some(shown)) => rows.push(Line::from(format!("  ⎿ {}", span(shown))).dim()),
        // A call of an earlier run has no facts: count the file's lines,
        // without the read's `[… more lines …]` note.
        (Some(text), None) => {
            let read = text.lines().filter(|line| !line.starts_with('['));
            rows.push(Line::from(format!("  ⎿ {}", lines(read.count()))).dim());
        }
        (None, _) => {}
    }
    rows
}

/// The lines a read showed: "12 lines" for a whole file, else which of
/// them, "lines 3–4 of 12".
fn span(shown: LinesShown) -> String {
    let last = shown.first.saturating_add(shown.count).saturating_sub(1);
    match (shown.whole(), shown.count) {
        (true, _) => lines(shown.total),
        (false, 0) => format!("no lines of {}", shown.total),
        (false, _) => format!("lines {}–{last} of {}", shown.first, shown.total),
    }
}

pub(super) fn list(view: &ToolCallView<'_>) -> Vec<Line<'static>> {
    let path = view.argument("path").unwrap_or(".").to_owned();
    let depth = view
        .call
        .function
        .arguments
        .get("depth")
        .and_then(|depth| depth.as_u64());
    let depth = depth
        .map(|depth| format!("depth {depth}"))
        .unwrap_or_default();
    let mut lines = vec![view.header(format!("list {path}"), depth)];
    lines.extend(view.result_lines(view.limit(RESULT_LINES)));
    lines
}

pub(super) fn edit(view: &ToolCallView<'_>) -> Vec<Line<'static>> {
    let path = view.argument("path").unwrap_or_default().to_owned();
    let mut lines = vec![view.header(format!("edit {path}"), "")];
    lines.extend(view.result_lines(view.limit(RESULT_LINES)));
    lines
}

pub(super) fn write(view: &ToolCallView<'_>) -> Vec<Line<'static>> {
    let path = view.argument("path").unwrap_or_default().to_owned();
    let content = view.argument("content").unwrap_or_default();
    let mut rows = vec![view.header(format!("write {path}"), lines(content.lines().count()))];
    if view.failed() {
        rows.extend(view.result_lines(view.limit(RESULT_LINES)));
    } else {
        rows.extend(excerpt(
            content,
            view.limit(RESULT_LINES),
            Style::new().dim(),
        ));
    }
    rows
}

pub(super) fn delete(view: &ToolCallView<'_>) -> Vec<Line<'static>> {
    let path = view.argument("path").unwrap_or_default().to_owned();
    let mut lines = vec![view.header(format!("delete {path}"), "")];
    if view.failed() {
        lines.extend(view.result_lines(view.limit(RESULT_LINES)));
    }
    lines
}

pub(super) fn moved(view: &ToolCallView<'_>) -> Vec<Line<'static>> {
    let from = view.argument("from").unwrap_or_default();
    let to = view.argument("to").unwrap_or_default();
    let mut lines = vec![view.header(format!("move {from} → {to}"), "")];
    if view.failed() {
        lines.extend(view.result_lines(view.limit(RESULT_LINES)));
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
    lines.extend(view.tail_lines(view.limit(RESULT_LINES + 2)));
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
    lines.extend(view.result_lines(view.limit(RESULT_LINES)));
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
