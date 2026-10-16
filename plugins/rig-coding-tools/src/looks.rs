//! How the terminal view (rig-tui) draws the tools' calls: each shows its
//! arguments and its whole result. `read` shows the file's lines and
//! `write` the new file in the colors of its language, `edit` its change
//! as a diff, and `shell` the command, as bash, and its output.

use rig_harness::rig_core::tool::{FilesRead, LinesShown};
use rig_tui::ToolCallView;
use rig_tui::highlight::{Language, code_rows, highlight};
use rig_tui::ratatui::style::Stylize;
use rig_tui::ratatui::text::{Line, Span};

pub(super) fn read(view: &ToolCallView<'_>) -> Vec<Line<'static>> {
    let path = view.argument("path").unwrap_or_default().to_owned();
    let number = |key: &str| usize::try_from(view.call.function.arguments.get(key)?.as_u64()?).ok();
    let range = match (number("offset"), number("limit")) {
        (Some(offset), Some(limit)) => format!("lines {offset}..{}", offset.saturating_add(limit)),
        (Some(offset), None) => format!("from line {offset}"),
        (None, Some(limit)) => format!("first {}", lines(limit)),
        (None, None) => String::new(),
    };
    let shown = view
        .facts
        .and_then(|facts| facts.get::<FilesRead>().ok().flatten());
    let shown = shown.and_then(|read| read.0.first().and_then(|file| file.lines));
    let detail = shown.map_or(range, span);
    let mut rows = vec![view.header(format!("read {path}"), detail)];
    match view.result_text() {
        Some(text) if !view.failed() => rows.extend(numbered_rows(&text, &path)),
        _ => rows.extend(view.result_lines()),
    }
    rows
}

/// A read's line split into its number and its code, if it is one.
fn numbered(line: &str) -> Option<(&str, &str)> {
    let (number, code) = line.split_once('\t')?;
    let digits = number.trim();
    let is_number = !digits.is_empty() && digits.chars().all(|c| c.is_ascii_digit());
    is_number.then_some((number, code))
}

/// A read's numbered lines, the numbers dim and the code in the colors of
/// `path`'s language; any other line, such as a note of the lines left,
/// dim.
fn numbered_rows(text: &str, path: &str) -> Vec<Line<'static>> {
    let code: Vec<&str> = text
        .lines()
        .filter_map(numbered)
        .map(|(_, code)| code)
        .collect();
    let colored = highlight(&code.join("\n"), Language::File(path)).unwrap_or_default();
    let mut colored = colored.into_iter();
    let rows = text.lines().enumerate().map(|(index, line)| {
        let prefix = if index == 0 { "  ⎿ " } else { "    " };
        let Some((number, code)) = numbered(line) else {
            return Line::from(format!("{prefix}{line}")).dim();
        };
        let plain = || vec![Span::from(code.replace('\t', "    "))];
        let code = colored.next().unwrap_or_else(plain);
        let number = Span::from(format!("{number}  ")).dim();
        Line::from([vec![Span::from(prefix), number], code].concat())
    });
    rows.collect()
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

pub(super) fn edit(view: &ToolCallView<'_>) -> Vec<Line<'static>> {
    let path = view.argument("path").unwrap_or_default().to_owned();
    let mut lines = vec![view.header(format!("edit {path}"), "")];
    lines.extend(view.result_lines());
    lines
}

pub(super) fn write(view: &ToolCallView<'_>) -> Vec<Line<'static>> {
    let path = view.argument("path").unwrap_or_default().to_owned();
    let content = view.argument("content").unwrap_or_default();
    let mut rows = vec![view.header(format!("write {path}"), lines(content.lines().count()))];
    if view.failed() {
        rows.extend(view.result_lines());
    } else {
        rows.extend(code_rows(content, Language::File(&path)));
    }
    rows
}

pub(super) fn shell(view: &ToolCallView<'_>) -> Vec<Line<'static>> {
    let command = view.argument("command").unwrap_or_default();
    let colored = highlight(command, Language::Named("bash")).unwrap_or_default();
    let mut lines = Vec::new();
    for (index, line) in command.lines().enumerate() {
        let plain = || vec![Span::from(line.to_owned()).bold()];
        let code = colored.get(index).cloned().unwrap_or_else(plain);
        let mut row = if index == 0 {
            view.header("$", "")
        } else {
            Line::default()
        };
        row.spans
            .push(Span::from(if index == 0 { " " } else { "    " }));
        row.spans.extend(code);
        lines.push(row);
    }
    if lines.is_empty() {
        lines.push(view.header("$", ""));
    }
    lines.extend(view.result_lines());
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
