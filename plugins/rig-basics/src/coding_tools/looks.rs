//! How the terminal view (rig-tui) draws the tools' calls: each shows its
//! arguments and its whole result. `read` shows the file's lines in the
//! colors of its language, numbered from what the call published
//! ([`FilesRead`]); `write` the new file the same way; `edit` its reply and
//! the whole change it published ([`FilesChanged`]) as a diff; and `shell`
//! the command, as bash, and its output.

use rig_harness::prelude::*;
use rig_harness::rig_core::tool::{FilesChanged, FilesRead, LinesShown};
use rig_tools::{Edit, Read, Shell, Write};
use rig_tui::highlight::{Language, code_rows, highlight};
use rig_tui::ratatui::style::{Style, Stylize};
use rig_tui::ratatui::text::{Line, Span};
use rig_tui::{OUTPUT_INDENT, RenderBatch, ToolCallView, text_rows};

/// Draws the calls of `tool` with `look`.
fn draw(batch: &mut RenderBatch, tool: &str, look: fn(&ToolCallView<'_>) -> Vec<Line<'static>>) {
    for (call, output) in batch.tools_mut(tool) {
        output.lines = look(&call.view());
    }
}

pub(super) fn draw_read(mut batch: ResMut<RenderBatch>) {
    draw(&mut batch, Read::NAME, read);
}

pub(super) fn draw_edit(mut batch: ResMut<RenderBatch>) {
    draw(&mut batch, Edit::NAME, edit);
}

pub(super) fn draw_write(mut batch: ResMut<RenderBatch>) {
    draw(&mut batch, Write::NAME, write);
}

pub(super) fn draw_shell(mut batch: ResMut<RenderBatch>) {
    draw(&mut batch, Shell::NAME, shell);
}

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
        Some(text) if !view.failed() => rows.extend(read_rows(&text, &path, shown)),
        _ => rows.extend(view.result_lines()),
    }
    rows
}

/// A read's lines in the colors of `path`'s language. With the lines it
/// published it showed, those are numbered from the first, the numbers
/// dim, and any line after them, such as a note of the lines left, is dim;
/// without, every line is code, unnumbered.
fn read_rows(text: &str, path: &str, shown: Option<LinesShown>) -> Vec<Line<'static>> {
    let lines: Vec<&str> = text.lines().collect();
    let count = shown.map_or(lines.len(), |shown| shown.count);
    let (code, notes) = lines.split_at_checked(count).unwrap_or((&lines, &[]));
    let colored = highlight(&code.join("\n"), Language::File(path)).unwrap_or_default();
    let mut colored = colored.into_iter();
    let code = code.iter().enumerate().map(|(index, line)| {
        let mut spans = vec![Span::from(OUTPUT_INDENT)];
        if let Some(shown) = shown {
            spans.push(Span::from(format!("{:>6}  ", shown.first + index)).dim());
        }
        let plain = || vec![Span::from(line.replace('\t', "    "))];
        spans.extend(colored.next().unwrap_or_else(plain));
        Line::from(spans)
    });
    let notes = notes
        .iter()
        .map(|line| Line::from(format!("{OUTPUT_INDENT}{line}")).dim());
    code.chain(notes).collect()
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

/// The reply without the file's lines it shows, then the hunks of each
/// change the call published, as a diff under the header that names the
/// file; just the reply when it published none, as after a resume without
/// its effect log.
pub(super) fn edit(view: &ToolCallView<'_>) -> Vec<Line<'static>> {
    let path = view.argument("path").unwrap_or_default().to_owned();
    let mut lines = vec![view.header(format!("edit {path}"), "")];
    let changed = view
        .facts
        .and_then(|facts| facts.get::<FilesChanged>().ok().flatten());
    match (view.result_text(), changed) {
        (Some(reply), Some(FilesChanged(changes))) if !view.failed() => {
            // The file's lines the reply shows the model are in the diff.
            let said = reply.split("\n[lines ").next().unwrap_or_default();
            lines.extend(text_rows(said, Style::new().dim()));
            for change in &changes {
                lines.extend(rig_tui::diff::change_lines(change, &view.theme));
            }
        }
        _ => lines.extend(view.result_lines()),
    }
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
        lines.push(if index == 0 {
            view.header_spans("$", code)
        } else {
            Line::from([vec![Span::from("    ")], code].concat())
        });
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
