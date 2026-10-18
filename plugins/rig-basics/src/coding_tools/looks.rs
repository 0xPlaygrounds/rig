//! How the terminal view (rig-tui) draws the tools' calls: each shows its
//! arguments and its whole result. `read` shows the file's lines in the
//! colors of its language, numbered from what the call published
//! ([`FilesRead`]); `write` the new file the same way; `edit` its reply and
//! the whole change it published ([`FilesChanged`]) as a diff; and `shell`
//! the command, as bash, and its output.

use rig_harness::rig_core::tool::{FileChange, FilesChanged, FilesRead, LinesShown};
use rig_tui::ToolCallView;
use rig_tui::highlight::{Language, code_rows, highlight};
use rig_tui::ratatui::style::Stylize;
use rig_tui::ratatui::text::{Line, Span};

pub(crate) fn read(view: &ToolCallView<'_>) -> Vec<Line<'static>> {
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
    let prefix = |index: usize| if index == 0 { "  ⎿ " } else { "    " };
    let code = code.iter().enumerate().map(|(index, line)| {
        let mut spans = vec![Span::from(prefix(index))];
        if let Some(shown) = shown {
            spans.push(Span::from(format!("{:>6}  ", shown.first + index)).dim());
        }
        let plain = || vec![Span::from(line.replace('\t', "    "))];
        spans.extend(colored.next().unwrap_or_else(plain));
        Line::from(spans)
    });
    let notes = notes
        .iter()
        .enumerate()
        .map(|(index, line)| Line::from(format!("{}{line}", prefix(count + index))).dim());
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

/// The reply, then the change the call published, as a diff; just the
/// reply when it published none, as after a resume without its effect log.
pub(crate) fn edit(view: &ToolCallView<'_>) -> Vec<Line<'static>> {
    let path = view.argument("path").unwrap_or_default().to_owned();
    let mut lines = vec![view.header(format!("edit {path}"), "")];
    let changed = view
        .facts
        .and_then(|facts| facts.get::<FilesChanged>().ok().flatten());
    let drawn = match (view.result_text(), changed) {
        (Some(reply), Some(FilesChanged(changes))) if !view.failed() => {
            let diff: String = changes.iter().map(FileChange::unified).collect();
            rig_tui::diff::unified_lines(&format!("{reply}\n{diff}"), &view.theme)
        }
        _ => None,
    };
    lines.extend(drawn.unwrap_or_else(|| view.result_lines()));
    lines
}

pub(crate) fn write(view: &ToolCallView<'_>) -> Vec<Line<'static>> {
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

pub(crate) fn shell(view: &ToolCallView<'_>) -> Vec<Line<'static>> {
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
