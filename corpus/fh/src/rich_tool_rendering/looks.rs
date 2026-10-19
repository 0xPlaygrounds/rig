//! The looks of the built-in tools' calls, each a function of the call as
//! the terminal view gives it ([`ToolCallView`]).

use rig_harness::rig_core::tool::{ExitStatus, FileChange, FilesChanged, FilesRead, LinesShown};
use rig_tools::fs::file_change;
use rig_tools::{SearchHits, split_status};
use rig_tui::ratatui::style::{Color, Style, Stylize};
use rig_tui::ratatui::text::{Line, Span};
use rig_tui::{OUTPUT_INDENT, ToolCallView, text_rows};
use serde_json::Value;

use crate::rich_tool_rendering::highlight::{Lang, spans};

/// The background of an added and of a removed line.
const ADDED: Color = Color::Indexed(22);
const REMOVED: Color = Color::Indexed(52);

/// Draws `view` with the look of its tool.
pub(crate) fn render(view: &ToolCallView<'_>) -> Vec<Line<'static>> {
    match view.name() {
        "edit" | "write" => change(view),
        "shell" => shell(view),
        "read" => read(view),
        "search" => search(view),
        _ => view.default_lines(),
    }
}

/// A line of a unified diff: its sign, then its text highlighted, on a
/// green or red background for an added or removed line; a hunk header
/// dimmed.
fn diff_line(line: &str, lang: Lang) -> Line<'static> {
    if line.starts_with("@@") {
        return Line::from(format!("  {line}")).dim();
    }
    let mut chars = line.chars();
    let sign = chars.next().unwrap_or(' ');
    let mut parts = vec![Span::from(format!("{sign} "))];
    parts.extend(spans(&chars.as_str().replace('\t', "    "), lang));
    let line = Line::from(parts);
    match sign {
        '+' => line.style(Style::new().bg(ADDED)),
        '-' => line.style(Style::new().bg(REMOVED)),
        _ => line.dim(),
    }
}

/// The unified diffs of a call of `edit` or `write`: what the tool
/// published once it ran in this run, else what the arguments say so far
/// (while they stream, while it runs, after a restore).
fn diffs(view: &ToolCallView<'_>, path: &str) -> Vec<String> {
    let published = view
        .facts
        .and_then(|facts| facts.get::<FilesChanged>().ok().flatten());
    if let Some(FilesChanged(changes)) = published {
        return changes
            .iter()
            .map(FileChange::unified)
            .filter(|diff| !diff.is_empty())
            .collect();
    }
    let arguments = &view.call.function.arguments;
    let text = |value: &Value, key| {
        let text = value.get(key).and_then(Value::as_str).unwrap_or_default();
        format!("{}\n", text.trim_end_matches('\n'))
    };
    let made = |before: Option<&str>, after: &str| {
        let diff = file_change(path, before, Some(after)).unified();
        (!diff.is_empty()).then_some(diff)
    };
    match arguments.get("edits").and_then(Value::as_array) {
        Some(edits) => edits
            .iter()
            .filter_map(|edit| made(Some(&text(edit, "old_text")), &text(edit, "new_text")))
            .collect(),
        None => made(None, &text(&Value::Object(arguments.clone()), "content"))
            .into_iter()
            .collect(),
    }
}

/// `edit` and `write`: the path with the lines added and removed, then
/// the change as a diff.
fn change(view: &ToolCallView<'_>) -> Vec<Line<'static>> {
    let path = view.argument("path").unwrap_or_default().to_owned();
    let diffs = diffs(view, &path);
    let body: Vec<&str> = diffs
        .iter()
        .flat_map(|diff| diff.lines())
        .filter(|line| !line.starts_with("---") && !line.starts_with("+++"))
        .collect();
    let added = body.iter().filter(|line| line.starts_with('+')).count();
    let removed = body.iter().filter(|line| line.starts_with('-')).count();
    let title = format!("{} {path}", view.name());
    let mut lines = vec![view.header(title, format!("+{added} -{removed}"))];
    if view.failed() {
        lines.extend(view.result_lines());
        return lines;
    }
    let lang = Lang::of(&path);
    lines.extend(body.iter().map(|line| diff_line(line, lang)));
    lines
}

/// `shell`: `$ command`; while it runs, its output so far; then its
/// output, its exit code and how long it ran, red when it failed.
fn shell(view: &ToolCallView<'_>) -> Vec<Line<'static>> {
    let command = view.argument("command").unwrap_or_default();
    let mut command_lines = command.lines();
    let first = command_lines.next().unwrap_or_default();
    let mut lines = vec![view.header(format!("$ {first}"), "")];
    lines.extend(command_lines.map(|line| Line::from(format!("  {line}")).bold()));
    let text = view.result_text();
    let Some(text) = text.as_deref().filter(|_| !view.failed()) else {
        match view.output {
            Some(output) => lines.extend(printed(output, Style::new().dim())),
            None => lines.extend(view.result_lines()),
        }
        return lines;
    };
    let (output, said) = split_status(text);
    let status = view
        .facts
        .and_then(|facts| facts.get::<ExitStatus>().ok().flatten());
    let failed = status.is_some_and(|status| !status.success());
    let style = if failed {
        Style::new().red()
    } else {
        Style::new().dim()
    };
    lines.extend(printed(output, style));
    // Answered in an earlier run, the call has no facts: the tool's own
    // line says how it ended.
    let Some(status) = status else {
        lines.extend(said.map(|said| Line::from(format!("{OUTPUT_INDENT}{said}")).dim()));
        return lines;
    };
    let said = match status {
        ExitStatus::Exited(code) => format!("exit {code}"),
        ExitStatus::Signaled => "killed by a signal".to_owned(),
        ExitStatus::TimedOut => "timed out".to_owned(),
        ExitStatus::Killed => "killed".to_owned(),
    };
    let took = view
        .took
        .map(|took| format!(" · {:.1}s", took.as_secs_f64()))
        .unwrap_or_default();
    let summary = Line::from(format!("{OUTPUT_INDENT}{said}{took}"));
    lines.push(if failed {
        summary.red()
    } else {
        summary.green()
    });
    lines
}

/// What a command printed. When the tool cut the output, its first line
/// is the tool's note saying how much and which file has all of it.
fn printed(output: &str, style: Style) -> Vec<Line<'static>> {
    let line = |line: &str| Line::styled(format!("    {}", line.replace('\t', "    ")), style);
    output.lines().map(line).collect()
}

/// `read`: a line with the path and the lines it showed, then the
/// content.
fn read(view: &ToolCallView<'_>) -> Vec<Line<'static>> {
    let path = view.argument("path").unwrap_or_default().to_owned();
    let Some(text) = view.result_text().filter(|_| !view.failed()) else {
        let mut lines = vec![view.header(format!("read {path}"), "")];
        lines.extend(view.result_lines());
        return lines;
    };
    // Answered in an earlier run, the call has no facts: no range.
    let read = view
        .facts
        .and_then(|facts| facts.get::<FilesRead>().ok().flatten());
    let shown = read.and_then(|read| read.0.first().and_then(|file| file.lines));
    let range = shown.map(range).unwrap_or_default();
    let mut lines = vec![view.header(format!("read {path}"), range)];
    let lang = Lang::of(&path);
    lines.extend(text.lines().map(|line| {
        let mut parts = vec![Span::from("    ")];
        parts.extend(spans(&line.replace('\t', "  "), lang));
        Line::from(parts)
    }));
    lines
}

/// The lines a read showed: `(12 lines)` for a whole file, else
/// `(lines 3–4 of 12)`.
fn range(shown: LinesShown) -> String {
    let last = shown.first.saturating_add(shown.count).saturating_sub(1);
    match (shown.whole(), shown.count) {
        (true, 1) => "(1 line)".to_owned(),
        (true, total) => format!("({total} lines)"),
        (false, 0) => format!("(no lines of {})", shown.total),
        (false, _) => format!("(lines {}–{last} of {})", shown.first, shown.total),
    }
}

/// `search`: how many matches, then each.
fn search(view: &ToolCallView<'_>) -> Vec<Line<'static>> {
    let pattern = view.argument("pattern").unwrap_or_default().to_owned();
    let title = format!("search {pattern}");
    let found = view
        .facts
        .and_then(|facts| facts.get::<SearchHits>().ok().flatten());
    let Some(SearchHits(found)) = found else {
        // Not run yet, failed, or answered in an earlier run.
        let mut lines = vec![view.header(title, "")];
        lines.extend(view.result_lines());
        return lines;
    };
    let count = match (found.hits.len(), found.stopped) {
        (1, _) => "1 match".to_owned(),
        (count, false) => format!("{count} matches"),
        (count, true) => format!("{count}+ matches"),
    };
    let hits: Vec<String> = found.hits.iter().map(ToString::to_string).collect();
    let mut lines = vec![view.header(title, count)];
    lines.extend(text_rows(&hits.join("\n"), Style::new().dim()));
    lines
}
