//! The looks of the built-in tools' calls, each a function of the call as
//! the terminal view gives it ([`ToolCallView`]).

use rig_harness::rig_core::tool::{ExitStatus, FilesChanged};
use rig_tools::SearchHits;
use rig_tools::fs::file_change;
use rig_tui::ratatui::style::{Color, Style, Stylize};
use rig_tui::ratatui::text::{Line, Span};
use rig_tui::{RESULT_LINES, ToolCallView, excerpt};
use serde_json::Value;

use crate::highlight::{Lang, spans};

/// Lines of a diff shown while it is collapsed.
pub const COLLAPSED_LINES: usize = 30;
/// Lines of a command's output shown while it is collapsed.
pub const SHELL_TAIL: usize = 5;
/// Matches of a search shown while it is collapsed.
const MATCHES: usize = 3;

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

/// The marker after a collapsed body: how many lines are not shown.
fn more(hidden: usize, view: &ToolCallView<'_>) -> Option<Line<'static>> {
    let how = if view.expanded {
        ""
    } else {
        " (Ctrl+O to expand)"
    };
    (hidden > 0).then(|| Line::from(format!("    … {hidden} more lines{how}")).dim())
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
            .into_iter()
            .filter_map(|change| change.diff)
            .collect();
    }
    let arguments = &view.call.function.arguments;
    let text = |value: &Value, key| {
        let text = value.get(key).and_then(Value::as_str).unwrap_or_default();
        format!("{}\n", text.trim_end_matches('\n'))
    };
    let made = |before: Option<&str>, after: &str| file_change(path, before, after).diff;
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
/// the change as a diff, cut after [`COLLAPSED_LINES`] lines.
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
        lines.extend(view.result_lines(RESULT_LINES));
        return lines;
    }
    let shown = view.limit(COLLAPSED_LINES);
    let lang = Lang::of(&path);
    lines.extend(body.iter().take(shown).map(|line| diff_line(line, lang)));
    lines.extend(more(body.len().saturating_sub(shown), view));
    lines
}

/// `shell`: `$ command`; while it runs, the output as it comes; then the
/// end of its output, its exit code and how long it ran, red when it
/// failed.
fn shell(view: &ToolCallView<'_>) -> Vec<Line<'static>> {
    let command = view.argument("command").unwrap_or_default();
    let mut command_lines = command.lines();
    let first = command_lines.next().unwrap_or_default();
    let mut lines = vec![view.header(format!("$ {first}"), "")];
    lines.extend(command_lines.map(|line| Line::from(format!("  {line}")).bold()));
    let status = view
        .facts
        .and_then(|facts| facts.get::<ExitStatus>().ok().flatten());
    let (Some(text), Some(status)) = (view.result_text(), status) else {
        // Running, or answered in an earlier run: as the view draws it.
        lines.extend(view.tail_lines(view.limit(SHELL_TAIL)));
        return lines;
    };
    // The shell tool ends its output with a line saying how it ended,
    // which `status` says as data.
    let mut output: Vec<&str> = text.lines().collect();
    output.pop();
    let shown = view.limit(SHELL_TAIL);
    let hidden = output.len().saturating_sub(shown);
    lines.extend(more(hidden, view));
    let style = if status.success() {
        Style::new().dim()
    } else {
        Style::new().red()
    };
    let tail = output.iter().skip(hidden);
    lines.extend(
        tail.map(|line| Line::styled(format!("    {}", line.replace('\t', "    ")), style)),
    );
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
    let summary = Line::from(format!("  ⎿ {said}{took}"));
    lines.push(if status.success() {
        summary.green()
    } else {
        summary.red()
    });
    lines
}

/// `read`: one line with the path and the lines read; the content when
/// expanded.
fn read(view: &ToolCallView<'_>) -> Vec<Line<'static>> {
    let path = view.argument("path").unwrap_or_default().to_owned();
    let Some(text) = view.result_text().filter(|_| !view.failed()) else {
        let mut lines = vec![view.header(format!("read {path}"), "")];
        lines.extend(view.result_lines(RESULT_LINES));
        return lines;
    };
    // `read` shows a numbered line per line read, then `[…]` notes.
    let count = text.lines().filter(|line| !line.starts_with('[')).count();
    let first = view
        .call
        .function
        .arguments
        .get("offset")
        .and_then(Value::as_u64)
        .and_then(|first| usize::try_from(first).ok())
        .unwrap_or(1);
    let range = match count.checked_sub(1) {
        None => "(empty)".to_owned(),
        Some(rest) => format!(
            "(lines {first}–{}, {count} lines)",
            first.saturating_add(rest)
        ),
    };
    let mut lines = vec![view.header(format!("read {path}"), range)];
    if view.expanded {
        let lang = Lang::of(&path);
        lines.extend(text.lines().map(|line| {
            let mut parts = vec![Span::from("    ")];
            parts.extend(spans(&line.replace('\t', "  "), lang));
            Line::from(parts)
        }));
    }
    lines
}

/// `search`: how many matches, then the first few.
fn search(view: &ToolCallView<'_>) -> Vec<Line<'static>> {
    let pattern = view.argument("pattern").unwrap_or_default().to_owned();
    let title = format!("search {pattern}");
    let found = view
        .facts
        .and_then(|facts| facts.get::<SearchHits>().ok().flatten());
    let Some(SearchHits(found)) = found else {
        // Not run yet, failed, or answered in an earlier run.
        let mut lines = vec![view.header(title, "")];
        lines.extend(view.result_lines(view.limit(MATCHES)));
        return lines;
    };
    let count = match (found.hits.len(), found.stopped) {
        (1, _) => "1 match".to_owned(),
        (count, false) => format!("{count} matches"),
        (count, true) => format!("{count}+ matches"),
    };
    let hits: Vec<String> = found.hits.iter().map(ToString::to_string).collect();
    let mut lines = vec![view.header(title, count)];
    lines.extend(excerpt(
        &hits.join("\n"),
        view.limit(MATCHES),
        Style::new().dim(),
    ));
    lines
}
