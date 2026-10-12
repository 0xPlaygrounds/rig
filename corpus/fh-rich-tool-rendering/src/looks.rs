//! The looks of the built-in tools' calls, each a function of the call and
//! [`ToolViews`].

use rig_harness::rig_core::message::ToolCall;
use rig_tools::fs::file_change;
use rig_tui::ratatui::style::{Color, Style, Stylize};
use rig_tui::ratatui::text::{Line, Span};
use rig_tui::{CallState, RESULT_LINES, ToolCallView, excerpt};
use serde_json::Value;

use crate::ToolViews;
use crate::highlight::{Lang, spans};

/// Lines of a diff shown while it is collapsed.
pub const COLLAPSED_LINES: usize = 30;
/// Lines of a command's output shown while it is collapsed.
pub const SHELL_TAIL: usize = 5;
/// Matches of a search shown while it is collapsed.
const MATCHES: usize = 3;
/// Most lines a call shows even expanded, so a huge one stays cheap to
/// lay out.
const EXPANDED_LINES: usize = 2000;

/// The background of an added and of a removed line.
const ADDED: Color = Color::Indexed(22);
const REMOVED: Color = Color::Indexed(52);

/// Draws `view` with the look of its tool.
pub(crate) fn render(view: &ToolCallView<'_>, views: &ToolViews) -> Vec<Line<'static>> {
    let expanded = views.expanded();
    match view.name() {
        "edit" => edit(view, expanded),
        "write" => write(view, expanded),
        "shell" => shell(view, views, expanded),
        "read" => read(view, expanded),
        "search" => search(view, expanded),
        _ => view.default_lines(),
    }
}

/// How many lines a call shows.
fn limit(expanded: bool, collapsed: usize) -> usize {
    if expanded { EXPANDED_LINES } else { collapsed }
}

/// The marker after a collapsed body: how many lines are not shown.
fn more(hidden: usize, expanded: bool) -> Option<Line<'static>> {
    let how = if expanded { "" } else { " (Ctrl+O to expand)" };
    (hidden > 0).then(|| Line::from(format!("    … {hidden} more lines{how}")).dim())
}

/// A line of a diff: its sign, then its text highlighted, on a green or
/// red background for an added or removed line.
fn diff_line(sign: char, text: &str, lang: Lang) -> Line<'static> {
    let background = match sign {
        '+' => Some(ADDED),
        '-' => Some(REMOVED),
        _ => None,
    };
    let mut parts = vec![Span::from(format!("{sign} "))];
    parts.extend(spans(&text.replace('\t', "    "), lang));
    let line = Line::from(parts);
    match background {
        Some(color) => line.style(Style::new().bg(color)),
        None => line.dim(),
    }
}

/// The diff lines of each replacement of an `edit` call, from its
/// arguments, so a call shows its change while it streams, runs and after
/// a restore; with the lines added and removed.
fn replacements(call: &ToolCall, path: &str) -> (Vec<(char, String)>, usize, usize) {
    let edits = call
        .function
        .arguments
        .get("edits")
        .and_then(Value::as_array);
    let (mut lines, mut added, mut removed) = (Vec::new(), 0, 0);
    for (index, edit) in edits.into_iter().flatten().enumerate() {
        let text = |key| {
            let text = edit.get(key).and_then(Value::as_str).unwrap_or_default();
            format!("{}\n", text.trim_end_matches('\n'))
        };
        let change = file_change(path, Some(&text("old_text")), &text("new_text"));
        added += change.added;
        removed += change.removed;
        if index > 0 {
            lines.push(('⋯', String::new()));
        }
        let diff = change.diff.unwrap_or_default();
        for line in diff.lines().filter(|line| !is_header(line)) {
            let mut chars = line.chars();
            let sign = chars.next().unwrap_or(' ');
            lines.push((sign, chars.as_str().to_owned()));
        }
    }
    (lines, added, removed)
}

/// Whether `line` is a unified diff's file or hunk header.
fn is_header(line: &str) -> bool {
    line.starts_with("---") || line.starts_with("+++") || line.starts_with("@@")
}

/// The diff's lines under the header, cut to `limit`, then how many more.
fn diff_body(lines: &[(char, String)], lang: Lang, expanded: bool) -> Vec<Line<'static>> {
    let shown = limit(expanded, COLLAPSED_LINES);
    let mut body: Vec<Line<'static>> = lines
        .iter()
        .take(shown)
        .map(|(sign, text)| match sign {
            '⋯' => Line::from("  ⋯").dim(),
            sign => diff_line(*sign, text, lang),
        })
        .collect();
    body.extend(more(lines.len().saturating_sub(shown), expanded));
    body
}

/// The error of a failed call, or a running call's progress.
fn status_lines(view: &ToolCallView<'_>) -> Option<Vec<Line<'static>>> {
    if view.failed() {
        return Some(view.result_lines(RESULT_LINES));
    }
    let running = matches!(view.state, CallState::Running(_)) && view.result.is_none();
    running.then(|| view.progress_lines())
}

/// `edit`: the path, then the change as a diff.
fn edit(view: &ToolCallView<'_>, expanded: bool) -> Vec<Line<'static>> {
    let path = view.argument("path").unwrap_or_default().to_owned();
    let (diff, added, removed) = replacements(view.call, &path);
    let mut lines = vec![view.header(format!("edit {path}"), format!("+{added} -{removed}"))];
    if let Some(status) = status_lines(view).filter(|_| view.failed()) {
        lines.extend(status);
        return lines;
    }
    lines.extend(diff_body(&diff, Lang::of(&path), expanded));
    lines
}

/// `write`: the path, then the whole new file as added lines.
fn write(view: &ToolCallView<'_>, expanded: bool) -> Vec<Line<'static>> {
    let path = view.argument("path").unwrap_or_default().to_owned();
    let content = view.argument("content").unwrap_or_default();
    let count = content.lines().count();
    let mut lines = vec![view.header(format!("write {path}"), format!("+{count}"))];
    if let Some(status) = status_lines(view).filter(|_| view.failed()) {
        lines.extend(status);
        return lines;
    }
    let added: Vec<(char, String)> = content.lines().map(|line| ('+', line.to_owned())).collect();
    lines.extend(diff_body(&added, Lang::of(&path), expanded));
    lines
}

/// A command's output and how it ended, from the `shell` tool's text: its
/// last line says `[exit code N]`, or why it was stopped.
fn split_exit(text: &str) -> (&str, Option<Ended>) {
    let trimmed = text.trim_end();
    let Some(open) = trimmed.rfind('[').filter(|_| trimmed.ends_with(']')) else {
        return (text, None);
    };
    let (output, marker) = trimmed.split_at(open);
    let inner = marker.trim_start_matches('[').trim_end_matches(']');
    let ended = match inner.strip_prefix("exit code ") {
        Some(code) => code.parse().ok().map(Ended::Exit),
        None if inner.starts_with("killed") || inner.starts_with("timed out") => {
            Some(Ended::Stopped(inner.to_owned()))
        }
        None => None,
    };
    match ended {
        Some(ended) => (output.trim_end_matches('\n'), Some(ended)),
        None => (text, None),
    }
}

/// How a command ended.
enum Ended {
    Exit(i64),
    Stopped(String),
}

/// `shell`: `$ command`, the end of its output, then its exit code and how
/// long it ran, red when it failed.
fn shell(view: &ToolCallView<'_>, views: &ToolViews, expanded: bool) -> Vec<Line<'static>> {
    let command = view.argument("command").unwrap_or_default();
    let mut command_lines = command.lines();
    let first = command_lines.next().unwrap_or_default();
    let mut lines = vec![view.header(format!("$ {first}"), "")];
    lines.extend(command_lines.map(|line| Line::from(format!("  {line}")).bold()));
    let Some(text) = view.result_text() else {
        lines.extend(view.progress_lines());
        return lines;
    };
    let (output, ended) = split_exit(&text);
    let output: Vec<&str> = output.lines().collect();
    let shown = limit(expanded, SHELL_TAIL);
    let hidden = output.len().saturating_sub(shown);
    lines.extend(more(hidden, expanded));
    let style = if view.failed() {
        Style::new().red()
    } else {
        Style::new().dim()
    };
    let tail = output.iter().skip(hidden);
    lines.extend(
        tail.map(|line| Line::styled(format!("    {}", line.replace('\t', "    ")), style)),
    );
    let took = views.duration(&view.call.id).map_or_else(
        || "earlier run".to_owned(),
        |took| format!("{:.1}s", took.as_secs_f64()),
    );
    let (said, ok) = match ended {
        Some(Ended::Exit(code)) => (format!("exit {code}"), code == 0 && !view.failed()),
        Some(Ended::Stopped(why)) => (why, false),
        None => ("ended".to_owned(), !view.failed()),
    };
    let summary = Line::from(format!("  ⎿ {said} · {took}"));
    lines.push(if ok { summary.green() } else { summary.red() });
    lines
}

/// `read`: one line with the path and the lines read; the content when
/// expanded.
fn read(view: &ToolCallView<'_>, expanded: bool) -> Vec<Line<'static>> {
    let path = view.argument("path").unwrap_or_default().to_owned();
    let Some(text) = view.result_text().filter(|_| !view.failed()) else {
        let mut lines = vec![view.header(format!("read {path}"), "")];
        lines.extend(status_lines(view).unwrap_or_default());
        return lines;
    };
    // `read` numbers each line it shows: `     12\ttext`.
    let numbers: Vec<&str> = text
        .lines()
        .filter_map(|line| line.split_once('\t').map(|(number, _)| number.trim()))
        .filter(|number| number.parse::<usize>().is_ok())
        .collect();
    let range = match (numbers.first(), numbers.last()) {
        (Some(first), Some(last)) => format!("(lines {first}–{last}, {} lines)", numbers.len()),
        _ => "(empty)".to_owned(),
    };
    let mut lines = vec![view.header(format!("read {path}"), range)];
    if expanded {
        let lang = Lang::of(&path);
        let content = text.lines().take(EXPANDED_LINES).map(|line| {
            let mut parts = vec![Span::from("    ")];
            parts.extend(spans(&line.replace('\t', "  "), lang));
            Line::from(parts)
        });
        lines.extend(content);
    }
    lines
}

/// `search`: how many matches, then the first few.
fn search(view: &ToolCallView<'_>, expanded: bool) -> Vec<Line<'static>> {
    let pattern = view.argument("pattern").unwrap_or_default().to_owned();
    let Some(text) = view.result_text().filter(|_| !view.failed()) else {
        let mut lines = vec![view.header(format!("search {pattern}"), "")];
        lines.extend(status_lines(view).unwrap_or_default());
        return lines;
    };
    // `search` answers a `path:line: text` line per match.
    let hits: Vec<&str> = text
        .lines()
        .filter(|line| !line.starts_with('[') && line.contains(':'))
        .collect();
    let count = match hits.len() {
        1 => "1 match".to_owned(),
        count => format!("{count} matches"),
    };
    let mut lines = vec![view.header(format!("search {pattern}"), count)];
    let shown = limit(expanded, MATCHES);
    lines.extend(excerpt(&hits.join("\n"), shown, Style::new().dim()));
    lines
}

/// What is known of a tool call whose argument JSON `json` is still
/// streaming: the file it names or its command, and the end of the text
/// it writes so far.
pub(crate) fn streaming(json: &str) -> Vec<Line<'static>> {
    let path = partial_string(json, "path");
    let command = partial_string(json, "command");
    let text = partial_string(json, "content").or_else(|| partial_string(json, "new_text"));
    let mut lines = Vec::new();
    match (path, command) {
        (Some(path), _) => {
            let so_far = text
                .as_deref()
                .map(|text| text.lines().count())
                .unwrap_or(0);
            lines.push(Line::from(format!("◌ {path} · {so_far} lines so far…")).yellow());
            let lang = Lang::of(&path);
            if let Some(text) = &text {
                let all: Vec<&str> = text.lines().collect();
                let tail = all.iter().skip(all.len().saturating_sub(3));
                lines.extend(tail.map(|line| diff_line('+', line, lang)));
            }
        }
        (None, Some(command)) => lines.push(Line::from(format!("◌ $ {command}…")).yellow()),
        (None, None) => {
            lines.push(Line::from(format!("◌ a tool call, {} bytes so far…", json.len())).yellow());
        }
    }
    lines
}

/// The value of the string field `key` of the JSON object `json`, which may
/// be cut anywhere: as much of it as has come.
fn partial_string(json: &str, key: &str) -> Option<String> {
    let quoted = format!("\"{key}\"");
    let (_, after) = json.split_once(&quoted)?;
    let after = after.trim_start().strip_prefix(':')?.trim_start();
    let mut chars = after.strip_prefix('"')?.chars();
    let mut value = String::new();
    while let Some(character) = chars.next() {
        match character {
            '"' => break,
            '\\' => match chars.next() {
                Some('n') => value.push('\n'),
                Some('t') => value.push('\t'),
                Some('r') => {}
                Some('u') => {
                    let hex: String = chars.by_ref().take(4).collect();
                    let decoded = u32::from_str_radix(&hex, 16).ok().and_then(char::from_u32);
                    value.push(decoded.unwrap_or('\u{fffd}'));
                }
                Some(other) => value.push(other),
                None => break,
            },
            other => value.push(other),
        }
    }
    Some(value)
}
