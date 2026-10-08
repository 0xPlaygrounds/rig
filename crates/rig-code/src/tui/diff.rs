//! Diffs drawn in the transcript: a line diff of an edit's old and new text,
//! and the colouring of a unified diff a tool returned.

use ratatui::style::{Color, Style, Stylize};
use ratatui::text::{Line, Span};
use similar::{ChangeTag, TextDiff};

/// Lines of unchanged context kept around each change.
const CONTEXT: usize = 2;

fn added() -> Style {
    Style::new().fg(Color::Green)
}

fn removed() -> Style {
    Style::new().fg(Color::Red)
}

/// The changes from `old` to `new`, line by line with `CONTEXT` lines
/// around each, at most `limit` lines and then a count of the rest.
pub fn diff_lines(old: &str, new: &str, limit: usize) -> Vec<Line<'static>> {
    let diff = TextDiff::from_lines(old, new);
    let mut lines = Vec::new();
    for (index, group) in diff.grouped_ops(CONTEXT).iter().enumerate() {
        if index > 0 {
            lines.push(Line::from("    ⋯").dark_gray());
        }
        for op in group {
            for change in diff.iter_changes(op) {
                let (sign, style) = match change.tag() {
                    ChangeTag::Equal => (' ', Style::new().dim()),
                    ChangeTag::Delete => ('-', removed()),
                    ChangeTag::Insert => ('+', added()),
                };
                let number = change
                    .new_index()
                    .or(change.old_index())
                    .map_or(String::new(), |index| format!("{:>4}", index + 1));
                let text = change.value().trim_end_matches(['\n', '\r']);
                lines.push(Line::from(vec![
                    Span::from(format!("{number} ")).dark_gray(),
                    Span::styled(format!("{sign} {}", text.replace('\t', "    ")), style),
                ]));
            }
        }
    }
    cap(lines, limit)
}

/// `text` coloured as a unified diff when it holds a hunk header, else
/// `None`. At most `limit` lines.
pub fn unified_lines(text: &str, limit: usize) -> Option<Vec<Line<'static>>> {
    if !text.lines().any(|line| line.starts_with("@@")) {
        return None;
    }
    let lines = text
        .lines()
        .map(|line| {
            let line = line.replace('\t', "    ");
            if line.starts_with("+++") || line.starts_with("---") {
                Line::from(line).bold()
            } else if line.starts_with("@@") {
                Line::from(line).cyan()
            } else if line.starts_with('+') {
                Line::styled(line, added())
            } else if line.starts_with('-') {
                Line::styled(line, removed())
            } else {
                Line::from(line).dim()
            }
        })
        .collect();
    Some(cap(lines, limit))
}

/// The first `limit` of `lines`, and a line counting the rest.
fn cap(mut lines: Vec<Line<'static>>, limit: usize) -> Vec<Line<'static>> {
    if lines.len() > limit {
        let rest = lines.len() - limit;
        lines.truncate(limit);
        lines.push(Line::from(format!("    … {rest} more diff lines")).dim());
    }
    lines
}
