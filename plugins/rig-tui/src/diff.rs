//! A unified diff a tool returned, drawn in the transcript: each line
//! with its numbers in the old and the new file, removed lines on red and
//! added ones on green ([`Theme`]), and the words a changed line changed
//! on a stronger shade of each.

use std::collections::HashMap;

use ratatui::style::{Style, Stylize};
use ratatui::text::{Line, Span};
use similar::{ChangeTag, TextDiff};

use crate::Theme;

/// What a line of the diff is.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// Text around the diff, such as a tool's summary.
    Other,
    /// A `---` or `+++` line naming a file.
    File,
    /// A hunk's `@@` header.
    Hunk,
    /// A line the same in both files, with its numbers in each.
    Same(usize, usize),
    /// A removed line, with its number in the old file.
    Removed(usize),
    /// An added line, with its number in the new file.
    Added(usize),
}

/// `text` drawn as a unified diff when it holds a hunk header, else
/// `None`, in `theme`'s colors.
pub fn unified_lines(text: &str, theme: &Theme) -> Option<Vec<Line<'static>>> {
    if !text.lines().any(|line| header(line).is_some()) {
        return None;
    }
    let lines: Vec<(Kind, String)> = classify(text);
    let width = lines
        .iter()
        .map(|(kind, _)| match kind {
            Kind::Same(old, new) => (*old).max(*new),
            Kind::Removed(at) | Kind::Added(at) => *at,
            _ => 0,
        })
        .max()
        .unwrap_or_default()
        .to_string()
        .len();
    let mut words = changed_words(&lines, theme);
    let drawn = lines.iter().enumerate().map(|(at, (kind, text))| {
        let number = |at: Option<usize>| at.map_or(" ".repeat(width), |at| format!("{at:>width$}"));
        let gutter =
            |old, new, sign| Span::from(format!("    {} {} {sign} ", number(old), number(new)));
        let text_of = |words: Option<Vec<Span<'static>>>| {
            words.unwrap_or_else(|| vec![Span::from(text.clone())])
        };
        match *kind {
            Kind::Other => Line::from(format!("    {text}")).dim(),
            Kind::File => Line::from(format!("    {text}")).bold(),
            Kind::Hunk => Line::from(format!("    {text}")).dim(),
            Kind::Same(old, new) => {
                let mut spans = vec![gutter(Some(old), Some(new), ' ').dim()];
                spans.push(Span::from(text.clone()));
                Line::from(spans)
            }
            Kind::Removed(old) => {
                let mut spans = vec![gutter(Some(old), None, '-')];
                spans.extend(text_of(words.remove(&at)));
                Line::from(spans).style(Style::new().red().bg(theme.removed))
            }
            Kind::Added(new) => {
                let mut spans = vec![gutter(None, Some(new), '+')];
                spans.extend(text_of(words.remove(&at)));
                Line::from(spans).style(Style::new().green().bg(theme.added))
            }
        }
    });
    Some(drawn.collect())
}

/// Each line of the diff `text` with what it is, tabs made spaces and
/// the sign of a hunk's line taken off. A hunk's lines are counted by its
/// header, so a removed line starting with `--` is not taken for a file.
fn classify(text: &str) -> Vec<(Kind, String)> {
    let (mut old, mut new) = (0, 0);
    let (mut old_left, mut new_left) = (0usize, 0usize);
    text.lines()
        .map(|line| {
            let line = line.replace('\t', "    ");
            let body = || line.get(1..).unwrap_or_default().to_owned();
            if old_left + new_left > 0 {
                let kind = match line.chars().next() {
                    Some('-') => {
                        old_left = old_left.saturating_sub(1);
                        old += 1;
                        Kind::Removed(old - 1)
                    }
                    Some('+') => {
                        new_left = new_left.saturating_sub(1);
                        new += 1;
                        Kind::Added(new - 1)
                    }
                    Some('\\') => return (Kind::Other, line),
                    _ => {
                        old_left = old_left.saturating_sub(1);
                        new_left = new_left.saturating_sub(1);
                        (old, new) = (old + 1, new + 1);
                        Kind::Same(old - 1, new - 1)
                    }
                };
                return (kind, body());
            }
            match header(&line) {
                Some(((old_start, old_count), (new_start, new_count))) => {
                    (old, new) = (old_start, new_start);
                    (old_left, new_left) = (old_count, new_count);
                    (Kind::Hunk, line)
                }
                None if line.starts_with("--- ") || line.starts_with("+++ ") => (Kind::File, line),
                None => (Kind::Other, line),
            }
        })
        .collect()
}

/// A hunk header's start and count in the old file and in the new, as in
/// `@@ -12,3 +12,4 @@`; a count left out is 1.
fn header(line: &str) -> Option<((usize, usize), (usize, usize))> {
    let rest = line.strip_prefix("@@ -")?;
    let (ranges, _) = rest.split_once(" @@")?;
    let (old, new) = ranges.split_once(" +")?;
    let range = |range: &str| -> Option<(usize, usize)> {
        match range.split_once(',') {
            Some((start, count)) => Some((start.parse().ok()?, count.parse().ok()?)),
            None => Some((range.parse().ok()?, 1)),
        }
    };
    Some((range(old)?, range(new)?))
}

/// The text of each removed line followed by an added one, and of that
/// added line, as spans with the words that changed between them on the
/// stronger shade, by the line's index. Removed and added lines pair up
/// in order; a pair with no word in common is left whole.
fn changed_words(lines: &[(Kind, String)], theme: &Theme) -> HashMap<usize, Vec<Span<'static>>> {
    let mut words = HashMap::new();
    let removed = |at: &usize| matches!(lines.get(*at), Some((Kind::Removed(_), _)));
    let added = |at: &usize| matches!(lines.get(*at), Some((Kind::Added(_), _)));
    let mut at = 0;
    while at < lines.len() {
        if !removed(&at) {
            at += 1;
            continue;
        }
        let removed_end = (at..lines.len())
            .find(|at| !removed(at))
            .unwrap_or(lines.len());
        let added_end = (removed_end..lines.len())
            .find(|at| !added(at))
            .unwrap_or(lines.len());
        for (old, new) in (at..removed_end).zip(removed_end..added_end) {
            let (Some((_, before)), Some((_, after))) = (lines.get(old), lines.get(new)) else {
                continue;
            };
            let diff = TextDiff::from_words(before.as_str(), after.as_str());
            if diff.ratio() < 0.3 {
                continue;
            }
            let (mut was, mut is) = (Vec::new(), Vec::new());
            for change in diff.iter_all_changes() {
                let word = change.value().to_owned();
                match change.tag() {
                    ChangeTag::Equal => {
                        was.push(Span::from(word.clone()));
                        is.push(Span::from(word));
                    }
                    ChangeTag::Delete => was.push(Span::from(word).bg(theme.removed_word)),
                    ChangeTag::Insert => is.push(Span::from(word).bg(theme.added_word)),
                }
            }
            words.insert(old, was);
            words.insert(new, is);
        }
        at = added_end.max(at + 1);
    }
    words
}

#[cfg(test)]
mod tests;
