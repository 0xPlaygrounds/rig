//! A unified diff a tool returned, drawn in the transcript: each line
//! with its numbers in the old and the new file, in the colors of the
//! file's language ([`highlight`](crate::highlight)), removed lines on red
//! and added ones on green ([`Theme`]), and the words a changed line
//! changed on a stronger shade of each.

use std::collections::HashMap;
use std::ops::Range;

use ratatui::style::{Style, Stylize};
use ratatui::text::{Line, Span};
use similar::{ChangeTag, TextDiff};

use crate::Theme;
use crate::highlight::{Language, highlight};

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
    let mut words = changed_words(&lines);
    let mut code = highlighted(&lines);
    let drawn = lines.iter().enumerate().map(|(at, (kind, text))| {
        let number = |at: Option<usize>| at.map_or(" ".repeat(width), |at| format!("{at:>width$}"));
        let gutter =
            |old, new, sign| Span::from(format!("    {} {} {sign} ", number(old), number(new)));
        let spans = code
            .remove(&at)
            .unwrap_or_else(|| vec![Span::from(text.clone())]);
        let mut marked =
            |shade| overlay(spans.clone(), &words.remove(&at).unwrap_or_default(), shade);
        match *kind {
            Kind::Other => Line::from(format!("    {text}")).dim(),
            Kind::File => Line::from(format!("    {text}")).bold(),
            Kind::Hunk => Line::from(format!("    {text}")).dim(),
            Kind::Same(old, new) => {
                Line::from([vec![gutter(Some(old), Some(new), ' ').dim()], spans].concat())
            }
            Kind::Removed(old) => {
                let shade = theme.changed_word(theme.removed_word);
                let gutter = gutter(Some(old), None, '-').red();
                Line::from([vec![gutter], marked(shade)].concat()).bg(theme.removed)
            }
            Kind::Added(new) => {
                let shade = theme.changed_word(theme.added_word);
                let gutter = gutter(None, Some(new), '+').green();
                Line::from([vec![gutter], marked(shade)].concat()).bg(theme.added)
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

/// Each code line's spans in the colors of the file's language, by the
/// line's index: the hunk's old side (same and removed lines) and its new
/// side (same and added) highlighted apart. None when the diff names no
/// file, or one whose language has no grammar.
fn highlighted(lines: &[(Kind, String)]) -> HashMap<usize, Vec<Span<'static>>> {
    let mut spans = HashMap::new();
    let files = lines.iter().filter(|(kind, _)| *kind == Kind::File);
    let mut paths = files.filter_map(|(_, line)| line.get(4..)).map(str::trim);
    let Some(path) = paths.rfind(|path| *path != "/dev/null") else {
        return spans;
    };
    let hunks = lines
        .iter()
        .enumerate()
        .collect::<Vec<_>>()
        .chunk_by(|_, (_, (kind, _))| *kind != Kind::Hunk)
        .map(<[_]>::to_vec)
        .collect::<Vec<_>>();
    let old = |kind: &Kind| matches!(kind, Kind::Same(..) | Kind::Removed(_));
    let new = |kind: &Kind| matches!(kind, Kind::Same(..) | Kind::Added(_));
    for hunk in &hunks {
        for side in [&old as &dyn Fn(&Kind) -> bool, &new] {
            let side: Vec<(usize, &String)> = hunk
                .iter()
                .filter(|(_, (kind, _))| side(kind))
                .map(|(at, (_, text))| (*at, text))
                .collect();
            let code: Vec<&str> = side.iter().map(|(_, text)| text.as_str()).collect();
            let Some(drawn) = highlight(&code.join("\n"), Language::File(path)) else {
                return spans;
            };
            for ((at, _), line) in side.iter().zip(drawn) {
                spans.entry(*at).or_insert(line);
            }
        }
    }
    spans
}

/// The bytes of each removed line followed by an added one, and of that
/// added line, that changed between them, by the line's index. Removed
/// and added lines pair up in order; a pair with too little in common is
/// left unmarked.
fn changed_words(lines: &[(Kind, String)]) -> HashMap<usize, Vec<Range<usize>>> {
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
            let (mut was_at, mut is_at) = (0, 0);
            for change in diff.iter_all_changes() {
                let length = change.value().len();
                match change.tag() {
                    ChangeTag::Equal => {
                        was_at += length;
                        is_at += length;
                    }
                    ChangeTag::Delete => {
                        was.push(was_at..was_at + length);
                        was_at += length;
                    }
                    ChangeTag::Insert => {
                        is.push(is_at..is_at + length);
                        is_at += length;
                    }
                }
            }
            words.insert(old, was);
            words.insert(new, is);
        }
        at = added_end.max(at + 1);
    }
    words
}

/// `spans` with `shade` laid over the bytes `ranges` of their joined
/// text, each span split where a range starts or ends inside it.
fn overlay(spans: Vec<Span<'static>>, ranges: &[Range<usize>], shade: Style) -> Vec<Span<'static>> {
    if ranges.is_empty() {
        return spans;
    }
    let mut laid = Vec::new();
    let mut start = 0;
    for span in spans {
        let text = span.content.as_ref();
        let end = start + text.len();
        let mut cuts = vec![0, text.len()];
        let inside = ranges.iter().flat_map(|range| [range.start, range.end]);
        cuts.extend(
            inside
                .filter(|at| start < *at && *at < end)
                .map(|at| at - start),
        );
        cuts.sort_unstable();
        cuts.dedup();
        for pair in cuts.windows(2) {
            let [from, to] = pair else { continue };
            let Some(piece) = text.get(*from..*to) else {
                continue;
            };
            let at = start + from;
            let marked = ranges.iter().any(|range| range.contains(&at));
            let style = if marked {
                span.style.patch(shade)
            } else {
                span.style
            };
            laid.push(Span::styled(piece.to_owned(), style));
        }
        start = end;
    }
    laid
}

#[cfg(test)]
mod tests;
