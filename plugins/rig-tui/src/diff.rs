//! A unified diff a tool returned, drawn in the transcript: each line
//! with its numbers in the old and the new file, in the colors of the
//! file's language once drawn ([`Code`]), removed lines on red
//! and added ones on green ([`Theme`]), and the words a changed line
//! changed on a stronger shade of each.

use std::collections::HashMap;
use std::ops::Range;

use ratatui::style::{Style, Stylize};
use ratatui::text::{Line, Span};
use rig_core::tool::FileChange;
use similar::{ChangeTag, TextDiff};

use crate::highlight::{Code, Language};
use crate::renderers::CodeLine;
use crate::{LookLine, Theme};

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
pub fn unified_lines(text: &str, theme: &Theme) -> Option<Vec<LookLine>> {
    if !text.lines().any(|line| header(line).is_some()) {
        return None;
    }
    let lines = classify(text);
    let files = lines.iter().filter(|(kind, _)| *kind == Kind::File);
    let mut paths = files.filter_map(|(_, line)| line.get(4..)).map(str::trim);
    let path = paths.rfind(|path| *path != "/dev/null");
    Some(draw(&lines, path, theme))
}

/// `change`'s hunks drawn as [`unified_lines`] draws them, in the colors
/// of its path's language, without the `---`/`+++` lines naming its file.
pub fn change_lines(change: &FileChange, theme: &Theme) -> Vec<LookLine> {
    let mut lines = classify(&change.unified());
    lines.retain(|(kind, _)| *kind != Kind::File);
    draw(&lines, Some(change.path.as_str()), theme)
}

/// The classified `lines` of a diff of the file at `path`, each drawn.
fn draw(lines: &[(Kind, String)], path: Option<&str>, theme: &Theme) -> Vec<LookLine> {
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
    let mut words = changed_words(lines);
    let mut code = path
        .map(|path| highlighted(lines, path))
        .unwrap_or_default();
    let drawn = lines.iter().enumerate().map(|(at, (kind, text))| {
        let number = |at: Option<usize>| at.map_or(" ".repeat(width), |at| format!("{at:>width$}"));
        let gutter =
            |old, new, sign| Span::from(format!("    {} {} {sign} ", number(old), number(new)));
        let spans = vec![Span::from(text.clone())];
        let mut marked =
            |shade| overlay(spans.clone(), &words.remove(&at).unwrap_or_default(), shade);
        let (line, gutter) = match *kind {
            Kind::Other => (Line::from(format!("    {text}")).dim(), None),
            Kind::File => (Line::from(format!("    {text}")).bold(), None),
            Kind::Hunk => (Line::from(format!("    {text}")).dim(), None),
            Kind::Same(old, new) => {
                let gutter = gutter(Some(old), Some(new), ' ').dim();
                (
                    Line::from([vec![gutter.clone()], spans.clone()].concat()),
                    Some(gutter),
                )
            }
            Kind::Removed(old) => {
                let shade = theme.changed_word(theme.removed_word);
                let gutter = gutter(Some(old), None, '-').red();
                let line = Line::from([vec![gutter.clone()], marked(shade)].concat());
                (line.bg(theme.removed), Some(gutter))
            }
            Kind::Added(new) => {
                let shade = theme.changed_word(theme.added_word);
                let gutter = gutter(None, Some(new), '+').green();
                let line = Line::from([vec![gutter.clone()], marked(shade)].concat());
                (line.bg(theme.added), Some(gutter))
            }
        };
        let code = gutter
            .zip(code.remove(&at))
            .map(|(gutter, (code, index))| CodeLine {
                code,
                index,
                at: gutter.content.len(),
            });
        LookLine { line, code }
    });
    drawn.collect()
}

/// Each line of the diff `text` with what it is, tabs made spaces and
/// the sign of a hunk's line taken off. A hunk holds only the lines that
/// [`fits`] it, so a removed line starting with `--` is not taken for a
/// file. It ends early at a line that doesn't fit, as after output cut off
/// by `head`, and that line is read as one outside any hunk.
fn classify(text: &str) -> Vec<(Kind, String)> {
    let (mut old, mut new) = (0, 0);
    let (mut old_left, mut new_left) = (0usize, 0usize);
    let step = |at: &mut usize| std::mem::replace(at, at.saturating_add(1));
    let raws: Vec<&str> = text.lines().collect();
    let mut in_blanks = false;
    let lines = raws.iter().enumerate().map(|(at, raw)| {
        let line = raw.replace('\t', "    ");
        let rest = raws.get(at + 1..).unwrap_or_default();
        // A blank line after one that fit fits while both sides have lines
        // left: [`fits`] looked past the whole run at its first.
        let fit = if in_blanks && raw.is_empty() {
            old_left > 0 && new_left > 0
        } else {
            fits(raw, rest, old_left, new_left)
        };
        in_blanks = fit && raw.is_empty();
        if !fit {
            (old_left, new_left) = (0, 0);
            return match header(&line) {
                Some(((old_start, old_count), (new_start, new_count))) => {
                    (old, new) = (old_start, new_start);
                    (old_left, new_left) = (old_count, new_count);
                    (Kind::Hunk, line)
                }
                None if line.starts_with("--- ") || line.starts_with("+++ ") => (Kind::File, line),
                None => (Kind::Other, line),
            };
        }
        let kind = match raw.chars().next() {
            Some('-') => {
                old_left -= 1;
                Kind::Removed(step(&mut old))
            }
            Some('+') => {
                new_left -= 1;
                Kind::Added(step(&mut new))
            }
            Some(' ') | None => {
                (old_left, new_left) = (old_left - 1, new_left - 1);
                Kind::Same(step(&mut old), step(&mut new))
            }
            _ => return (Kind::Other, line),
        };
        (kind, line.get(1..).unwrap_or_default().to_owned())
    });
    lines.collect()
}

/// Whether the line `raw`, with the lines `rest` after it, belongs to a
/// hunk with `old` and `new` lines left: its sign fits what is left. A
/// blank line is an empty context line, as `diff.suppressBlankEmpty`
/// writes one, only when its hunk ends with it or the blank lines after it,
/// or the next line that isn't blank fits what is left, so a blank line
/// after a hunk cut short is not.
fn fits(raw: &str, rest: &[&str], old: usize, new: usize) -> bool {
    match raw.chars().next() {
        Some('-') => old > 0,
        Some('+') => new > 0,
        Some(' ') => old > 0 && new > 0,
        None if old > 0 && new > 0 => {
            let blanks = rest.iter().take_while(|line| line.is_empty()).count();
            let taken = blanks.min(old - 1).min(new - 1);
            let (old, new) = (old - 1 - taken, new - 1 - taken);
            let next = rest.get(blanks).filter(|_| taken == blanks);
            old == 0 && new == 0 || next.is_some_and(|next| fits(next, &[], old, new))
        }
        // `\ No newline at end of file` can come between a hunk's removed
        // and added lines.
        _ => raw.starts_with("\\ ") && (old > 0 || new > 0),
    }
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

/// Each code line's code and its place in it, by the line's index: the
/// hunk's old side (same and removed lines) and its new side (same and
/// added) each a code in `path`'s language. Empty when the language has
/// no grammar.
fn highlighted(lines: &[(Kind, String)], path: &str) -> HashMap<usize, (Code, usize)> {
    let mut spans = HashMap::new();
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
            let code = Code::new(&code.join("\n"), Language::File(path));
            if !code.has_grammar() {
                return spans;
            }
            for (index, (at, _)) in side.iter().enumerate() {
                spans.entry(*at).or_insert_with(|| (code.clone(), index));
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
