//! Markdown in the model's answers drawn as styled lines, in the theme's
//! tints, by [`rig_markdown::markdown`]. Fenced code is colored by its
//! language's grammar, when it has one.

use std::collections::{HashMap, HashSet};

use ratatui::text::{Line, Span};
use rig_markdown::markdown::{Highlight, Kind, Palette};

use crate::LookLine;
use crate::Theme;
use crate::highlight::{Code, Language};
use crate::renderers::CodeLine;

/// `text` as markdown in `theme`'s tints, in lines not yet wrapped, its
/// tables fit to `width` (at their widest cells with `None`).
pub fn render(text: &str, theme: &Theme, width: Option<usize>) -> Vec<LookLine> {
    let palette = Palette {
        code: theme.code,
        note: theme.note,
        tip: theme.call_done,
        important: theme.important,
        warning: theme.call_running,
        caution: theme.call_failed,
        plain: theme.plain,
    };
    let lines = rig_markdown::markdown::render(text, &palette, width, &Grammars);
    lines
        .into_iter()
        .map(|(line, kind)| {
            let code = match kind {
                Kind::Code(mark) => mark,
                Kind::Text => None,
            };
            LookLine { line, code }
        })
        .collect()
}

pub(crate) use rig_markdown::markdown::has_table;

/// Fenced code in the colors of its language's grammar.
struct Grammars;

impl Highlight for Grammars {
    type Mark = CodeLine;

    fn block(
        &self,
        code: &str,
        language: &str,
        prefix: &[Span<'static>],
    ) -> Option<Vec<(Line<'static>, Option<CodeLine>)>> {
        let colored = Some(Code::new(code, Language::Named(language))).filter(Code::has_grammar)?;
        let lines = (0..colored.len()).map(|index| {
            let look = colored.line(index, prefix.to_vec());
            (look.line, look.code)
        });
        Some(lines.collect())
    }
}

/// The private-use characters [`render_marked`] may stand marks in with.
fn stand_ins() -> impl Iterator<Item = char> {
    let planes = [0xE000..=0xF8FF, 0xF_0000..=0xF_FFFD, 0x10_0000..=0x10_FFFD];
    planes.into_iter().flatten().filter_map(char::from_u32)
}

/// `text` as markdown at `width`, as [`render`] draws it, with each of `marks` drawn
/// before its byte: past the end at the end, inside a character before
/// it. Each mark stands in the text as a private-use character the text
/// neither holds nor names while it is rendered, so markdown on both sides
/// of it is kept. A text that leaves too few of them free is drawn plain.
pub fn render_marked(
    text: &str,
    marks: &[(usize, Span<'static>)],
    theme: &Theme,
    width: Option<usize>,
) -> Vec<LookLine> {
    if marks.is_empty() {
        return render(text, theme, width);
    }
    let held: HashSet<char> = text.chars().chain(referenced(text)).collect();
    let free = stand_ins().filter(|ch| !held.contains(ch));
    let mut placed: Vec<(usize, char, &Span<'static>)> = free
        .zip(marks)
        .map(|(stand_in, (at, mark))| (text.floor_char_boundary(*at), stand_in, mark))
        .collect();
    if placed.len() < marks.len() {
        return plain_marked(text, marks);
    }
    placed.sort_by_key(|(at, ..)| *at);
    let mut marked = String::with_capacity(text.len() + 4 * placed.len());
    let mut from = 0;
    for (at, stand_in, _) in &placed {
        marked.push_str(text.get(from..*at).unwrap_or_default());
        marked.push(*stand_in);
        from = *at;
    }
    marked.push_str(text.get(from..).unwrap_or_default());
    let stood_for: HashMap<char, &Span<'static>> = placed
        .into_iter()
        .map(|(_, stand_in, mark)| (stand_in, mark))
        .collect();
    let mark_of = |ch: char| stood_for.get(&ch).copied();
    let swap = |span: Span<'static>| -> Vec<Span<'static>> {
        if !span.content.chars().any(|ch| mark_of(ch).is_some()) {
            return vec![span];
        }
        let mut spans = Vec::new();
        let mut run = String::new();
        for ch in span.content.chars() {
            match mark_of(ch) {
                Some(mark) => {
                    if !run.is_empty() {
                        spans.push(Span::styled(std::mem::take(&mut run), span.style));
                    }
                    spans.push(mark.clone());
                }
                None => run.push(ch),
            }
        }
        if !run.is_empty() {
            spans.push(Span::styled(run, span.style));
        }
        spans
    };
    let marked_in = |line: &Line<'_>| {
        let mut chars = line.spans.iter().flat_map(|span| span.content.chars());
        chars.any(|ch| mark_of(ch).is_some())
    };
    render(&marked, theme, width)
        .into_iter()
        .map(|look| {
            // A mark in a line of code is swapped into its colored spans.
            let (line, code) = match look.code {
                Some(_) if marked_in(&look.line) => (look.colored(), None),
                code => (look.line, code),
            };
            let spans = line.spans.into_iter().flat_map(swap).collect::<Vec<_>>();
            let line = Line { spans, ..line };
            LookLine { line, code }
        })
        .collect()
}

/// Every character a numeric character reference in `text` names, such as
/// `&#xE000;`, wherever it stands. A stand-in never joins text, so these and
/// the text's own are all the private-use characters markdown can draw
/// from it once marked, since no named reference names one.
fn referenced(text: &str) -> impl Iterator<Item = char> + '_ {
    text.split("&#").skip(1).filter_map(|after| {
        let (digits, _) = after.split_once(';')?;
        let code = match digits.strip_prefix(['x', 'X']) {
            Some(hex) => u32::from_str_radix(hex, 16),
            None => digits.parse(),
        };
        char::from_u32(code.ok()?)
    })
}

/// `text` as plain lines with each of `marks` before its byte.
fn plain_marked(text: &str, marks: &[(usize, Span<'static>)]) -> Vec<LookLine> {
    let mut placed: Vec<(usize, Option<&Span<'static>>)> = marks
        .iter()
        .map(|(at, mark)| (text.floor_char_boundary(*at), Some(mark)))
        .collect();
    placed.sort_by_key(|(at, _)| *at);
    placed.push((text.len(), None));
    let mut lines = vec![Line::default()];
    let mut from = 0;
    for (at, mark) in placed {
        let piece = text.get(from..at).unwrap_or_default();
        for (index, row) in piece.split('\n').enumerate() {
            if index > 0 {
                lines.push(Line::default());
            }
            if let Some(line) = lines.last_mut()
                && !row.is_empty()
            {
                line.push_span(Span::from(row.to_owned()));
            }
        }
        if let (Some(line), Some(mark)) = (lines.last_mut(), mark) {
            line.push_span(mark.clone());
        }
        from = at;
    }
    lines.into_iter().map(LookLine::from).collect()
}

#[cfg(test)]
mod tests;
