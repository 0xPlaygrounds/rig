//! Markdown in the model's answers drawn as styled lines, in the theme's
//! tints, by [`rig_markdown::markdown`]. Fenced code is colored by its
//! language's grammar, when it has one.

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

/// The first of the private-use characters [`render_marked`] stands
/// marks in with.
const SENTINEL: u32 = 0xE000;

/// The most marks one text takes: the private-use area's size.
const MOST_MARKS: usize = 0x1900;

/// `text` as markdown at `width`, as [`render`] draws it, with each of `marks` drawn
/// before its byte: past the end at the end, inside a character before
/// it. Each mark stands in the text as a private-use character while it
/// is rendered, so markdown on both sides of it is kept.
pub fn render_marked(
    text: &str,
    marks: &[(usize, Span<'static>)],
    theme: &Theme,
    width: Option<usize>,
) -> Vec<LookLine> {
    if marks.is_empty() {
        return render(text, theme, width);
    }
    let mut placed: Vec<(usize, usize)> = marks
        .iter()
        .take(MOST_MARKS)
        .enumerate()
        .map(|(index, (at, _))| {
            let mut at = (*at).min(text.len());
            while !text.is_char_boundary(at) {
                at = at.saturating_sub(1);
            }
            (at, index)
        })
        .collect();
    placed.sort_by_key(|(at, _)| *at);
    let mut marked = String::with_capacity(text.len() + 3 * placed.len());
    let mut from = 0;
    for (at, index) in &placed {
        marked.push_str(text.get(from..*at).unwrap_or_default());
        let code = SENTINEL.saturating_add(u32::try_from(*index).unwrap_or(u32::MAX));
        marked.extend(char::from_u32(code));
        from = *at;
    }
    marked.push_str(text.get(from..).unwrap_or_default());
    let mark_of = |ch: char| {
        let index = u32::from(ch).checked_sub(SENTINEL)?;
        marks
            .get(usize::try_from(index).ok()?)
            .map(|(_, span)| span)
    };
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

#[cfg(test)]
mod tests;
