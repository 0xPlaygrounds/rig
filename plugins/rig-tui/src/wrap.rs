//! Word wrapping of styled lines to a width, as the transcript wraps its
//! rows: at word boundaries, a list item's continuation rows under its
//! text. A plugin drawing its own lines wraps them with [`wrap_all`] to
//! look the same.

use ratatui::style::Style;
use ratatui::text::{Line, Span};
use unicode_width::UnicodeWidthStr;

use crate::RowWrap;

/// Markers after which a wrapped line's continuation rows are indented, so
/// a list item or a tool result keeps its left edge.
const HANGING_MARKERS: [&str; 5] = ["•", "-", "*", "›", ">"];

/// `line` wrapped at word boundaries into rows at most `width` columns
/// wide, each with how it continues the row before it (`None` for the
/// first). A word wider than a row is broken. Continuation rows are
/// indented like the text after the line's leading spaces and list marker.
/// Every row keeps the line's style, so a background fills each.
pub fn wrap(line: &Line<'_>, width: usize) -> Vec<(Line<'static>, Option<RowWrap>)> {
    let width = width.max(1);
    let cells: Vec<(&str, Style)> = line
        .styled_graphemes(Style::default())
        .map(|grapheme| (grapheme.symbol, grapheme.style))
        .collect();
    let total: usize = cells.iter().map(|(symbol, _)| symbol.width()).sum();
    if total <= width {
        return vec![(join(&cells).style(line.style), None)];
    }
    let indent = hanging_indent(&cells).min(width / 2);
    let mut rows = Vec::new();
    let mut row: Vec<(&str, Style)> = Vec::new();
    let mut row_width = 0;
    // How the row being filled continues the one before it.
    let mut joint: Option<RowWrap> = None;
    for word in words(&cells) {
        let word_width: usize = word.iter().map(|(symbol, _)| symbol.width()).sum();
        let blank = word.iter().all(|(symbol, _)| symbol.trim().is_empty());
        if row_width + word_width > width && row_width > indent {
            // Spaces at a break are dropped rather than carried over, but
            // not the indent the row starts with.
            let kept = if joint.is_some() { indent } else { 0 };
            let mut gap = Vec::new();
            while row.len() > kept
                && let Some(&(symbol, _)) =
                    row.last().filter(|(symbol, _)| symbol.trim().is_empty())
            {
                gap.push(symbol);
                row.pop();
            }
            gap.reverse();
            rows.push((join(&row), joint.take()));
            row.clear();
            row.extend(std::iter::repeat_n((" ", Style::default()), indent));
            row_width = indent;
            if blank {
                gap.extend(word.iter().map(|(symbol, _)| *symbol));
            }
            joint = Some(RowWrap {
                gap: gap.concat(),
                indent,
            });
            if blank {
                continue;
            }
        }
        for &(symbol, style) in word {
            let cell = symbol.width();
            if row_width + cell > width && row_width > indent {
                rows.push((join(&row), joint.take()));
                row.clear();
                row.extend(std::iter::repeat_n((" ", Style::default()), indent));
                row_width = indent;
                joint = Some(RowWrap {
                    gap: String::new(),
                    indent,
                });
            }
            row.push((symbol, style));
            row_width += cell;
        }
    }
    if !row.is_empty() || rows.is_empty() {
        rows.push((join(&row), joint));
    }
    let styled = rows
        .into_iter()
        .map(|(row, joint)| (row.style(line.style), joint));
    styled.collect()
}

/// Every line of `lines` wrapped to `width`.
pub fn wrap_all(lines: &[Line<'_>], width: usize) -> Vec<Line<'static>> {
    let rows = lines.iter().flat_map(|line| wrap(line, width));
    rows.map(|(row, _)| row).collect()
}

/// The cells split into words, each with the spaces that follow it.
fn words<'a, 'b>(cells: &'b [(&'a str, Style)]) -> impl Iterator<Item = &'b [(&'a str, Style)]> {
    let space = |symbol: &str| symbol.trim().is_empty();
    cells.chunk_by(move |(before, _), (after, _)| !space(before) || space(after))
}

/// The width of the leading spaces, plus a list marker and its space.
fn hanging_indent(cells: &[(&str, Style)]) -> usize {
    let spaces = cells
        .iter()
        .take_while(|(symbol, _)| *symbol == " ")
        .count();
    let rest = cells.get(spaces..).unwrap_or_default();
    let marker: usize = match rest {
        [(first, _), (" ", _), ..] if HANGING_MARKERS.contains(first) => first.width() + 1,
        _ => numbered_marker(rest),
    };
    spaces + marker
}

/// The width of a leading `12. ` or `3) `, else 0.
fn numbered_marker(cells: &[(&str, Style)]) -> usize {
    let digits = cells
        .iter()
        .take_while(|(symbol, _)| symbol.len() == 1 && symbol.bytes().all(|b| b.is_ascii_digit()))
        .count();
    match cells.get(digits..digits + 2) {
        Some([(dot, _), (" ", _)]) if digits > 0 && (*dot == "." || *dot == ")") => digits + 2,
        _ => 0,
    }
}

/// One row from its cells, neighbouring cells of one style in one span.
fn join(cells: &[(&str, Style)]) -> Line<'static> {
    let spans: Vec<Span<'static>> = cells
        .chunk_by(|(_, before), (_, after)| before == after)
        .map(|run| {
            let style = run.first().map(|(_, style)| *style).unwrap_or_default();
            Span::styled(
                run.iter().map(|(symbol, _)| *symbol).collect::<String>(),
                style,
            )
        })
        .collect();
    Line::from(spans)
}

#[cfg(test)]
mod tests;
