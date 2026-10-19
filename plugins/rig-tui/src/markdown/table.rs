//! A markdown table laid out to a width: its columns at their widest cell
//! when they fit, else shrunk fairly with each cell wrapped in its column,
//! and when even that leaves a column too narrow to read, each row as a
//! record of `header  value` lines.

use ratatui::style::{Modifier, Stylize};
use ratatui::text::{Line, Span};
use unicode_width::UnicodeWidthStr;

use crate::wrap::wrap;

/// A cell: its lines (a `<br>` starts another), each its spans.
pub(super) type Cell = Vec<Vec<Span<'static>>>;

/// A table being written: its rows of cells, the header's first.
#[derive(Default)]
pub(super) struct Table {
    pub(super) rows: Vec<Vec<Cell>>,
    pub(super) header_rows: usize,
}

/// Between two columns.
const GAP: &str = "   ";

/// A column is not made narrower than its longest word, up to this, nor
/// than this many cells, unless its widest cell is.
const LONGEST_WORD: usize = 30;
const NARROWEST: usize = 6;

/// `table`'s lines at `width` (any width with `None`), the first after
/// `first` and the others after `rest`, the indents of the block it is in.
pub(super) fn lines(
    table: Table,
    (first, rest): (Vec<Span<'static>>, Vec<Span<'static>>),
    width: Option<usize>,
) -> Vec<Line<'static>> {
    let Table { rows, header_rows } = table;
    let columns = rows.iter().map(Vec::len).max().unwrap_or(0);
    let natural: Vec<usize> = (0..columns)
        .map(|at| column(&rows, at).map(cell_width).max().unwrap_or(0))
        .collect();
    let indent = Line::from(first.clone()).width();
    let gaps = GAP.len() * columns.saturating_sub(1);
    let room = width.map(|width| width.saturating_sub(indent + gaps));
    let fitted = match room {
        None => Some(natural.clone()),
        Some(room) => fit(&rows, &natural, room),
    };
    let mut prefixes = std::iter::once(first).chain(std::iter::repeat(rest));
    match fitted {
        Some(widths) => grid(&rows, header_rows, &widths, &mut prefixes),
        None if rows.len() > header_rows => {
            let room = width.unwrap_or(0).saturating_sub(indent);
            records(&rows, header_rows, room, &mut prefixes)
        }
        // A header alone has no records: its cells at their own width.
        None => grid(&rows, header_rows, &natural, &mut prefixes),
    }
}

/// The cells of column `at`, the header's too.
fn column(rows: &[Vec<Cell>], at: usize) -> impl Iterator<Item = &Cell> {
    rows.iter().filter_map(move |row| row.get(at))
}

/// The widest of `cell`'s lines.
fn cell_width(cell: &Cell) -> usize {
    let line = |spans: &Vec<Span<'static>>| spans.iter().map(|span| span.content.width()).sum();
    cell.iter().map(line).max().unwrap_or(0)
}

/// The widest word of `cell`, at most [`LONGEST_WORD`].
fn longest_word(cell: &Cell) -> usize {
    let words = cell
        .iter()
        .flatten()
        .flat_map(|span| span.content.split_whitespace());
    words.map(str::width).max().unwrap_or(0).min(LONGEST_WORD)
}

/// The columns' widths in `room` cells: their `natural` widths when those
/// fit, else each its least plus a share of the rest by how much more it
/// wants; `None` when the least do not fit.
fn fit(rows: &[Vec<Cell>], natural: &[usize], room: usize) -> Option<Vec<usize>> {
    if natural.iter().sum::<usize>() <= room {
        return Some(natural.to_vec());
    }
    let least: Vec<usize> = (0..natural.len())
        .map(|at| {
            let word = column(rows, at).map(longest_word).max().unwrap_or(0);
            let wide = natural.get(at).copied().unwrap_or(0);
            word.max(NARROWEST).min(wide)
        })
        .collect();
    let spare = room.checked_sub(least.iter().sum())?;
    let wants: Vec<usize> = natural.iter().zip(&least).map(|(n, l)| n - l).collect();
    let wanted: usize = wants.iter().sum();
    let mut widths: Vec<usize> = least
        .iter()
        .zip(&wants)
        .map(|(least, wants)| least + (wants * spare).checked_div(wanted).unwrap_or(0))
        .collect();
    // What the shares rounded down goes a cell at a time to those wanting.
    let mut left = room.saturating_sub(widths.iter().sum());
    while left > 0 {
        let mut grew = false;
        for (width, natural) in widths.iter_mut().zip(natural) {
            if left > 0 && *width < *natural {
                *width += 1;
                left -= 1;
                grew = true;
            }
        }
        if !grew {
            break;
        }
    }
    Some(widths)
}

/// `cell`'s lines wrapped to `width`.
fn wrapped(cell: Option<&Cell>, width: usize) -> Vec<Line<'static>> {
    let lines = cell.into_iter().flatten();
    let rows = lines.flat_map(|spans| wrap(&Line::from(spans.clone()), width));
    rows.map(|(row, _)| row).collect()
}

/// The rows in columns of `widths`, the header's bold and underlined, a
/// blank line between the body's rows when any of them wrapped.
fn grid(
    rows: &[Vec<Cell>],
    header_rows: usize,
    widths: &[usize],
    prefixes: &mut impl Iterator<Item = Vec<Span<'static>>>,
) -> Vec<Line<'static>> {
    let laid: Vec<Vec<Vec<Line<'static>>>> = rows
        .iter()
        .map(|row| {
            let cells = widths.iter().enumerate();
            cells
                .map(|(at, width)| wrapped(row.get(at), *width))
                .collect()
        })
        .collect();
    let tall = laid
        .iter()
        .skip(header_rows)
        .any(|row| row.iter().any(|cell| cell.len() > 1));
    let mut lines = Vec::new();
    for (index, row) in laid.into_iter().enumerate() {
        let header = index < header_rows;
        if tall && index > header_rows {
            lines.push(Line::default());
        }
        let height = row.iter().map(Vec::len).max().unwrap_or(0).max(1);
        for at in 0..height {
            let mut spans = prefixes.next().unwrap_or_default();
            for (column, (cell, width)) in row.iter().zip(widths).enumerate() {
                if column > 0 {
                    spans.push(Span::from(GAP));
                }
                let line = cell.get(at);
                for span in line.into_iter().flat_map(|line| line.spans.iter()) {
                    let mut span = span.clone();
                    if header {
                        span.style = span
                            .style
                            .add_modifier(Modifier::BOLD | Modifier::UNDERLINED);
                    }
                    spans.push(span);
                }
                if column + 1 < widths.len() {
                    let used = line.map_or(0, Line::width);
                    spans.push(Span::from(" ".repeat(width.saturating_sub(used))));
                }
            }
            lines.push(Line::from(spans));
        }
    }
    lines
}

/// The body's rows as records in `room` cells, a blank line between
/// them: each cell after its column's header in bold, or under it when
/// the two do not fit side by side.
fn records(
    rows: &[Vec<Cell>],
    header_rows: usize,
    room: usize,
    prefixes: &mut impl Iterator<Item = Vec<Span<'static>>>,
) -> Vec<Line<'static>> {
    let labels: Vec<String> = rows
        .first()
        .filter(|_| header_rows > 0)
        .map(|header| {
            let text = |cell: &Cell| {
                let lines = cell.iter().map(|spans| {
                    spans
                        .iter()
                        .map(|span| span.content.as_ref())
                        .collect::<String>()
                });
                lines.collect::<Vec<_>>().join(" ")
            };
            header.iter().map(text).collect()
        })
        .unwrap_or_default();
    let label_width = labels.iter().map(|label| label.width()).max().unwrap_or(0);
    let beside = label_width + 2 + 2 * NARROWEST <= room;
    let mut lines = Vec::new();
    for (index, row) in rows.iter().skip(header_rows).enumerate() {
        if index > 0 {
            lines.push(Line::default());
        }
        for (at, cell) in row.iter().enumerate() {
            let label = labels.get(at).cloned().unwrap_or_default();
            let mut spans = prefixes.next().unwrap_or_default();
            if beside {
                let value = wrapped(Some(cell), room - label_width - 2);
                let pad = " ".repeat(label_width - label.width() + 2);
                spans.push(Span::from(label).bold());
                spans.push(Span::from(pad));
                let mut value = value.into_iter();
                spans.extend(value.next().map(|line| line.spans).unwrap_or_default());
                lines.push(Line::from(spans));
                for line in value {
                    let mut spans = prefixes.next().unwrap_or_default();
                    spans.push(Span::from(" ".repeat(label_width + 2)));
                    spans.extend(line.spans);
                    lines.push(Line::from(spans));
                }
            } else {
                spans.push(Span::from(label).bold());
                lines.push(Line::from(spans));
                for line in wrapped(Some(cell), room.saturating_sub(2).max(1)) {
                    let mut spans = prefixes.next().unwrap_or_default();
                    spans.push(Span::from("  "));
                    spans.extend(line.spans);
                    lines.push(Line::from(spans));
                }
            }
        }
    }
    lines
}

#[cfg(test)]
mod tests;
