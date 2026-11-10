use std::ops::Range;

use ratatui_core::style::Style;
use ratatui_core::text::Line;
use unicode_width::UnicodeWidthStr;

use super::{RowWrap, join, words, wrap, wrap_typed};

#[test]
fn a_word_keeps_the_spaces_after_it() {
    let dim = Style::new().dim();
    let cells: Vec<(&str, Style)> = "  ab  cd e"
        .split_inclusive(|_| true)
        .map(|symbol| (symbol, dim))
        .collect();
    let split: Vec<String> = words(&cells)
        .map(|word| word.iter().map(|(symbol, _)| *symbol).collect())
        .collect();
    assert_eq!(split, ["  ", "ab  ", "cd ", "e"]);
    assert_eq!(words(&[]).count(), 0);
}

#[test]
fn neighbouring_cells_of_one_style_are_one_span() {
    let (dim, bold) = (Style::new().dim(), Style::new().bold());
    let line = join(&[("a", dim), ("b", dim), ("c", bold), ("d", dim)]);
    let spans: Vec<(&str, Style)> = line
        .spans
        .iter()
        .map(|span| (span.content.as_ref(), span.style))
        .collect();
    assert_eq!(spans, [("ab", dim), ("c", bold), ("d", dim)]);
    assert!(join(&[]).spans.is_empty());
}

#[test]
fn a_list_item_wraps_under_its_text() {
    let rows: Vec<String> = wrap(&Line::from("• one two three"), 10)
        .iter()
        .map(|(row, _)| row.to_string())
        .collect();
    assert_eq!(rows, ["• one two", "  three"]);
}

#[test]
fn a_line_wraps_by_its_text_alone_however_its_spans_split_it() {
    // A ZWJ sequence, a combining accent, a variation selector and a
    // keycap, each split across spans in the line cut into characters.
    let text = "let s = \"👩\u{200d}💻 e\u{301}  ✌\u{fe0f} 1\u{fe0f}\u{20e3}\" + words that go on";
    let (dim, bold) = (Style::new().dim(), Style::new().bold());
    let pieces = text.chars().enumerate().map(|(at, ch)| {
        let style = if at % 2 == 0 { dim } else { bold };
        ratatui_core::text::Span::styled(ch.to_string(), style)
    });
    let split = Line::from(pieces.collect::<Vec<_>>());
    let rows = |line: &Line<'_>, width| {
        let rows = wrap(line, width).into_iter();
        rows.map(|(row, joint)| (row.to_string(), joint))
            .collect::<Vec<_>>()
    };
    for width in 1..=text.len() {
        assert_eq!(
            rows(&Line::from(text), width),
            rows(&split, width),
            "{width}"
        );
    }
}

/// The bytes of `line` each of its `rows` shows: past the gap the row's
/// break dropped, its text after its indent.
fn sources(line: &str, rows: &[(Line<'static>, Option<RowWrap>)]) -> Vec<Range<usize>> {
    let mut end = 0;
    let ranges = rows.iter().map(|(row, joint)| {
        let (gap, indent) = joint
            .as_ref()
            .map_or((0, 0), |joint| (joint.gap.len(), joint.indent));
        let bytes: usize = row.spans.iter().map(|span| span.content.len()).sum();
        let start = (end + gap).min(line.len());
        end = start + bytes.saturating_sub(indent);
        start..end
    });
    ranges.collect()
}

#[test]
fn no_word_that_fits_a_row_is_split() {
    let letters = ["a", "z", "é", "e\u{301}", "世", "👩\u{200d}💻", "✌\u{fe0f}"];
    let mut rng = fastrand::Rng::with_seed(39);
    for _ in 0..2000 {
        // Some lines start with spaces, and some put a column after their
        // first word, so the hanging indents are wrapped too.
        let mut line = " ".repeat(rng.usize(..4) * usize::from(rng.bool()));
        let mut words = Vec::new();
        for index in 0..rng.usize(1..30) {
            if index > 0 {
                let column = index == 1 && rng.bool();
                line.push_str(&" ".repeat(if column { rng.usize(2..5) } else { 1 }));
            }
            let word: String = (0..rng.usize(1..14))
                .filter_map(|_| letters.get(rng.usize(..letters.len())).copied())
                .collect();
            words.push(line.len()..line.len() + word.len());
            line.push_str(&word);
        }
        // From 4, so a row past an indent of at most half the width has
        // room for the widest grapheme.
        let width = rng.usize(4..40);
        let rows = wrap(&Line::from(line.as_str()), width);
        let ranges = sources(&line, &rows);
        let indent = rows.iter().filter_map(|(_, joint)| joint.as_ref());
        let indent = indent.map(|joint| joint.indent).max().unwrap_or(0);
        for ((row, joint), range) in rows.iter().zip(&ranges) {
            let row = row.to_string();
            let skip = joint.as_ref().map_or(0, |joint| joint.indent);
            assert!(row.width() <= width, "{row:?} of {line:?} at {width}");
            assert_eq!(
                row.get(skip..),
                line.get(range.clone()),
                "{line:?} at {width}"
            );
        }
        assert_eq!(
            ranges.last().map(|range| range.end),
            Some(line.len()),
            "{line:?} at {width}"
        );
        for word in words {
            let fits = line.get(word.clone()).unwrap_or_default().width() + indent <= width;
            let whole = ranges
                .iter()
                .any(|row| row.start <= word.start && word.end <= row.end);
            assert!(!fits || whole, "{word:?} of {line:?} at {width}: {rows:?}");
        }
    }
}

#[test]
fn a_column_after_two_spaces_wraps_under_its_text() {
    let rows: Vec<String> = wrap(&Line::from("     4  one two three"), 16)
        .iter()
        .map(|(row, _)| row.to_string())
        .collect();
    assert_eq!(rows, ["     4  one two", "        three"]);
    let typed: Vec<String> = wrap_typed(&Line::from("fix  the bug now"), 10)
        .iter()
        .map(|(row, _)| row.to_string())
        .collect();
    assert_eq!(typed, ["fix  the", "bug now"]);
    // A first row, which has no indent, breaks once it is full, whatever
    // the indent of the rows after it.
    let first = wrap(&Line::from("e👩\u{200d}💻  x"), 2);
    let first = first.first().map(|(row, _)| row.to_string());
    assert_eq!(first.as_deref(), Some("e"));
    let rows: Vec<String> = wrap(&Line::from("/help List the commands"), 12)
        .iter()
        .map(|(row, _)| row.to_string())
        .collect();
    assert_eq!(rows, ["/help List", "the commands"]);
}
