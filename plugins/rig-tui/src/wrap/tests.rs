use ratatui::style::Style;
use ratatui::text::Line;

use super::{join, words, wrap};

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
        ratatui::text::Span::styled(ch.to_string(), style)
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
