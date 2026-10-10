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
        .map(ToString::to_string)
        .collect();
    assert_eq!(rows, ["• one two", "  three"]);
}
