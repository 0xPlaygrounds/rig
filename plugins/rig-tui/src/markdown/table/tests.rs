use ratatui::style::Modifier;
use unicode_width::UnicodeWidthStr;

use crate::Theme;
use crate::markdown::render;

/// `text` rendered and drawn, colored.
fn rendered(text: &str, theme: &Theme, width: Option<usize>) -> Vec<ratatui::text::Line<'static>> {
    let lines = render(text, theme, width);
    lines.iter().map(crate::LookLine::colored).collect()
}

fn text(markdown: &str, width: usize) -> Vec<String> {
    let lines = rendered(markdown, &Theme::default(), Some(width));
    lines
        .iter()
        .map(|line| line.to_string().trim_end().to_owned())
        .collect()
}

const WIDE: &str = "\
| Tool | What it did |
| --- | --- |
| read | Read the first five lines of the workspace manifest, which name the package and its edition |
| shell | Ran ls on a path that does not exist and failed with exit code 2 |
";

#[test]
fn a_table_that_fits_is_drawn_at_its_widest_cells_with_a_bold_header() {
    let markdown = "| a | bb |\n| - | - |\n| ccc | d |\n";
    assert_eq!(text(markdown, 80), ["a     bb", "ccc   d"]);
    let lines = rendered(markdown, &Theme::default(), Some(80));
    let header = lines
        .first()
        .and_then(|line| line.spans.iter().find(|span| span.content == "a"));
    let bold = Modifier::BOLD | Modifier::UNDERLINED;
    assert!(header.is_some_and(|span| span.style.add_modifier.contains(bold)));
}

#[test]
fn a_wide_table_wraps_its_cells_in_columns_that_fit() {
    let lines = text(WIDE, 50);
    assert!(lines.iter().all(|line| line.width() <= 50), "{lines:#?}");
    let words = |text: &str| {
        text.split_whitespace()
            .map(str::to_owned)
            .collect::<Vec<_>>()
    };
    let drawn: Vec<String> = lines.iter().flat_map(|line| words(line)).collect();
    assert!(
        words(WIDE)
            .iter()
            .filter(|word| word.chars().all(char::is_alphanumeric))
            .all(|word| drawn.contains(word)),
        "{lines:#?}"
    );
    // Each row's first line starts with its tool, the second column beside it.
    assert!(
        lines
            .iter()
            .any(|line| line.starts_with("read") && line.contains("Read the")),
        "{lines:#?}"
    );
    // Rows that wrapped are apart.
    assert!(lines.iter().any(String::is_empty), "{lines:#?}");
}

#[test]
fn a_table_too_narrow_for_its_columns_is_drawn_as_records() {
    let lines = text(WIDE, 16);
    assert!(lines.iter().all(|line| line.width() <= 16), "{lines:#?}");
    assert_eq!(
        lines.first().map(String::as_str),
        Some("Tool"),
        "{lines:#?}"
    );
    assert_eq!(
        lines.get(1).map(String::as_str),
        Some("  read"),
        "{lines:#?}"
    );
    assert!(lines.iter().any(|line| line == "What it did"), "{lines:#?}");
}

#[test]
fn a_br_breaks_a_cell_and_a_line() {
    let lines = text("| a | b |\n| - | - |\n| one<br>two | x |\n", 80);
    assert_eq!(lines, ["a     b", "one   x", "two"]);
    assert_eq!(text("first<br/>second", 80), ["first", "second"]);
}
