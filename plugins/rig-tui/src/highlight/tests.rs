use super::*;

#[test]
fn code_is_drawn_in_the_terminals_palette_and_plain_text_is_not_highlighted() {
    let Some(lines) = highlight("fn main() {\n    let x = 1;\n}\n", Language::File("a.rs")) else {
        return;
    };
    assert_eq!(lines.len(), 3);
    let colors: Vec<Color> = lines
        .iter()
        .flatten()
        .filter_map(|span| span.style.fg)
        .collect();
    assert!(!colors.is_empty());
    assert!(
        colors
            .iter()
            .all(|color| matches!(color, Color::Indexed(_)))
    );
    let text: String = lines
        .first()
        .into_iter()
        .flatten()
        .map(|span| span.content.as_ref())
        .collect();
    assert_eq!(text, "fn main() {");
    assert!(highlight("hello", Language::File("notes.txt")).is_none());
    assert!(highlight("ls -la", Language::Named("bash")).is_some());
}
