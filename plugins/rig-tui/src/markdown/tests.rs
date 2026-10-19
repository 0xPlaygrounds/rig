use super::render;

fn text(markdown: &str) -> Vec<String> {
    render(markdown, None)
        .iter()
        .map(|line| line.to_string())
        .collect()
}

#[test]
fn a_fence_language_is_not_a_line() {
    let lines = text("Here:\n\n```rust\nfn main() {}\n```\n");
    assert_eq!(lines, ["Here:", "", "  fn main() {}"]);
}

#[test]
fn an_image_is_drawn_as_a_link() {
    assert_eq!(
        text("See ![the plot](plot.png)."),
        ["See the plot (plot.png)."]
    );
}

#[test]
fn a_fenced_block_is_drawn_in_its_languages_colors() {
    let lines = render("```rust\nfn main() {}\n```\n", None);
    let colored = lines
        .iter()
        .flat_map(|line| &line.spans)
        .filter(|span| matches!(span.style.fg, Some(ratatui::style::Color::Indexed(_))));
    assert!(colored.count() > 1, "{lines:?}");
    assert_eq!(text("```rust\nfn main() {}\n```\n"), ["  fn main() {}"]);
}
