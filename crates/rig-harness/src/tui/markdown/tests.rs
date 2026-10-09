use super::render;

fn text(markdown: &str) -> Vec<String> {
    render(markdown)
        .iter()
        .map(|line| line.to_string())
        .collect()
}

#[test]
fn a_fence_language_is_not_a_line() {
    let lines = text("Here:\n\n```rust\nfn main() {}\n```\n");
    assert_eq!(lines, ["Here:", "", "  fn main() {}"]);
}
