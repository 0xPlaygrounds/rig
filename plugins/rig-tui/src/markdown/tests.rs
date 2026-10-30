use super::{render, render_marked};

/// `text` rendered and drawn, colored.
fn rendered(
    text: &str,
    theme: &crate::Theme,
    width: Option<usize>,
) -> Vec<ratatui::text::Line<'static>> {
    let lines = render(text, theme, width);
    lines.iter().map(crate::LookLine::colored).collect()
}

fn text(markdown: &str) -> Vec<String> {
    rendered(markdown, &crate::Theme::default(), None)
        .iter()
        .map(|line| line.to_string())
        .collect()
}

#[test]
fn a_fence_language_is_the_blocks_label_not_its_code() {
    let lines = text("Here:\n\n```rust\nfn main() {}\n```\n");
    assert_eq!(lines, ["Here:", "", "  rust", "  fn main() {}"]);
    assert_eq!(text("```\nplain\n```\n"), ["  plain"]);
    assert_eq!(text("```text\nplain\n```\n"), ["  plain"]);
    assert_eq!(text("```nolang\nplain\n```\n"), ["  plain"]);
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
    let lines = rendered(
        "```rust\nfn main() {}\n```\n",
        &crate::Theme::default(),
        None,
    );
    let colored = lines
        .iter()
        .flat_map(|line| &line.spans)
        .filter(|span| matches!(span.style.fg, Some(ratatui::style::Color::Indexed(_))));
    assert!(colored.count() > 1, "{lines:?}");
}

#[test]
fn an_alert_is_on_its_tint_under_its_name() {
    let theme = crate::Theme::default();
    let lines = rendered("> [!WARNING]\n> Back it up first.\n", &theme, None);
    let rows: Vec<String> = lines.iter().map(ToString::to_string).collect();
    assert_eq!(rows, ["Warning", "  Back it up first."]);
    assert!(
        lines
            .iter()
            .all(|line| line.style.bg == Some(theme.call_running))
    );
    // A plain quote has no tint and no name.
    let quote = rendered("> just a quote\n", &theme, None);
    assert!(quote.iter().all(|line| line.style.bg.is_none()));
}

#[test]
fn code_is_on_the_code_tint_and_a_block_says_its_language() {
    let theme = crate::Theme::default();
    let lines = rendered("Run `cargo test`.\n\n```sh\nls\n```\n", &theme, None);
    let inline = lines
        .first()
        .and_then(|line| line.spans.iter().find(|span| span.content == "cargo test"));
    assert!(inline.is_some_and(|span| span.style.bg == Some(theme.code)));
    let block: Vec<_> = lines.iter().skip(2).collect();
    assert_eq!(
        block.first().map(ToString::to_string).as_deref(),
        Some("  sh")
    );
    assert!(block.iter().all(|line| line.style.bg == Some(theme.code)));
    let plain = theme.without_color();
    let rows: Vec<String> = rendered("Run `cargo test`.", &plain, None)
        .iter()
        .map(ToString::to_string)
        .collect();
    assert_eq!(rows, ["Run `cargo test`."]);
}

#[test]
fn a_task_says_done_or_open_with_a_mark() {
    assert_eq!(
        text("- [x] built\n- [ ] tested\n"),
        ["• ✓ built", "• ○ tested"]
    );
}

#[test]
fn a_mark_in_a_code_block_is_drawn_in_the_lines_colors() {
    let text = "```rust\nlet x = 1;\n```\n";
    let at = text.find("1;").unwrap_or_default();
    let mark = ratatui::text::Span::from("[m]");
    let lines = render_marked(text, &[(at, mark.clone())], &crate::Theme::default(), None);
    let line = lines
        .get(1)
        .map(crate::LookLine::colored)
        .unwrap_or_default();
    assert_eq!(line.to_string(), "  let x = [m]1;");
    assert!(line.spans.contains(&mark), "{line:?}");
    assert!(
        line.spans.iter().any(|span| span.style.fg.is_some()),
        "{line:?}"
    );
}
