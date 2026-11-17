use super::{render, render_marked};

/// `text` rendered and drawn, colored.
pub(super) fn rendered(
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

/// `text` with `marks` rendered, a line per row.
fn marked(text: &str, marks: &[(usize, &'static str)]) -> String {
    let marks: Vec<_> = marks
        .iter()
        .map(|(at, mark)| (*at, ratatui::text::Span::from(*mark)))
        .collect();
    let lines = render_marked(text, &marks, &crate::Theme::default(), None);
    let rows: Vec<String> = lines.iter().map(ToString::to_string).collect();
    rows.join("\n")
}

/// Private-use characters in the text stay themselves, and each mark is
/// drawn once where it goes, however many share a byte.
#[test]
fn private_use_text_is_kept_apart_from_marks() {
    assert_eq!(marked("a\u{E000}b", &[(0, "[m]")]), "[m]a\u{E000}b");
    assert_eq!(
        marked("a\u{E000}\u{E001}b", &[(1, "[1]"), (1, "[2]"), (9, "[3]")]),
        "a[1][2]\u{E000}\u{E001}b[3]"
    );
    // A numeric character reference becomes the character it names.
    assert_eq!(marked("a&#xE000;b", &[(0, "[m]")]), "[m]a\u{E000}b");
    assert_eq!(
        marked("\u{E000}&#xE001;", &[(0, "[m]")]),
        "[m]\u{E000}\u{E001}"
    );
    // A stand-in can end what kept a reference literal, so it decodes.
    assert_eq!(marked(r"\&#xE000;", &[(1, "[m]")]), "\\[m]\u{E000}");
    assert_eq!(marked("    &#xE000;", &[(0, "[m]")]), "[m]    \u{E000}");
    assert_eq!(
        marked("<!-- &#xE000; -->", &[(1, "[m]")]),
        "<[m]!-- \u{E000} -->"
    );
}

/// A text holding every private-use character is drawn whole, its marks
/// in place, though none is free to stand a mark in.
#[test]
fn marks_are_kept_when_no_private_use_character_is_free() {
    let planes = [0xE000..=0xF8FF, 0xF_0000..=0xF_FFFD, 0x10_0000..=0x10_FFFD];
    let used: String = planes
        .into_iter()
        .flatten()
        .filter_map(char::from_u32)
        .collect();
    let text = format!("{used}\nnext");
    let drawn = marked(&text, &[(0, "[1]"), (text.len(), "[2]")]);
    assert_eq!(drawn, format!("[1]{used}\nnext[2]"));
}
