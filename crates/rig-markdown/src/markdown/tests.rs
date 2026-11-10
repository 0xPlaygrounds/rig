use ratatui_core::text::{Line, Span};

use super::{Highlight, Kind, Palette, Plain, render};

/// Colors every named language, marking each line with its place.
struct Numbered;

impl Highlight for Numbered {
    type Mark = usize;

    fn block(
        &self,
        code: &str,
        _: &str,
        prefix: &[Span<'static>],
    ) -> Option<Vec<(Line<'static>, Option<usize>)>> {
        let lines = code.split('\n').enumerate().map(|(index, line)| {
            let spans = [prefix, &[Span::from(line.to_owned())]].concat();
            (Line::from(spans), Some(index))
        });
        Some(lines.collect())
    }
}

/// Each line's text and kind.
fn drawn<H: Highlight>(markdown: &str, highlight: &H) -> Vec<(String, Kind<H::Mark>)> {
    let lines = render(markdown, &Palette::default(), None, highlight);
    lines
        .into_iter()
        .map(|(line, kind)| (line.to_string(), kind))
        .collect()
}

#[test]
fn every_code_line_says_it_is_code_and_a_colored_one_carries_its_mark() {
    let markdown = "Intro\n\n```rust\na\nb\n```\n\n    indented\n\nafter\n";
    let colored = drawn(markdown, &Numbered);
    let kinds: Vec<_> = colored.iter().map(|(_, kind)| *kind).collect();
    assert_eq!(
        kinds,
        [
            Kind::Text,
            Kind::Text,
            Kind::Text,
            Kind::Code(Some(0)),
            Kind::Code(Some(1)),
            Kind::Text,
            Kind::Code(None),
            Kind::Text,
            Kind::Text,
        ],
        "{colored:#?}"
    );
    let label = colored.get(2).map(|(text, _)| text.as_str());
    assert_eq!(label, Some("  rust"));
    let plain = drawn(markdown, &Plain);
    let code: Vec<&str> = (plain.iter())
        .filter(|(_, kind)| matches!(kind, Kind::Code(_)))
        .map(|(text, _)| text.as_str())
        .collect();
    assert_eq!(code, ["  a", "  b", "  indented"]);
}

/// Each line's text.
fn texts<H: Highlight>(markdown: &str, highlight: &H) -> Vec<String> {
    let lines = drawn(markdown, highlight).into_iter();
    lines.map(|(text, _)| text).collect()
}

#[test]
fn a_list_item_that_starts_with_code_keeps_its_marker() {
    for (markdown, marker) in [
        ("1. ```\n   cargo build\n   ```\n2. Run it.\n", "1. "),
        ("- ```\n  x\n  ```\n", "• "),
        ("- ```sh\n  x\n  ```\n", "• "),
    ] {
        for lines in [texts(markdown, &Plain), texts(markdown, &Numbered)] {
            let first = lines.first().map(String::as_str).unwrap_or_default();
            assert!(first.starts_with(marker), "{markdown:?}: {lines:#?}");
        }
    }
}
