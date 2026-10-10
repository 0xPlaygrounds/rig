use ratatui::text::Span;

use super::{Piece, fit};

/// The `keep` of the default plugins' items: the model and the status
/// always stay.
const ALWAYS: u8 = u8::MAX;

fn text(pieces: &[Piece], gap: &str) -> String {
    let texts: Vec<&str> = pieces
        .iter()
        .map(|piece| piece.span.content.as_ref())
        .collect();
    texts.join(gap)
}

fn left() -> Vec<Piece> {
    vec![
        Piece::new(2, Span::from("my session")),
        Piece::new(ALWAYS, Span::from("deepseek/deepseek-flash")),
        Piece::new(12, Span::from("reasoning default")),
        Piece::new(ALWAYS, Span::from("thinking… (Esc stops)")),
        Piece::new(15, Span::from("2 subagents working")),
    ]
}

/// Tokens, cache reads, cost and context.
fn meter() -> Vec<Piece> {
    vec![
        Piece::new(4, Span::from("↑5.4k ↓1.2k")),
        Piece::new(1, Span::from("cache 27k")),
        Piece::new(3, Span::from("$0.012")),
        Piece::new(5, Span::from("ctx 30k/1M (3%)")),
    ]
}

#[test]
fn the_meter_shrinks_before_the_status() {
    let (mut left, mut right) = (left(), meter());
    fit(&mut left, &mut right, 86);
    let (status, usage) = (text(&left, "  "), text(&right, " "));
    // The meter goes, and the session's name with it.
    assert_eq!(
        status,
        "deepseek/deepseek-flash  reasoning default  thinking… (Esc stops)  2 subagents working"
    );
    assert!(usage.is_empty(), "{usage}");
}

#[test]
fn cache_and_a_long_session_name_go_before_what_the_session_spends() {
    // The resumed dogfood session: a long name must not hide the spending.
    let mut left = vec![
        Piece::new(2, Span::from("stats refactor")),
        Piece::new(ALWAYS, Span::from("model")),
        Piece::new(ALWAYS, Span::from("idle")),
    ];
    let mut right = meter();
    // "model  idle" is 11 wide; the gap 2; tokens, cost and context 34.
    fit(&mut left, &mut right, 47);
    assert_eq!(left.len(), 2);
    assert_eq!(text(&right, " "), "↑5.4k ↓1.2k $0.012 ctx 30k/1M (3%)");
    // Then the cost.
    fit(&mut left, &mut right, 40);
    assert_eq!(text(&left, "  "), "model  idle");
    assert_eq!(text(&right, " "), "↑5.4k ↓1.2k ctx 30k/1M (3%)");
}

#[test]
fn the_model_and_status_always_stay() {
    let (mut left, mut right) = (left(), meter());
    fit(&mut left, &mut right, 10);
    assert!(right.is_empty());
    assert_eq!(
        text(&left, "  "),
        "deepseek/deepseek-flash  thinking… (Esc stops)"
    );
}
