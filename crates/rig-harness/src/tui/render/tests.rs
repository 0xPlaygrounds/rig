use ratatui::text::Span;

use super::{Piece, fit, join, keep};

fn text(pieces: Vec<Piece>, gap: &'static str) -> String {
    join(pieces, gap).to_string()
}

fn left() -> Vec<Piece> {
    vec![
        Piece::new(keep::SESSION, Span::from("my session")),
        Piece::new(keep::ALWAYS, Span::from("deepseek/deepseek-flash")),
        Piece::new(keep::REASONING, Span::from("reasoning default")),
        Piece::new(keep::ALWAYS, Span::from("thinking… (Esc stops)")),
        Piece::new(keep::SUBAGENTS, Span::from("2 subagents working")),
    ]
}

fn meter() -> Vec<Piece> {
    vec![
        Piece::new(keep::TOKENS, Span::from("↑5.4k ↓1.2k")),
        Piece::new(keep::CACHE, Span::from("cache 27k")),
        Piece::new(keep::COST, Span::from("$0.012")),
        Piece::new(keep::CONTEXT, Span::from("ctx 30k/1M (3%)")),
    ]
}

#[test]
fn a_wide_line_keeps_everything() {
    let (mut left, mut right) = (left(), meter());
    fit(&mut left, &mut right, 200);
    assert_eq!(left.len(), 5);
    assert_eq!(right.len(), 4);
}

#[test]
fn the_meter_shrinks_before_the_status() {
    let (mut left, mut right) = (left(), meter());
    fit(&mut left, &mut right, 86);
    let (status, usage) = (text(left, "  "), text(right, " "));
    // The meter goes, then the session's name.
    assert_eq!(
        status,
        "deepseek/deepseek-flash  reasoning default  thinking… (Esc stops)  2 subagents working"
    );
    assert!(usage.is_empty(), "{usage}");
}

#[test]
fn cache_and_cost_go_first() {
    let mut left = vec![
        Piece::new(keep::ALWAYS, Span::from("model")),
        Piece::new(keep::ALWAYS, Span::from("idle")),
    ];
    let mut right = meter();
    // "model  idle" is 11 wide; the gap 2; tokens and context 27.
    fit(&mut left, &mut right, 40);
    assert_eq!(text(right, " "), "↑5.4k ↓1.2k ctx 30k/1M (3%)");
    assert_eq!(text(left, "  "), "model  idle");
}

#[test]
fn the_model_and_status_always_stay() {
    let (mut left, mut right) = (left(), meter());
    fit(&mut left, &mut right, 10);
    assert!(right.is_empty());
    assert_eq!(
        text(left, "  "),
        "deepseek/deepseek-flash  thinking… (Esc stops)"
    );
}
