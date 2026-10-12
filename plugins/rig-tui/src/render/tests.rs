use ratatui::style::Style;
use ratatui::text::{Line, Span};

use super::{Piece, fit, light};
use crate::Highlight;

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
fn a_piece_that_shrinks_is_cut_before_it_goes() {
    let goal = || Piece::new(20, Span::from("goal: ship the parser")).shrinking_to(8);
    let mut left = vec![Piece::new(ALWAYS, Span::from("model")), goal()];
    // "model  " is 7 wide; the goal is cut to the 13 columns left.
    fit(&mut left, &mut Vec::new(), 20);
    assert_eq!(text(&left, "  "), "model  goal: ship t…");
    // Never below its least; then it goes.
    let mut left = vec![Piece::new(ALWAYS, Span::from("model")), goal()];
    fit(&mut left, &mut Vec::new(), 14);
    assert_eq!(text(&left, "  "), "model");
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

#[test]
fn a_question_shows_its_body_choices_and_typed_answer() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    use ratatui::layout::Position;

    use super::super::pick::{Choosing, Item, Listing, Row};
    use super::draw_choice;

    let shown = Listing {
        title: "Allow `rm -rf build`?",
        body: "It matches `rm -rf`.\nType a reason to deny it with.",
        items: ["Deny", "Allow once", "Allow for this session"]
            .map(|label| Item {
                label,
                detail: None,
            })
            .into(),
        free_text: true,
        selected: 0,
        edit: None,
    };
    let choosing = Choosing {
        entity: bevy_ecs::entity::Entity::PLACEHOLDER,
        filter: "allow".to_owned(),
        selected: 0,
        draft: None,
    };
    // What is typed filters the choices and is an answer of its own.
    assert_eq!(
        shown.rows(&choosing.filter),
        [Row::Item(1), Row::Item(2), Row::Typed]
    );
    let Ok(mut terminal) = Terminal::new(TestBackend::new(60, 20));
    assert!(
        terminal
            .draw(|frame| draw_choice(frame, &choosing, &shown))
            .is_ok()
    );
    let buffer = terminal.backend().buffer();
    let screen: Vec<String> = (0..buffer.area.height)
        .map(|y| {
            (0..buffer.area.width)
                .filter_map(|x| {
                    buffer
                        .cell(Position::new(x, y))
                        .map(|cell| cell.symbol().to_owned())
                })
                .collect()
        })
        .collect();
    let screen = screen.join("\n");
    for shown in [
        "Allow `rm -rf build`?",
        "It matches `rm -rf`.",
        "› allow",
        "Allow once",
        "Allow for this session",
        "✎ allow",
    ] {
        assert!(screen.contains(shown), "{shown} not in\n{screen}");
    }
    assert!(!screen.contains("Deny"), "{screen}");
}

#[test]
fn a_panel_below_the_input_takes_the_rows_under_the_input_box() {
    use bevy_ecs::prelude::*;
    use ratatui::layout::{Constraint, Rect};

    use super::super::panel::{self, PanelCanvas, Placement, TuiPanel};
    use super::split;

    let screen = Rect::new(0, 0, 80, 24);
    let mut world = World::new();
    let suggestions = world
        .spawn(TuiPanel::new(Placement::BelowInput(Constraint::Length(1))))
        .id();
    // Too tall to leave the transcript its rows: skipped.
    world.spawn(TuiPanel::new(Placement::BelowInput(Constraint::Length(30))));
    let mut panels = world.query::<(Entity, &TuiPanel, &mut PanelCanvas)>();
    let mut panels: Vec<_> = panels.iter_mut(&mut world).collect();
    panels.sort_by_key(|(entity, ..)| *entity);
    let asked: Vec<u16> = panels
        .iter()
        .filter_map(|(_, panel, _)| match panel.placement {
            Placement::BelowInput(size) => Some(panel::rows_of(size, screen.height)),
            _ => None,
        })
        .collect();
    let [above, status, _, input, below] = split(screen, 1, 0, &asked);
    assert_eq!(status.y + 1, input.y);
    let transcript = panel::lay_out(
        panels.into_iter().map(|(_, panel, canvas)| (panel, canvas)),
        above,
        below,
        screen,
    );
    assert_eq!(transcript, above);
    let area = |entity: Entity| world.get::<PanelCanvas>(entity).map(PanelCanvas::area);
    assert_eq!(area(suggestions), Some(Rect::new(0, input.bottom(), 80, 1)));
    assert_eq!(input.bottom() + 1, screen.bottom());
}

#[test]
fn a_highlight_may_match_case_and_the_one_placed_first_wins() {
    let line = Line::from("Error: error at ERROR");
    let (current, rest) = (Style::new().on_red(), Style::new().on_yellow());
    let highlight = |text: &str, style: Style, match_case: bool| Highlight {
        text: text.to_owned(),
        style,
        match_case,
    };
    // The texts lit in `style`.
    let lit = |highlights: &[Highlight], style: Style| -> Vec<String> {
        let lit = light(line.clone(), highlights);
        let spans = lit.spans.iter().filter(|span| span.style.bg == style.bg);
        spans.map(|span| span.content.to_string()).collect()
    };
    let any_case = [highlight("error", rest, false)];
    assert_eq!(lit(&any_case, rest), ["Error", "error", "ERROR"]);
    let exact = [highlight("error", rest, true)];
    assert_eq!(lit(&exact, rest), ["error"]);
    // The current match, placed first, is set apart from the rest.
    let both = [highlight("Error:", current, true), any_case[0].clone()];
    assert_eq!(lit(&both, current), ["Error:"]);
    assert_eq!(lit(&both, rest), ["error", "ERROR"]);
}
