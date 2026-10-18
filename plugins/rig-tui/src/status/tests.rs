use bevy_app::prelude::*;
use bevy_ecs::prelude::*;

use ratatui::text::Span;
use rig_ecs::agent::Agent;

use super::*;

const CONTEXT: StatusItem = StatusItem::new("context", Side::Right, 40, 5);

/// Whether the gauge plugin hides the `context` item instead of
/// replacing it.
#[derive(Resource, Default)]
struct Hiding(bool);

/// The owner shows its `context` item every frame.
fn show(mut items: Query<&mut StatusItems, With<Agent>>) {
    for mut items in &mut items {
        items.show(CONTEXT.says("ctx 30k/1M (3%)", Tone::Dim));
    }
}

/// Another plugin replaces it, or hides it, by name.
fn gauge(mut items: Query<&mut StatusItems, With<Agent>>, hiding: Res<Hiding>) {
    for mut items in &mut items {
        match hiding.0 {
            true => items.hide("context"),
            false => {
                items.show(StatusItem::new("context", Side::Right, 40, 5).says("▮▯▯", Tone::Red))
            }
        }
    }
}

fn shown(app: &mut App, agent: Entity) -> Vec<(String, String)> {
    let items = app.world().get::<StatusItems>(agent);
    let items = items.into_iter().flat_map(StatusItems::iter);
    items.map(|item| (item.name.to_string(), item.text.clone())).collect()
}

#[test]
fn a_plugin_replaces_or_hides_another_plugins_item_by_name() {
    let mut app = App::new();
    add(&mut app);
    app.init_resource::<Hiding>()
        // Added first, the replacing plugin still runs after the owner.
        .add_systems(PostUpdate, gauge.in_set(StatusSystems::Edit))
        .add_systems(PostUpdate, show.in_set(StatusSystems::Show));
    let agent = app.world_mut().spawn(Agent).id();
    app.update();
    assert_eq!(shown(&mut app, agent), [("context".to_owned(), "▮▯▯".to_owned())]);
    app.world_mut().resource_mut::<Hiding>().0 = true;
    app.update();
    assert_eq!(shown(&mut app, agent), []);
}

/// The `keep` of the default plugins' items: the model and the status
/// always stay.
const ALWAYS: u8 = u8::MAX;

fn text(pieces: &[Piece], gap: &str) -> String {
    let texts: Vec<String> = pieces.iter().map(|piece| piece.line.to_string()).collect();
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
fn two_items_at_one_place_both_show_by_name() {
    let mut items = StatusItems::default();
    let at = |name| StatusItem::new(name, Side::Left, 45, 9);
    items.show(at("plan").says("plan", Tone::Cyan));
    items.show(at("mode").says("auto", Tone::Yellow));
    let (left, right) = pieces(items.iter());
    assert_eq!(text(&left, "  "), "auto  plan");
    assert!(right.is_empty());
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
fn a_status_line_shows_its_agents_items_in_their_styles() {
    use ratatui::style::Color;
    use rig_ecs::agent::AgentId;

    use crate::tests::view_app;
    use crate::{TuiFrame, TuiTestScreen};

    let (mut app, agent) = view_app();
    app.insert_resource(TuiTestScreen::new(80, 24));
    let helper = (Agent, AgentId("helper".to_owned()));
    let helper = app.world_mut().spawn(helper).id();
    let line = app
        .world_mut()
        .spawn(StatusLine {
            agent: Some(helper),
        })
        .id();
    let gauge = StatusItem::new("gauge", Side::Left, 10, 5);
    let gauge = gauge.styled([("▮▮", Tone::Red), ("▯", Tone::Dim)]);
    if let Some(mut items) = app.world_mut().get_mut::<StatusItems>(helper) {
        items.show(gauge);
    }
    if let Some(mut items) = app.world_mut().get_mut::<StatusItems>(agent) {
        items.show(StatusItem::new("mine", Side::Left, 10, 5).says("main's", Tone::Plain));
    }
    app.update();
    let world = app.world();
    let area = world.resource::<TuiFrame>().area(line).unwrap_or_default();
    let buffer = world.resource::<TuiTestScreen>().buffer();
    let row: String = (area.left()..area.right())
        .filter_map(|x| {
            buffer
                .cell((x, area.y))
                .map(|cell| cell.symbol().to_owned())
        })
        .collect();
    assert!(row.starts_with("▮▮▯"), "{row:?}");
    assert!(!row.contains("main's"), "{row:?}");
    let first = buffer.cell((area.x, area.y)).map(|cell| cell.fg);
    assert_eq!(first, Some(Color::Red));
}
