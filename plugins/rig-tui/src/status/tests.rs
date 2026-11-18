use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use rig_ecs::spawn::spawn_root;

use ratatui::text::Span;

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

fn shown(app: &mut App, agent: Entity) -> Vec<(&'static str, String)> {
    let items = app.world().get::<StatusItems>(agent);
    let items = items.into_iter().flat_map(StatusItems::iter);
    items.map(|item| (item.name, item.text.clone())).collect()
}

#[test]
fn a_plugin_replaces_or_hides_another_plugins_item_by_name() {
    let mut app = App::new();
    add(&mut app);
    app.init_resource::<Hiding>()
        // Added first, the replacing plugin still runs after the owner.
        .add_systems(PostUpdate, gauge.in_set(StatusSystems::Edit))
        .add_systems(PostUpdate, show.in_set(StatusSystems::Show));
    let agent = spawn_root(app.world_mut(), ()).id();
    app.update();
    assert_eq!(shown(&mut app, agent), [("context", "▮▯▯".to_owned())]);
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
    use ratatui::style::{Color, Stylize};
    use ratatui::text::Line;
    use rig_ecs::agent::AgentId;

    use crate::tests::view_app;
    use crate::{TuiFrame, TuiTestScreen};

    let (mut app, agent) = view_app();
    app.insert_resource(TuiTestScreen::new(80, 24));
    let helper = AgentId("helper".to_owned());
    let helper = spawn_root(app.world_mut(), helper).id();
    let line = app
        .world_mut()
        .spawn(StatusLine {
            agent: Some(helper),
        })
        .id();
    let gauge = StatusItem::new("gauge", Side::Left, 10, 5);
    let gauge = gauge.styled(Line::from(vec![
        Span::from("▮▮").red(),
        Span::from("▯").dim(),
    ]));
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

/// The main input's bottom row on the test screen, and the input's height.
fn input_bottom_row(app: &mut App) -> (String, u16) {
    use crate::{MainEditor, TuiFrame, TuiTestScreen};

    let world = app.world_mut();
    let input = world
        .query_filtered::<Entity, With<MainEditor>>()
        .single(world)
        .unwrap_or(Entity::PLACEHOLDER);
    let world = app.world();
    let area = world.resource::<TuiFrame>().area(input).unwrap_or_default();
    let buffer = world.resource::<TuiTestScreen>().buffer();
    let bottom = area.bottom().saturating_sub(1);
    let row = (area.left()..area.right())
        .filter_map(|x| {
            buffer
                .cell((x, bottom))
                .map(|cell| cell.symbol().to_owned())
        })
        .collect();
    (row, area.height)
}

/// Shows `items` on `agent`, the shown agent, on an 80 by 24 test screen,
/// then draws a frame.
fn draw_with(app: &mut App, agent: Entity, items: &[(&'static str, Side, &str)]) {
    use crate::TuiTestScreen;
    use rig_harness::main_agent::ShownAgent;

    app.insert_resource(TuiTestScreen::new(80, 24))
        .insert_resource(ShownAgent(Some(agent)));
    if let Some(mut status) = app.world_mut().get_mut::<StatusItems>(agent) {
        for (order, (name, side, text)) in (0u8..).zip(items) {
            status.show(StatusItem::new(name, *side, order, 5).says(*text, Tone::Plain));
        }
    }
    app.update();
}

#[test]
fn without_the_status_line_plugin_the_input_keeps_an_empty_bottom_row() {
    use crate::TuiPlugin;
    use crate::tests::{Delivered, press, type_text, view_app, view_app_of};

    let (mut app, agent) = view_app_of(|app| {
        app.add_plugins(TuiPlugin);
    });
    draw_with(&mut app, agent, &[("model", Side::Left, "acme-model")]);
    let (row, height) = input_bottom_row(&mut app);
    assert_eq!(row.trim(), "", "{row:?}");
    assert_eq!(height, 3);
    type_text(&mut app, "hello");
    press(&mut app, "enter");
    assert_eq!(app.world().resource::<Delivered>().0, ["hello"]);
    // With it, the same row shows the line.
    let (mut app, agent) = view_app();
    draw_with(&mut app, agent, &[("model", Side::Left, "acme-model")]);
    let (row, height) = input_bottom_row(&mut app);
    assert!(row.contains("acme-model"), "{row:?}");
    assert_eq!(height, 3);
}

#[test]
fn hidden_parts_are_not_drawn_but_stay_among_the_items() {
    use crate::TuiPlugins;
    use crate::tests::view_app_of;
    use rig_harness::main_agent::ShownAgent;
    use rig_harness::settings::Settings;

    let settings = "[status]\nhide = [\"cost\", \"cache\", \"agent\", \"nothing-draws-this\"]\n";
    let (mut app, agent) = view_app_of(|app| {
        app.insert_resource(Settings::from_toml(settings))
            .add_plugins(TuiPlugins);
    });
    let items = [
        ("tokens", Side::Right, "12k tok"),
        ("cache", Side::Right, "cache 80%"),
        ("cost", Side::Right, "$0.12"),
    ];
    // The app's items and the running turn's are hidden by name too.
    let app_items = &mut app.world_mut().resource_mut::<AppStatus>().0;
    app_items.show(StatusItem::new("cost", Side::Right, 50, 5).says("app $9", Tone::Plain));
    app_items.show(StatusItem::new("update", Side::Left, 50, 5).says("v2 out", Tone::Plain));
    let mut turn = StatusItems::default();
    turn.show(StatusItem::new("cache", Side::Left, 60, 5).says("turn cache", Tone::Plain));
    turn.show(StatusItem::new("turn", Side::Left, 61, 5).says("turn 3s", Tone::Plain));
    app.world_mut().spawn((TurnOf(agent), turn));
    draw_with(&mut app, agent, &items);
    let (row, _) = input_bottom_row(&mut app);
    for shown in ["12k tok", "v2 out", "turn 3s"] {
        assert!(row.contains(shown), "{shown}: {row:?}");
    }
    for hidden in ["cache 80%", "$0.12", "app $9", "turn cache"] {
        assert!(!row.contains(hidden), "{hidden}: {row:?}");
    }
    let mut kept: Vec<&str> = (app.world().get::<StatusItems>(agent).into_iter())
        .flat_map(StatusItems::iter)
        .map(|item| item.name)
        .collect();
    kept.sort_unstable();
    assert_eq!(kept, ["cache", "cost", "tokens"]);
    assert_eq!(
        app.world().resource::<Settings>().problems(),
        [] as [String; 0]
    );
    // The placeholder for no agent is no part.
    app.world_mut().despawn(agent);
    app.insert_resource(ShownAgent(None));
    app.world_mut().write_message(crate::RequestRedraw);
    app.update();
    let (row, _) = input_bottom_row(&mut app);
    assert!(row.contains("no agent"), "{row:?}");
}

#[test]
fn an_input_spawned_after_startup_gets_the_status_line() {
    use crate::tests::view_app_of;
    use crate::{MainEditor, TuiViewPlugin};

    let (mut app, agent) = view_app_of(|app| {
        app.add_plugins((TuiViewPlugin, StatusLinePlugin));
    });
    app.update();
    app.world_mut().spawn((
        Editor::default(),
        MainEditor,
        TuiPanel::new(Placement::Bottom(Constraint::Length(3))),
    ));
    draw_with(&mut app, agent, &[("model", Side::Left, "acme-model")]);
    let (row, _) = input_bottom_row(&mut app);
    assert!(row.contains("acme-model"), "{row:?}");
}
