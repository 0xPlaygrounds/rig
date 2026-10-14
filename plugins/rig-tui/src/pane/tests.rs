use bevy_app::PostUpdate;
use ratatui::layout::Constraint;
use ratatui::text::Line;
use rig_core::completion::Message;
use rig_ecs::agent::{AgentId, CallOf, Conversation, Partial, TurnOf};
use rig_harness::prelude::*;

use crate::tests::{main_pane, main_view, view_app};
use crate::{
    ItemRow, MessageRenderer, PaneKeys, Placement, Role, ScrollTo, TranscriptItem, TranscriptPane,
    TranscriptSource, TranscriptView, TuiFrame, TuiPanel, TuiSystems, TuiTestScreen,
};

/// The rows of the screen's last frame within the panel `entity`'s area,
/// a line per row.
fn rows_in(app: &App, entity: Entity) -> Vec<String> {
    let world = app.world();
    let area = world
        .resource::<TuiFrame>()
        .area(entity)
        .unwrap_or_default();
    let screen = world.resource::<TuiTestScreen>().rows();
    let rows = screen
        .iter()
        .skip(usize::from(area.y))
        .take(usize::from(area.height));
    let column = |row: &String| {
        let cells = row.chars().skip(usize::from(area.x));
        cells.take(usize::from(area.width)).collect::<String>()
    };
    rows.map(column).collect()
}

/// Has `agent` stream a reply that says `text` so far.
fn stream(app: &mut App, agent: Entity, text: &str) {
    let world = app.world_mut();
    let turn = world.spawn(TurnOf(agent)).id();
    let partial = Partial {
        text: text.to_owned(),
        reasoning: String::new(),
    };
    world.spawn((CallOf(turn), partial));
}

#[test]
fn a_pane_over_a_live_subagent_streams_beside_the_main_one() {
    let (mut app, agent) = view_app();
    app.insert_resource(TuiTestScreen::new(100, 24));
    let helper = (Name::new("helper"), AgentId("helper".to_owned()));
    let helper = app
        .world_mut()
        .spawn((helper, Conversation::default()))
        .id();
    let pane = TranscriptPane::new(TranscriptSource::Agent(helper)).titled("helper");
    let placement = TuiPanel::new(Placement::Right(Constraint::Length(40)));
    let pane = app.world_mut().spawn((pane, placement, PaneKeys)).id();
    stream(&mut app, agent, "the main reply so far");
    stream(&mut app, helper, "the helper's reply so far");
    app.update();
    let main = rows_in(&app, main_pane(app.world()));
    let side = rows_in(&app, pane);
    let has = |rows: &[String], text: &str| rows.iter().any(|row| row.contains(text));
    assert!(has(&main, "the main reply so far"), "{main:#?}");
    assert!(!has(&main, "the helper's reply"), "{main:#?}");
    assert!(has(&side, "the helper's reply so far"), "{side:#?}");
    assert!(has(&side, "helper"), "{side:#?}");
}

#[test]
fn a_look_spawned_under_a_pane_applies_in_that_pane_only() {
    let (mut app, agent) = view_app();
    app.insert_resource(TuiTestScreen::new(100, 24));
    if let Some(mut conversation) = app.world_mut().get_mut::<Conversation>(agent) {
        conversation.push(Message::user("where is the config read?"));
    }
    let mut saved = rig_core::conversation::Conversation::default();
    saved.push(Message::user("where is the config read?"));
    let value = TranscriptPane::new(TranscriptSource::Value(saved)).titled("saved");
    let placement = TuiPanel::new(Placement::Left(Constraint::Length(45)));
    let pane = app.world_mut().spawn((value, placement)).id();
    let look = MessageRenderer::new(Role::User, |view| {
        vec![Line::from(format!("asked: {}", view.text))]
    });
    app.world_mut().spawn((look, ChildOf(pane)));
    app.update();
    let main = rows_in(&app, main_pane(app.world()));
    let side = rows_in(&app, pane);
    let has = |rows: &[String], text: &str| rows.iter().any(|row| row.contains(text));
    assert!(has(&side, "asked: where is the config read?"), "{side:#?}");
    assert!(has(&main, "› where is the config read?"), "{main:#?}");
    assert!(!has(&main, "asked:"), "{main:#?}");
}

/// Where the main pane's view was, and whether it changed, as a plugin's
/// system in `Draw` read it in the last frame.
#[derive(Resource, Default)]
struct SeenInDraw(Option<(Option<(TranscriptItem, usize)>, Option<u16>, bool)>);

#[test]
fn a_panes_scroll_and_width_are_placed_before_plugins_draw() {
    let (mut app, agent) = view_app();
    app.insert_resource(TuiTestScreen::new(60, 20))
        .init_resource::<SeenInDraw>()
        .add_systems(
            PostUpdate,
            (|views: Query<Ref<TranscriptView>, With<crate::MainPane>>,
              mut seen: ResMut<SeenInDraw>| {
                for view in &views {
                    let changed = view.is_changed();
                    seen.0 = Some((view.top().cloned(), view.width(), changed));
                }
            })
            .in_set(TuiSystems::Draw),
        );
    for index in 0..30 {
        if let Some(mut conversation) = app.world_mut().get_mut::<Conversation>(agent) {
            conversation.push(Message::user(format!("question {index}")));
        }
    }
    app.update();
    app.update();
    let seen = |app: &App| app.world().resource::<SeenInDraw>().0.clone();
    assert_eq!(seen(&app).map(|(top, ..)| top), Some(None));
    let hit = app.world().get::<Conversation>(agent).and_then(|c| c.id(3));
    let item = hit.map(TranscriptItem::Message);
    let pane = main_pane(app.world());
    let row = ItemRow::Top(0);
    app.world_mut().trigger(ScrollTo {
        pane,
        item: item.clone(),
        row,
    });
    app.update();
    // The frame the scroll was asked in places it, changed, before Draw.
    let at = item.map(|item| (item, 0));
    assert_eq!(seen(&app), Some((at.clone(), Some(60), true)));
    // A resized screen lays the pane out at its new width, changed.
    app.world_mut()
        .resource_mut::<TuiTestScreen>()
        .resize(50, 20);
    app.update();
    assert_eq!(seen(&app), Some((at, Some(50), true)));
    assert_eq!(main_view(&app).width(), Some(50));
    // A frame that moves nothing leaves the view unchanged.
    app.world_mut().write_message(crate::RequestRedraw);
    app.update();
    assert!(seen(&app).is_some_and(|(.., changed)| !changed));
}
