use bevy_app::PostUpdate;
use ratatui::layout::Constraint;
use ratatui::text::Line;
use rig_core::completion::Message;
use rig_ecs::agent::{AgentId, CallOf, Conversation, Partial, TurnOf};
use rig_harness::prelude::*;

use crate::tests::{main_pane, main_view, press, view_app};
use crate::{
    ItemRow, KeyFocus, PaneKeys, Placement, RenderBatch, RenderSystems, ScrollTo, TranscriptItem,
    TranscriptPane, TranscriptRender, TranscriptSource, TranscriptView, TuiFrame, TuiPanel,
    TuiSystems, TuiTestScreen,
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
    app.add_systems(
        TranscriptRender,
        (move |mut batch: ResMut<RenderBatch>| {
            if batch.pane != Some(pane) {
                return;
            }
            for (_, output) in batch.messages_mut() {
                output.decorate("asked", vec![Line::from("asked here")]);
            }
        })
        .in_set(RenderSystems::Decorate),
    );
    app.update();
    let main = rows_in(&app, main_pane(app.world()));
    let side = rows_in(&app, pane);
    let has = |rows: &[String], text: &str| rows.iter().any(|row| row.contains(text));
    assert!(has(&side, "asked here"), "{side:#?}");
    assert!(has(&main, "› where is the config read?"), "{main:#?}");
    assert!(!has(&main, "asked here"), "{main:#?}");
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

/// A header panel shown while the main pane is scrolled up.
#[derive(Component)]
struct Header;

/// Shows the header while the main pane is scrolled up: a row, else none.
fn fit_header(
    views: Query<&TranscriptView, With<crate::MainPane>>,
    mut headers: Query<&mut TuiPanel, With<Header>>,
) {
    let scrolled = views.iter().any(|view| view.top().is_some());
    let rows = u16::from(scrolled);
    for mut panel in &mut headers {
        panel.set_if_neq(TuiPanel::new(Placement::Top(Constraint::Length(rows))));
    }
}

#[test]
fn a_panel_fitted_to_a_panes_scroll_is_laid_out_in_the_same_frame() {
    let (mut app, agent) = view_app();
    app.insert_resource(TuiTestScreen::new(60, 20)).add_systems(
        PostUpdate,
        (
            fit_header.in_set(TuiSystems::Fit),
            (|mut headers: Query<&mut crate::PanelCanvas, With<Header>>| {
                for mut canvas in &mut headers {
                    canvas.render(Line::raw("HEADER"));
                }
            })
            .in_set(TuiSystems::Draw),
        ),
    );
    let header = app
        .world_mut()
        .spawn((Header, TuiPanel::new(Placement::Top(Constraint::Length(0)))))
        .id();
    for index in 0..30 {
        if let Some(mut conversation) = app.world_mut().get_mut::<Conversation>(agent) {
            conversation.push(Message::user(format!("question {index}")));
        }
    }
    app.update();
    app.update();
    let pane = main_pane(app.world());
    let area = |app: &App| app.world().resource::<TuiFrame>().area(pane);
    let before = area(&app).map(|area| (area.y, area.height));
    assert_eq!(before.map(|(y, _)| y), Some(0));
    let item = app.world().get::<Conversation>(agent).and_then(|c| c.id(3));
    let item = item.map(TranscriptItem::Message);
    app.world_mut().trigger(ScrollTo {
        pane,
        item: item.clone(),
        row: ItemRow::Top(0),
    });
    app.update();
    // The frame the scroll was placed in draws the header, the pane below.
    let rows = rows_in(&app, header);
    assert_eq!(
        rows.iter().map(|row| row.trim_end()).collect::<Vec<_>>(),
        ["HEADER"]
    );
    let shrunk = before.map(|(y, height)| (y.saturating_add(1), height.saturating_sub(1)));
    assert_eq!(area(&app).map(|area| (area.y, area.height)), shrunk);
    assert_eq!(main_view(&app).top().cloned(), item.map(|item| (item, 0)));
}

/// The numbers of the `row-N` lines in `rows`, top to bottom.
fn numbered(rows: &[String]) -> Vec<usize> {
    let number = |row: &String| {
        let (_, rest) = row.split_once("row-")?;
        let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
        digits.parse().ok()
    };
    rows.iter().filter_map(number).collect()
}

#[test]
fn a_page_key_keeps_a_line_of_text_and_stops_at_the_ends() {
    // 20 rows page by 18, more than the 10 lines a page used to be.
    for height in [20, 10, 2, 1] {
        let (mut app, _) = view_app();
        app.insert_resource(TuiTestScreen::new(60, 30));
        let lines: Vec<String> = (1..=60).map(|n| format!("row-{n}")).collect();
        let mut saved = rig_core::conversation::Conversation::default();
        saved.push(Message::user(lines.join("\n")));
        let pane = TranscriptPane::new(TranscriptSource::Value(saved));
        let placement = TuiPanel::new(Placement::Top(Constraint::Length(height)));
        let pane = app.world_mut().spawn((pane, placement, PaneKeys)).id();
        app.world_mut().resource_mut::<KeyFocus>().0 = Some(pane);
        app.update();
        let shown = |app: &App| numbered(&rows_in(app, pane));
        let bottom = shown(&app);
        press(&mut app, "pageup");
        let first = shown(&app);
        if height == 1 {
            // One row keeps its text, with no marker over it.
            let row = rows_in(&app, pane).concat();
            assert!(row.contains("row-") && !row.contains("more below"), "{row}");
        }
        press(&mut app, "pageup");
        let second = shown(&app);
        press(&mut app, "pagedown");
        let back = shown(&app);
        let progress = [&bottom, &first, &second, &back].map(|rows| rows.first().copied());
        if height >= 10 {
            // The old top text row is the new bottom one, and back.
            assert_eq!(first.last(), bottom.first(), "{progress:?}");
            assert_eq!(second.last(), first.first(), "{progress:?}");
            assert_eq!(back.first(), second.last(), "{progress:?}");
        } else {
            let [bottom, first, second, back] = progress;
            let shown = progress.iter().all(Option::is_some);
            assert!(
                shown && first < bottom && second < first && back > second,
                "{progress:?}"
            );
        }
        // A blank row is above the message, so a short pane at the top
        // may show no text: the top stays put.
        for _ in 0..70 {
            press(&mut app, "pageup");
        }
        let top = rows_in(&app, pane);
        press(&mut app, "pageup");
        assert_eq!(rows_in(&app, pane), top, "height {height}");
        if height >= 10 {
            assert_eq!(shown(&app).first(), Some(&1));
        }
        for _ in 0..70 {
            press(&mut app, "pagedown");
        }
        assert_eq!(shown(&app).last(), Some(&60), "height {height}");
    }
}
