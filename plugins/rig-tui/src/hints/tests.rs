use bevy_app::prelude::*;
use rig_ecs::agent::TurnOf;
use rig_harness::prelude::*;
use unicode_width::UnicodeWidthStr;

use super::{fit_keys, keys};
use crate::tests::{Delivered, press, type_text, typed, view_app_with};
use crate::{KeyFocus, Placement, PromptHistory, TuiPanel, TuiPlugins, TuiTestScreen};

/// The input's three rows on the screen, from the top: its keys, its text
/// and its status line.
fn input(app: &App) -> Vec<String> {
    let rows = app.world().resource::<TuiTestScreen>().rows();
    rows.get(rows.len().saturating_sub(3)..)
        .map(<[String]>::to_vec)
        .unwrap_or_default()
}

/// The row above the input's text.
fn key_row(app: &App) -> String {
    input(app).first().cloned().unwrap_or_default()
}

/// The view, with the key hints when `hints`, on a 100 by 24 screen,
/// drawn once.
fn screen(hints: bool) -> (App, Entity) {
    let (mut app, agent) = view_app_with(|app| {
        if hints {
            app.add_plugins(TuiPlugins);
        }
    });
    app.insert_resource(TuiTestScreen::new(100, 24));
    app.update();
    (app, agent)
}

#[test]
fn without_key_hints_the_input_keeps_its_band_and_the_keys_that_lead_out() {
    let (mut app, agent) = screen(false);
    // The band keeps its rows, with nothing over an empty input, idle or
    // with a turn running.
    assert_eq!(input(&app).len(), 3);
    assert_eq!(key_row(&app).trim(), "", "{:?}", input(&app));
    let turn = app.world_mut().spawn(TurnOf(agent)).id();
    app.update();
    assert_eq!(key_row(&app).trim(), "", "{:?}", input(&app));
    app.world_mut().despawn(turn);
    app.update();
    // Typing and Enter still send.
    type_text(&mut app, "hello");
    press(&mut app, "enter");
    assert_eq!(app.world().resource::<Delivered>().0, ["hello"]);
    press(&mut app, "ctrl+c");
    // An earlier prompt still says how to get back to the draft.
    app.world_mut().resource_mut::<PromptHistory>().entries = vec!["older".into()];
    type_text(&mut app, "my draft");
    press(&mut app, "ctrl+a up");
    assert_eq!(typed(&app), "older");
    assert!(
        key_row(&app).contains("Down back to your draft"),
        "{:?}",
        input(&app)
    );
    press(&mut app, "down");
    assert_eq!(typed(&app), "my draft");
    press(&mut app, "ctrl+c");
    // A question answered in the input still says its keys over an empty
    // answer, and is answered.
    let ask = Ask::new("PR title").initial("Fix the parser");
    let ask = app.world_mut().spawn((ask, ForAgent(agent))).id();
    app.update();
    press(&mut app, "ctrl+c");
    assert!(
        key_row(&app).contains("Enter answers") && key_row(&app).contains("Esc refuses"),
        "{:?}",
        input(&app)
    );
    type_text(&mut app, "Fix it");
    press(&mut app, "enter");
    assert!(app.world().get::<Outcome>(ask).is_some());
    // A panel with the keys still says how to give them back.
    let panel = TuiPanel::new(Placement::Top(ratatui::layout::Constraint::Length(3)));
    let panel = app.world_mut().spawn(panel).id();
    app.world_mut().resource_mut::<KeyFocus>().0 = Some(panel);
    app.update();
    assert!(
        key_row(&app).contains("Esc gives them back"),
        "{:?}",
        input(&app)
    );
}

#[test]
fn with_key_hints_an_empty_input_lists_its_keys() {
    let (mut app, agent) = screen(true);
    assert!(key_row(&app).contains("Enter sends"), "{:?}", input(&app));
    assert!(key_row(&app).contains("/ commands"), "{:?}", input(&app));
    app.world_mut().spawn(TurnOf(agent));
    app.update();
    assert!(key_row(&app).contains("Enter steers"), "{:?}", input(&app));
    assert!(key_row(&app).contains("Esc stops"), "{:?}", input(&app));
}

#[test]
fn a_narrow_key_row_keeps_what_matters_most_and_says_it_dropped_some() {
    for keys in [keys(true), keys(false)] {
        let all = fit_keys(keys, 500);
        let names: Vec<&str> = keys.iter().map(|(name, _)| *name).collect();
        assert_eq!(all, format!(" {} ", names.join(" · ")));
        let most = keys.iter().map(|(_, keep)| *keep).max().unwrap_or_default();
        let top = keys.iter().find(|(_, keep)| *keep == most);
        let top = top.map_or("", |(name, _)| *name);
        for width in format!(" {top} · … ").width()..all.width() {
            let row = fit_keys(keys, width);
            assert!(row.width() <= width, "{row}");
            assert!(row.trim_end().ends_with('…'), "{row}");
            // Nothing dropped matters more than something kept, and what is
            // kept keeps its order.
            let kept: Vec<&str> = row
                .trim()
                .split(" · ")
                .filter(|item| *item != "…")
                .collect();
            let rank = |item: &&str| {
                keys.iter()
                    .find(|(name, _)| name == item)
                    .map(|(_, keep)| *keep)
            };
            let lowest = kept.iter().filter_map(rank).min().unwrap_or_default();
            for (name, keep) in keys {
                assert!(
                    *keep < lowest || kept.contains(name),
                    "{name} at {width}: {row}"
                );
            }
            let order: Vec<&str> = names
                .iter()
                .copied()
                .filter(|name| kept.contains(name))
                .collect();
            assert_eq!(kept, order, "{width}");
        }
    }
    assert!(fit_keys(keys(true), 15).contains("Esc stops"));
}

#[test]
fn without_the_terminal_view_it_adds_nothing() {
    let mut app = App::new();
    app.add_plugins(super::KeyHintsPlugin);
    app.finish();
    assert!(app.world().get_resource::<super::KeyHints>().is_none());
}

/// Added before the view, or before [`TuiPlugins`], it still lists the keys.
#[test]
fn added_before_the_view_it_lists_the_keys() {
    use rig_harness::ask::AskPlugin;
    use rig_harness::pick::PickPlugin;

    let mut app = App::new();
    app.insert_resource(rig_harness::prelude::Invoked::default())
        .add_plugins((
            AgentPlugin,
            AskPlugin,
            PickPlugin,
            super::KeyHintsPlugin,
            TuiPlugins,
        ));
    app.finish();
    assert!(app.world().get_resource::<super::KeyHints>().is_some());
}
