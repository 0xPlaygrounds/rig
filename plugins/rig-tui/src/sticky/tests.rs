use bevy_app::prelude::*;
use ratatui::layout::{Constraint, Position};
use ratatui::style::Modifier;
use rig_core::completion::Message;
use rig_core::conversation::Entry;
use rig_core::message::{CallId, ToolName, ToolResultContent, UserContent};
use rig_ecs::agent::Conversation;
use rig_ecs::prelude::{Author, MessageId};
use rig_harness::prelude::*;

use super::StickyPromptPlugin;
use crate::tests::{drawn, keep_drawn, main_pane, press, say, view_app_of, view_app_with};
use crate::{
    ItemRow, KeyFocus, Placement, ScrollTo, Theme, TranscriptItem, TranscriptPane,
    TranscriptSource, TuiPanel, TuiPlugin, TuiPlugins, TuiTestScreen,
};

/// A multi-line assistant reply of `lines` numbered paragraphs.
fn long_reply(prefix: &str, lines: usize) -> Message {
    let body: Vec<String> = (1..=lines).map(|n| format!("{prefix} line {n}")).collect();
    Message::assistant(body.join("\n\n"))
}

/// The top row drawn on the test screen, trimmed at the right.
fn top_row(app: &App) -> String {
    app.world()
        .resource::<TuiTestScreen>()
        .rows()
        .first()
        .map(|row| row.trim_end().to_owned())
        .unwrap_or_default()
}

/// The main pane's view on a `width` by `height` screen, with `TuiPlugins`
/// when `sticky` is true, or bare `TuiPlugin` otherwise.
fn screen(sticky: bool, width: u16, height: u16) -> (App, Entity) {
    let (mut app, agent) = view_app_of(|app| {
        if sticky {
            app.add_plugins(TuiPlugins);
        } else {
            app.add_plugins(TuiPlugin);
        }
    });
    app.insert_resource(TuiTestScreen::new(width, height));
    keep_drawn(&mut app);
    app.update();
    (app, agent)
}

/// The row range of message `id` in the main pane's drawn transcript.
fn message_rows(app: &App, id: MessageId) -> Option<std::ops::Range<usize>> {
    drawn(app).rows_of(&TranscriptItem::Message(id))
}

#[test]
fn shows_over_top_row_only_when_user_message_scrolls_off_screen() {
    let (mut app, agent) = screen(true, 60, 12);
    say(&mut app, agent, Message::user("fix the parser"));
    say(&mut app, agent, Message::assistant("short reply"));
    // The prompt is visible on screen: nothing overlaid on row 0.
    assert_eq!(top_row(&app), "");

    // Push a long reply so the user prompt scrolls above the viewport.
    say(&mut app, agent, long_reply("detail", 30));
    assert_eq!(top_row(&app), "› fix the parser");
    let theme = *app.world().resource::<Theme>();
    let cell = app
        .world()
        .resource::<TuiTestScreen>()
        .buffer()
        .cell(Position::new(0, 0));
    assert_eq!(cell.map(|c| c.bg), Some(theme.user));

    // Scroll back up to the top: the prompt is visible again, so no overlay.
    let pane = main_pane(app.world());
    let first_id = app.world().get::<Conversation>(agent).and_then(|c| c.id(0));
    assert!(first_id.is_some());
    app.world_mut().trigger(ScrollTo {
        pane,
        item: first_id.map(TranscriptItem::Message),
        row: ItemRow::Top(0),
    });
    app.update();
    assert_eq!(top_row(&app), "");
}

#[test]
fn switches_between_turns_and_hides_when_prompt_reaches_top() {
    let (mut app, agent) = screen(true, 60, 14);
    say(&mut app, agent, Message::user("first question"));
    say(&mut app, agent, long_reply("first reply", 25));
    say(&mut app, agent, Message::user("second question"));
    say(&mut app, agent, long_reply("second reply", 25));

    let first_id = app.world().get::<Conversation>(agent).and_then(|c| c.id(0));
    let second_id = app.world().get::<Conversation>(agent).and_then(|c| c.id(2));
    assert!(first_id.is_some() && second_id.is_some());
    let second_rows = second_id
        .and_then(|id| message_rows(&app, id))
        .unwrap_or_default();
    assert!(second_rows.len() >= 2, "{second_rows:?}");

    // At the bottom of turn 2: turn 2's prompt is pinned.
    assert_eq!(top_row(&app), "› second question");

    // Scroll up until turn 2's prompt is at the top of the screen:
    // its blank separator row and its text row both hide the overlay.
    let pane = main_pane(app.world());
    app.world_mut().trigger(ScrollTo {
        pane,
        item: second_id.map(TranscriptItem::Message),
        row: ItemRow::Top(0),
    });
    app.update();
    assert_eq!(drawn(&app).on_screen().start, second_rows.start);
    assert_eq!(top_row(&app), "");

    // One row down puts the prompt text itself on row 0: still visible, no overlay.
    press(&mut app, "shift+down");
    assert_eq!(drawn(&app).on_screen().start, second_rows.start + 1);
    assert_eq!(top_row(&app), "› second question");

    // One more row down scrolls turn 2's prompt completely off screen: overlay returns.
    press(&mut app, "shift+down");
    assert_eq!(drawn(&app).on_screen().start, second_rows.end);
    assert_eq!(top_row(&app), "› second question");

    // Scroll up into turn 1's reply (above turn 2's separator row): turn 1's prompt pins.
    press(&mut app, "shift+up shift+up shift+up");
    assert!(drawn(&app).on_screen().start < second_rows.start);
    assert_eq!(top_row(&app), "› first question");

    // Scroll all the way back to turn 1's prompt: hidden again.
    app.world_mut().trigger(ScrollTo {
        pane,
        item: first_id.map(TranscriptItem::Message),
        row: ItemRow::Top(0),
    });
    app.update();
    assert_eq!(top_row(&app), "");
}

#[test]
fn hides_while_multiline_prompt_is_partially_visible_and_collapses_when_off_screen() {
    let (mut app, agent) = screen(true, 40, 14);
    say(
        &mut app,
        agent,
        Message::user("line one\n\n\tline two\nline three with a very long tail"),
    );
    say(&mut app, agent, long_reply("answer", 30));

    let id = app.world().get::<Conversation>(agent).and_then(|c| c.id(0));
    assert!(id.is_some());
    let rows = id.and_then(|id| message_rows(&app, id)).unwrap_or_default();
    assert!(rows.len() >= 4, "{rows:?}");

    // Scroll so only the last row of the prompt is still visible on row 0.
    let pane = main_pane(app.world());
    app.world_mut().trigger(ScrollTo {
        pane,
        item: id.map(TranscriptItem::Message),
        row: ItemRow::Top(0),
    });
    app.update();
    let steps = vec!["shift+down"; rows.len() - 1].join(" ");
    press(&mut app, &steps);
    assert_eq!(drawn(&app).on_screen().start, rows.end - 1);
    assert!(
        !top_row(&app).starts_with("› line one"),
        "prompt still partially visible: {}",
        top_row(&app)
    );

    // One more row down scrolls the last line off screen: collapsed & shortened.
    press(&mut app, "shift+down");
    assert_eq!(drawn(&app).on_screen().start, rows.end);
    let shown = top_row(&app);
    assert!(shown.starts_with("› line one line two "), "{shown}");
    assert!(shown.ends_with('…'), "{shown}");
    assert!(
        unicode_width::UnicodeWidthStr::width(shown.as_str()) <= 40,
        "{shown}"
    );
}

#[test]
fn skips_non_user_and_tool_result_messages() {
    let (mut app, agent) = screen(true, 60, 12);
    say(&mut app, agent, Message::user("real user prompt"));
    say(&mut app, agent, long_reply("reply", 15));
    // A plugin-authored message and a tool-result-with-note message must not replace it.
    if let Some(mut conversation) = app.world_mut().get_mut::<Conversation>(agent) {
        conversation.push(Entry::new(Message::user("plugin reminder")).by(Author::plugin("hook")));
        if let Ok(shell) = ToolName::new("shell") {
            conversation.push(Message::User {
                content: vec![
                    UserContent::tool_result(
                        CallId::from_wire("c1"),
                        shell,
                        vec![ToolResultContent::text("ok")],
                    ),
                    UserContent::text("compaction note"),
                ],
            });
        }
    }
    say(&mut app, agent, long_reply("after", 20));
    assert_eq!(top_row(&app), "› real user prompt");
}

#[test]
fn uses_reversed_style_without_color() {
    let (mut app, agent) = screen(true, 60, 12);
    let plain = app.world().resource::<Theme>().without_color();
    app.insert_resource(plain);
    say(&mut app, agent, Message::user("plain prompt"));
    say(&mut app, agent, long_reply("reply", 30));

    assert_eq!(top_row(&app), "› plain prompt");
    let cell = app
        .world()
        .resource::<TuiTestScreen>()
        .buffer()
        .cell(Position::new(0, 0));
    assert!(
        cell.is_some_and(|c| c.modifier.contains(Modifier::REVERSED)),
        "{cell:?}"
    );
}

#[test]
fn titled_pane_keeps_its_title_row_and_one_row_pane_skips_overlay() {
    let (mut app, _) = view_app_with(|app| {
        app.add_plugins(StickyPromptPlugin);
    });
    app.insert_resource(TuiTestScreen::new(60, 20));
    let mut saved = rig_core::conversation::Conversation::default();
    saved.push(Message::user("helper task"));
    saved.push(long_reply("step", 25));

    let titled = app
        .world_mut()
        .spawn((
            TranscriptPane::new(TranscriptSource::Value(saved.clone())).titled("helper"),
            TuiPanel::new(Placement::Top(Constraint::Length(6))),
        ))
        .id();
    let tiny = app
        .world_mut()
        .spawn((
            TranscriptPane::new(TranscriptSource::Value(saved)),
            TuiPanel::new(Placement::Top(Constraint::Length(1))),
        ))
        .id();
    app.world_mut().resource_mut::<KeyFocus>().0 = Some(titled);
    app.update();

    let rows = app.world().resource::<TuiTestScreen>().rows();
    // Row 0 is the pane title; row 1 is the sticky prompt inside the pane.
    assert_eq!(rows.first().map(|r| r.trim_end()), Some(" helper"));
    assert_eq!(rows.get(1).map(|r| r.trim_end()), Some("› helper task"));
    // The 1-row pane below it (row 6) keeps its transcript text instead of overlaying.
    let tiny_row = rows.get(6).map(|r| r.trim_end()).unwrap_or_default();
    assert!(!tiny_row.starts_with("› "), "entity {tiny}: {tiny_row}");
}

#[test]
fn bare_tui_plugin_draws_no_sticky_prompt() {
    let (mut app, agent) = screen(false, 60, 12);
    say(&mut app, agent, Message::user("first prompt"));
    say(&mut app, agent, long_reply("reply", 30));
    assert!(!top_row(&app).starts_with("› "), "{}", top_row(&app));
}

#[test]
fn without_the_terminal_view_it_adds_nothing() {
    let mut app = App::new();
    app.add_plugins(StickyPromptPlugin);
    app.finish();
    app.update();
}

#[test]
fn added_before_the_view_it_still_draws_sticky_prompt() {
    let (mut app, agent) = view_app_of(|app| {
        app.add_plugins((StickyPromptPlugin, TuiPlugin));
    });
    app.insert_resource(TuiTestScreen::new(60, 12));
    say(&mut app, agent, Message::user("added first"));
    say(&mut app, agent, long_reply("reply", 30));
    assert_eq!(top_row(&app), "› added first");
}
