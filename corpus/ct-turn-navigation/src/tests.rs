use std::sync::mpsc::Receiver;

use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness_test_support::{
    Ended, MemoryStore, app_on, calls, log_ended, press, reply, run_until, scripted_models,
    tui_view,
};
use rig_tui::{Focus, TuiTestScreen};

use super::*;

/// What the `dump` tool returns: a long output.
fn dump(called: On<ToolCalled<serde_json::Value>>, mut calls: Query<&mut Decision, ToolCalls>) {
    let output: Vec<String> = (1..=40).map(|n| format!("output line {n}")).collect();
    if let Ok(mut call) = calls.get_mut(called.call) {
        call.decide(
            Decider::plugin("dump"),
            ToolResult::success(output.join("\n").into()),
        );
    }
}

/// A turn of the model: a `dump` call, then a long answer.
fn turn(n: usize) -> [Vec<MockStreamEvent>; 2] {
    let answer: Vec<String> = (1..=30)
        .map(|line| format!("answer {n}, line {line}"))
        .collect();
    [
        calls(&format!("c{n}"), "dump", serde_json::json!({})),
        reply(&answer.join("\n\n")),
    ]
}

/// The prompt of turn `n`, wider than a narrow terminal.
fn prompt(n: usize) -> String {
    format!("prompt {n}: now make it work on Windows, and keep the Linux build green too")
}

/// Sends prompt `n` as the user typed it and runs its turn to its end.
fn ask(app: &mut App, wakes: &Receiver<()>, agent: Entity, n: usize) {
    let typed = Deliver::new(agent, prompt(n), DeliveryMode::Queue).with_author(Author::user());
    app.world_mut().trigger(typed);
    run_until(app, wakes, |world| world.resource::<Ended>().0.len() >= n);
    // The frame after the turn draws it; the next one shows its header.
    app.update();
    app.update();
}

/// The turn the header shows, if it shows.
fn header(app: &App) -> Option<usize> {
    let header = &app.world().resource::<PromptHeader>().0;
    header.as_ref().map(|header| header.turn)
}

/// The screen's first row: the header's, while it shows.
fn first_row(app: &App) -> String {
    let rows = app.world().resource::<TuiTestScreen>().rows();
    rows.first().cloned().unwrap_or_default()
}

/// The first row prompt `n` (from 1) is drawn at, and the top row.
fn rows(app: &App, agent: Entity, n: usize) -> (Option<usize>, usize) {
    let drawn = app.world().resource::<DrawnTranscript>();
    let conversation = app.world().get::<Conversation>(agent);
    let ids = conversation
        .map(|conversation| turns(conversation))
        .unwrap_or_default();
    let id = ids.get(n - 1).map(|(id, _)| *id);
    let start = id.and_then(|id| drawn.rows_of(&TranscriptItem::Message(id)));
    (start.map(|rows| rows.start), drawn.on_screen().start)
}

/// Presses `keys` and runs a frame more, for the header it noted.
fn press_and_show(app: &mut App, keys: &str) {
    press(app, keys);
    app.update();
}

#[test]
fn keys_jump_between_prompts_and_the_header_follows_the_top_row() {
    let script = (1..=4)
        .flat_map(turn)
        .chain([reply("streamed answer\n\nmore")]);
    let model = MockCompletionModel::from_stream_turns(script);
    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.insert_resource(Settings::default())
        .insert_resource(scripted_models([("test/model", &model)]))
        .add_plugins((AgentPlugin, JournalPlugin))
        .add_open_tool("dump", "Print a lot.", ToolOptions::default(), dump)
        .init_resource::<Ended>()
        .add_observer(log_ended)
        .insert_resource(TuiTestScreen::new(120, 24));
    tui_view(&mut app);
    app.add_plugins(TurnNavigationPlugin);
    let agent = app
        .world_mut()
        .spawn((Agent, ModelChoice("test/model".to_owned())))
        .id();
    app.finish();
    app.cleanup();
    app.update();
    app.world_mut().trigger(Focus { entity: agent });
    for n in 1..=4 {
        ask(&mut app, &wakes, agent, n);
    }
    // At the live bottom: no header.
    assert_eq!(header(&app), None);
    assert!(!first_row(&app).starts_with('▸'), "{}", first_row(&app));

    // Twice back from the bottom: turn 3's prompt is the top row.
    press_and_show(&mut app, "ctrl+up");
    assert_eq!(header(&app), Some(4));
    press_and_show(&mut app, "ctrl+up");
    let (third, top) = rows(&app, agent, 3);
    assert_eq!(third, Some(top));
    assert_eq!(header(&app), Some(3));
    let shown = first_row(&app);
    assert!(
        shown.starts_with("▸ 3 · prompt 3: now make it work"),
        "{shown}"
    );
    assert!(shown.contains("1 tool"), "{shown}");

    // Down by rows: the header turns to 4 exactly at turn 4's prompt.
    let (fourth, top) = rows(&app, agent, 4);
    let fourth = fourth.unwrap_or_default();
    assert!(fourth > top + 1, "turn 3 is longer than a row");
    let to_boundary = vec!["shift+down"; fourth - top - 1].join(" ");
    press_and_show(&mut app, &to_boundary);
    assert_eq!(rows(&app, agent, 4), (Some(fourth), fourth - 1));
    assert_eq!(header(&app), Some(3));
    press_and_show(&mut app, "shift+down");
    assert_eq!(rows(&app, agent, 4), (Some(fourth), fourth));
    assert_eq!(header(&app), Some(4));
    assert!(first_row(&app).starts_with("▸ 4 ·"), "{}", first_row(&app));

    // A turn streams while the view is scrolled up: it stays, marked.
    ask(&mut app, &wakes, agent, 5);
    assert_eq!(rows(&app, agent, 4), (Some(fourth), fourth));
    assert_eq!(header(&app), Some(4));
    let screen = app.world().resource::<TuiTestScreen>().rows();
    assert!(
        screen.iter().any(|row| row.contains("↓ new output")),
        "{screen:#?}"
    );
    press_and_show(&mut app, "ctrl+end");
    assert_eq!(header(&app), None);
    assert!(app.world().resource::<TranscriptView>().top().is_none());
    assert!(!first_row(&app).starts_with('▸'));

    // On turn 2, narrowed from 120 to 40 columns: still turn 2, one row.
    press_and_show(&mut app, "ctrl+home");
    assert_eq!(header(&app), Some(1));
    press_and_show(&mut app, "ctrl+down");
    assert_eq!(header(&app), Some(2));
    app.insert_resource(TuiTestScreen::new(40, 24));
    app.world_mut().write_message(RequestRedraw);
    app.update();
    app.update();
    let (second, top) = rows(&app, agent, 2);
    assert_eq!(second, Some(top));
    assert_eq!(header(&app), Some(2));
    let shown = first_row(&app);
    assert!(shown.starts_with("▸ 2 · prompt 2"), "{shown}");
    assert!(
        shown.contains('…') && shown.trim_end().ends_with("1 tool"),
        "{shown}"
    );
}
