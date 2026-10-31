use rig_core::message::CallId;
use rig_core::test_utils::MockCompletionModel;
use rig_harness::harness_protocol::SessionId;
use rig_harness::pick::PickPlugin;
use rig_harness_test_support::{
    Ended, app_in, calls, first_agent, log_ended, reply, run_until, scratch_home, scripted_models,
    tui_view,
};
use rig_tui::crossterm::event::{Event, KeyCode, KeyEvent, KeyModifiers};
use rig_tui::{DrawnRow, Editor, MainEditor};

use super::*;

/// What `cargo build` printed, an error among its lines.
const BUILD_OUTPUT: &str = "   Compiling app v0.1.0\n\
    warning: unused import\n\
    --> src/lib.rs:1:5\n\
    |\n\
    1 | use std::fmt;\n\
    error[E0425]: cannot find function `helper` in this scope\n\
    --> src/main.rs:2:5\n\
    error: could not compile `app`";

const FIRST_REPLY: &str =
    "Here is the entry point:\n\nfn main() {\n    run(1);\n}\nfn run(n: u32) {}";
const LAST_ASK: &str = "Add fn helper() and call it.";
const LAST_REPLY: &str = "Added `fn helper(x)`; `fn main()` calls it.";

/// A `shell` that prints the build's output.
fn shell(called: On<ToolCalled<serde_json::Value>>, mut calls: Query<&mut Decision, ToolCalls>) {
    if let Ok(mut call) = calls.get_mut(called.call) {
        call.decide(
            Decider::plugin("fh-transcript-search"),
            ToolResult::success(BUILD_OUTPUT.into()),
        );
    }
}

/// The scrolls asked for, in order.
#[derive(Resource, Default)]
struct Scrolls(Vec<(Option<TranscriptItem>, ItemRow)>);

fn note_scroll(scroll: On<ScrollTo>, mut scrolls: ResMut<Scrolls>) {
    scrolls.0.push((scroll.item.clone(), scroll.row));
}

/// The agent's turns the view stopped.
#[derive(Resource, Default)]
struct Interrupts(usize);

fn note_interrupt(_: On<StopTurn>, mut interrupts: ResMut<Interrupts>) {
    interrupts.0 += 1;
}

/// Ctrl+F, then `keys`, in one frame: the bar takes the keys after it.
fn search(app: &mut App, keys: impl IntoIterator<Item = KeyCode>) {
    let ctrl_f = KeyEvent::new(KeyCode::Char('f'), KeyModifiers::CONTROL);
    let mut events = app.world_mut().resource_mut::<PendingInput>();
    events.0.push(KeyInput::Event(Event::Key(ctrl_f)));
    press(app, keys);
}

fn input(app: &App) -> &str {
    let world = app.world();
    let main = world.try_query_filtered::<&Editor, With<MainEditor>>();
    let editor = main.and_then(|mut main| main.single(world).ok());
    editor.map_or("", Editor::text)
}

/// The main pane's view, as it is now.
fn main_view(app: &App) -> TranscriptView {
    let world = app.world();
    let main = world.try_query_filtered::<&TranscriptView, With<MainPane>>();
    let view = main.and_then(|mut main| main.single(world).ok());
    view.cloned().unwrap_or_default()
}

/// Changes the main pane's view.
fn edit_view(app: &mut App, edit: impl FnOnce(&mut TranscriptView)) {
    let world = app.world_mut();
    let mut main = world.query_filtered::<&mut TranscriptView, With<MainPane>>();
    let edited = main.single_mut(world).map(|mut view| edit(&mut view));
    assert!(edited.is_ok(), "the default screen has a main pane");
}

/// The terminal reports `keys`; one frame handles them.
fn press(app: &mut App, keys: impl IntoIterator<Item = KeyCode>) {
    for code in keys {
        let key = KeyEvent::new(code, KeyModifiers::NONE);
        let mut events = app.world_mut().resource_mut::<PendingInput>();
        events.0.push(KeyInput::Event(Event::Key(key)));
    }
    app.update();
}

fn typed(text: &str) -> Vec<KeyCode> {
    text.chars().map(KeyCode::Char).collect()
}

/// `agent`'s transcript as the view lays it out, `width` columns wide.
fn laid_out(app: &mut App, agent: Entity, width: u16) -> DrawnTranscript {
    let world = app.world_mut();
    let mut panes = world.query_filtered::<&TranscriptView, With<MainPane>>();
    let view = panes.single(world).ok().cloned();
    let drawn = view.and_then(|view| TranscriptLayout::lay_out(world, &view, agent, width));
    drawn.unwrap_or_default()
}

/// The matches lit in the rows of `drawn` with `text`.
fn lit(drawn: &DrawnTranscript, text: &str) -> usize {
    let rows = drawn.rows().iter().filter(|row| row.text.contains(text));
    rows.map(DrawnRow::lit_total).sum()
}

fn counter(app: &mut App) -> Option<String> {
    let world = app.world_mut();
    let mut bars = world.query::<&TranscriptSearch>();
    bars.iter(world).next().map(TranscriptSearch::counter)
}

#[test]
fn finds_an_error_in_a_tool_output_counts_regex_matches_and_scrolls_back() {
    let model = MockCompletionModel::from_stream_turns([
        reply(FIRST_REPLY),
        calls("c1", "shell", serde_json::json!({})),
        reply("The build failed: a name is missing."),
        reply(LAST_REPLY),
        reply("Still E0425."),
    ]);
    let home = scratch_home("transcript-search");
    let (mut app, wakes) = app_in(&home, &SessionId::generate());
    app.insert_resource(scripted_models([("test/model", &model)]))
        .add_plugins((
            AgentPlugin,
            SessionLogPlugin,
            PickPlugin,
            TranscriptSearchPlugin,
        ))
        .add_open_tool("shell", "Run a command.", ToolOptions::default(), shell)
        .init_resource::<Ended>()
        .init_resource::<Scrolls>()
        .init_resource::<Interrupts>()
        .add_observer(log_ended)
        .add_observer(note_scroll)
        .add_observer(note_interrupt);
    tui_view(&mut app);
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    let model = "test/model".to_owned();
    app.world_mut().trigger(SetModel {
        entity: agent,
        model,
        kind: ChoiceKind::ForThisRun,
    });
    let asks = ["Where does it start?", "Run cargo build.", LAST_ASK];
    for (turn, ask) in asks.into_iter().enumerate() {
        app.world_mut()
            .trigger(Deliver::new(agent, ask, DeliveryMode::Steer));
        run_until(&mut app, &wakes, |world| {
            world.resource::<Ended>().0.len() > turn
        });
    }
    assert_eq!(app.world().resource::<Ended>().0.len(), 3);
    app.world_mut().resource_mut::<ShownAgent>().0 = Some(agent);

    // Turn 2's `shell` output, with the error.
    let call = TranscriptItem::Call(CallId::from_wire("c1"));
    let whole = laid_out(&mut app, agent, u16::MAX);
    let rows = whole.rows_of(&call).unwrap_or_default();
    let before: Vec<&str> = whole
        .rows()
        .iter()
        .take(rows.start)
        .map(|row| row.text.as_str())
        .collect();
    assert!(before.iter().any(|row| row.contains("Run cargo build.")));
    assert!(before.iter().all(|row| !row.contains(LAST_ASK)));
    let row = whole.item(&call).and_then(|call| {
        call.rows()
            .iter()
            .position(|row| row.text.contains("E0425"))
    });

    // `/` on the input starts a slash command, also after typing ended
    // the reading of a selected item.
    rig_harness_test_support::press(&mut app, "/");
    assert_eq!((counter(&mut app), input(&app)), (None, "/"));
    edit_view(&mut app, |view| view.selected = Some(call.clone()));
    let keys = [KeyCode::Backspace, KeyCode::Char('x'), KeyCode::Backspace];
    press(&mut app, keys.into_iter().chain([KeyCode::Char('/')]));
    assert_eq!((counter(&mut app), input(&app)), (None, "/"));
    rig_harness_test_support::press(&mut app, "backspace");
    assert_eq!(input(&app), "");

    // With an item of the transcript selected, `/` opens the search.
    // Lower case finds it: the view lights the match and scrolls to its
    // row; Enter takes it.
    edit_view(&mut app, |view| view.selected = Some(call.clone()));
    rig_harness_test_support::press(&mut app, "/");
    assert_eq!(
        (counter(&mut app).as_deref(), input(&app)),
        (Some("0/0"), "")
    );
    let mut keys = typed("e0425");
    keys.push(KeyCode::Enter);
    press(&mut app, keys);
    assert_eq!(counter(&mut app).as_deref(), Some("1/1"));
    let view = main_view(&app);
    let spans = view.highlights.iter().flat_map(|lit| &lit.spans);
    let spans: Vec<Option<&TranscriptItem>> = spans.map(|span| span.item.as_ref()).collect();
    assert_eq!(spans, [Some(&call)]);
    // Each key searches as it is typed; the last scroll is to the match.
    let scrolls = &app.world().resource::<Scrolls>().0;
    let at = row.map(|row| (Some(call.clone()), ItemRow::Top(row)));
    assert_eq!(scrolls.last(), at.as_ref());
    let scrolled = scrolls.len();
    let drawn = laid_out(&mut app, agent, u16::MAX);
    assert_eq!(drawn.lit_in(&call), 1);

    // Esc after Enter: the view stays at the match, unlit; the bar took
    // the keys, so none was typed and Esc stopped nothing. A second Esc
    // gives the keys back to the input, still stopping nothing.
    press(&mut app, [KeyCode::Esc]);
    assert_eq!(counter(&mut app), None);
    let view = main_view(&app);
    assert!(view.highlights.is_empty());
    assert_eq!(app.world().resource::<Scrolls>().0.len(), scrolled);
    press(&mut app, [KeyCode::Esc]);
    let view = main_view(&app);
    assert_eq!(view.selected, None);
    assert_eq!(
        (input(&app), app.world().resource::<Interrupts>().0),
        ("", 0)
    );

    // A capital letter makes the search match case. What arrives while
    // searching is searched too, with no key pressed.
    search(&mut app, typed("E0425"));
    assert_eq!(counter(&mut app).as_deref(), Some("1/1"));
    app.world_mut()
        .trigger(Deliver::new(agent, "Again?", DeliveryMode::Steer));
    run_until(&mut app, &wakes, |world| {
        world.resource::<Ended>().0.len() > 3
    });
    app.update();
    assert_eq!(counter(&mut app).as_deref(), Some("2/2"));
    press(&mut app, [KeyCode::Esc]);

    // A regular expression counts what the same expression finds in the
    // rendered transcript.
    search(&mut app, typed(r"re:fn \w+\("));
    let whole = laid_out(&mut app, agent, u16::MAX);
    let reference = Regex::new(r"fn \w+\(").map_or(0, |regex| {
        let rows = whole.rows().iter();
        rows.map(|row| regex.find_iter(&row.text).count()).sum()
    });
    assert_eq!(reference, 5, "{whole:#?}");
    let counted = format!("{reference}/{reference}");
    assert_eq!(counter(&mut app), Some(counted));
    assert_eq!(lit(&whole, ""), reference);
    press(&mut app, [KeyCode::Esc]);

    // The view lights what the expression matched, not each place the
    // text it matched is: the last reply's `fn main` starts no line.
    search(&mut app, typed(r"re:^fn \w+"));
    assert_eq!(counter(&mut app).as_deref(), Some("1/1"));
    let drawn = laid_out(&mut app, agent, u16::MAX);
    assert_eq!((lit(&drawn, "fn run(n"), lit(&drawn, "calls it")), (1, 0));
    press(&mut app, [KeyCode::Esc]);

    // Laid out narrow, the error's line wraps inside the query: the
    // search rejoins the line to find it, at the row the view counts it
    // in, and the view lights it across both rows.
    let split = "cannot find function";
    search(&mut app, typed(split));
    assert_eq!(counter(&mut app).as_deref(), Some("1/1"));
    let narrow = laid_out(&mut app, agent, 24);
    assert!(narrow.rows().iter().all(|row| !row.text.contains(split)));
    assert_eq!(narrow.lit_in(&call), 1);
    let hits = Regex::new(split).map_or_else(|_| Vec::new(), |regex| find(&regex, agent, &narrow));
    let narrow = narrow.item(&call).unwrap_or_default();
    let lit = narrow.rows().iter().position(|row| row.lit_total() > 0);
    let starts = |hit: &Hit| hit.item() == Some(&call) && Some(hit.row) == lit;
    assert!(
        matches!(hits.as_slice(), [hit] if starts(hit)),
        "{narrow:#?}"
    );

    // Esc without Enter: the view scrolls back to where it was, following
    // the newest output.
    press(&mut app, [KeyCode::Esc]);
    let view = main_view(&app);
    assert!(view.highlights.is_empty());
    let last = app.world().resource::<Scrolls>().0.last().cloned();
    assert_eq!(last, Some((None, ItemRow::Top(0))));
    assert_eq!(app.world().resource::<ShownAgent>().0, Some(agent));
}
