use rig_core::message::CallId;
use rig_core::test_utils::MockCompletionModel;
use rig_harness::harness_protocol::SessionId;
use rig_harness::pick::PickPlugin;
use rig_harness_test_support::{
    Ended, app_in, calls, first_agent, log_ended, reply, run_until, scratch_home, scripted_models,
};
use rig_tui::crossterm::event::{Event, KeyCode, KeyEvent, KeyModifiers};

use super::*;

/// What `cargo build` printed: the error is past the lines the transcript
/// shows of a result.
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

/// Ctrl+F, then `keys` in one frame.
fn search(app: &mut App, agent: Entity, keys: impl IntoIterator<Item = KeyCode>) {
    let world = app.world_mut();
    let mut bindings = world.query::<(Entity, &KeyBinding)>();
    let binding = bindings
        .iter(world)
        .find(|(_, binding)| binding.label == "Search the transcript")
        .map_or(Entity::PLACEHOLDER, |(entity, _)| entity);
    world.trigger(KeyPressed {
        entity: binding,
        agent: Some(agent),
        released: false,
    });
    app.update();
    press(app, keys);
}

/// The terminal reports `keys`; one frame handles them.
fn press(app: &mut App, keys: impl IntoIterator<Item = KeyCode>) {
    for code in keys {
        let key = KeyEvent::new(code, KeyModifiers::NONE);
        app.world_mut()
            .write_message(TerminalEvent(Event::Key(key)));
    }
    app.update();
}

fn typed(text: &str) -> Vec<KeyCode> {
    text.chars().map(KeyCode::Char).collect()
}

fn counter(app: &mut App) -> Option<String> {
    let world = app.world_mut();
    let mut bars = world.query::<&TranscriptSearch>();
    bars.iter(world).next().map(TranscriptSearch::counter)
}

#[test]
fn finds_an_error_in_a_cut_tool_output_counts_regex_matches_and_scrolls_back() {
    let model = MockCompletionModel::from_stream_turns([
        reply(FIRST_REPLY),
        calls("c1", "shell", serde_json::json!({})),
        reply("The build failed: a name is missing."),
        reply(LAST_REPLY),
    ]);
    let home = scratch_home("transcript-search");
    let (mut app, wakes) = app_in(&home, &SessionId::generate());
    app.insert_resource(scripted_models([("test/model", &model)]))
        .add_plugins((
            AgentPlugin,
            JournalPlugin,
            PickPlugin,
            TranscriptSearchPlugin,
        ))
        .add_open_tool("shell", "Run a command.", ToolOptions::default(), shell)
        .init_resource::<Ended>()
        .init_resource::<Scrolls>()
        .add_observer(log_ended)
        .add_observer(note_scroll);
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    let model = "test/model".to_owned();
    app.world_mut().trigger(SetModel {
        entity: agent,
        model,
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
    app.world_mut().resource_mut::<Shown>().0 = Some(agent);

    // Lower case finds the code in turn 2's cut `shell` output: the view
    // goes there, expands the call and lights the match; Enter takes it.
    let mut keys = typed("e0425");
    keys.push(KeyCode::Enter);
    search(&mut app, agent, keys);
    assert_eq!(counter(&mut app).as_deref(), Some("1/1"));
    let call = TranscriptItem::Call(CallId::from_wire("c1"));
    let view = app.world().resource::<TranscriptView>();
    assert!(view.expanded.contains(&call));
    let lit: Vec<(&str, bool)> = view
        .highlights
        .iter()
        .map(|lit| (lit.text.as_str(), lit.match_case))
        .collect();
    assert_eq!(lit, [("e0425", false)]);
    let scrolls = &app.world().resource::<Scrolls>().0;
    // No frame drew it (no terminal here): the view reveals the call,
    // which a frame then draws whole, and the search scrolls to its row.
    assert_eq!(scrolls.as_slice(), [(Some(call.clone()), ItemRow::Reveal)]);

    // Esc after Enter: the view stays at the match, unlit.
    press(&mut app, [KeyCode::Esc]);
    assert_eq!(counter(&mut app), None);
    let view = app.world().resource::<TranscriptView>();
    assert!(view.highlights.is_empty() && view.expanded.contains(&call));
    assert_eq!(app.world().resource::<Scrolls>().0.len(), 1);
    app.world_mut()
        .resource_mut::<TranscriptView>()
        .expanded
        .clear();

    // A capital letter makes the search, and what it lights, match case.
    search(&mut app, agent, typed("E0425"));
    assert_eq!(counter(&mut app).as_deref(), Some("1/1"));
    let lit = app.world().resource::<TranscriptView>().highlights.clone();
    assert!(lit.iter().all(|lit| lit.text == "E0425" && lit.match_case));
    press(&mut app, [KeyCode::Esc]);

    // A regular expression counts what the same expression finds in the
    // texts of the session.
    search(&mut app, agent, typed(r"re:fn \w+\("));
    let reference = Regex::new(r"fn \w+\(").map_or(0, |regex| {
        [
            asks.join("\n"),
            FIRST_REPLY.to_owned(),
            BUILD_OUTPUT.to_owned(),
            LAST_REPLY.to_owned(),
        ]
        .iter()
        .flat_map(|text| text.lines())
        .map(|line| regex.find_iter(line).count())
        .sum()
    });
    assert_eq!(reference, 5);
    let counted = format!("{reference}/{reference}");
    assert_eq!(counter(&mut app), Some(counted));

    // Esc without Enter: what the search expanded collapses and the view
    // scrolls back to where it was, following the newest output.
    press(&mut app, [KeyCode::Esc]);
    let view = app.world().resource::<TranscriptView>();
    assert!(view.highlights.is_empty() && view.expanded.is_empty());
    let last = app.world().resource::<Scrolls>().0.last().cloned();
    assert_eq!(last, Some((None, ItemRow::Top(0))));
    assert_eq!(app.world().resource::<Shown>().0, Some(agent));
}
