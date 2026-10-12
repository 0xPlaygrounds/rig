use rig_core::test_utils::MockCompletionModel;
use rig_harness::harness_protocol::SessionId;
use rig_harness::pick::PickPlugin;
use rig_harness_test_support::{
    Ended, app_in, calls, first_agent, log_ended, reply, run_until, scratch_home, scripted_models,
};
use rig_tui::RESULT_LINES;
use rig_tui::crossterm::event::{Event, KeyCode, KeyEvent, KeyModifiers};
use rig_tui::ratatui::style::Color;

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
        call.decide(ToolResult::success(BUILD_OUTPUT.into()));
    }
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

fn bar(app: &mut App) -> Option<(String, Option<(Block, Hit)>, Vec<Line<'static>>)> {
    let world = app.world_mut();
    let mut bars = world.query::<&TranscriptSearch>();
    let search = bars.iter(world).next()?;
    let current = search
        .current()
        .map(|(block, hit)| (block.clone(), hit.clone()));
    Some((search.counter(), current, search.context(7, 80)))
}

#[test]
fn finds_an_error_in_a_cut_tool_output_counts_regex_matches_and_closes() {
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
        .add_observer(log_ended);
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

    // Ctrl+F opens the bar with the keys.
    let world = app.world_mut();
    let mut bindings = world.query::<(Entity, &KeyBinding)>();
    let binding = bindings
        .iter(world)
        .find(|(_, binding)| binding.label == "Search the transcript")
        .map(|(entity, _)| entity);
    assert!(binding.is_some());
    world.trigger(KeyPressed {
        entity: binding.unwrap_or(Entity::PLACEHOLDER),
        agent: Some(agent),
        released: false,
    });
    app.update();
    assert!(app.world().resource::<KeyFocus>().0.is_some());

    // Lower case finds the code in the cut output of turn 2's `shell`, and
    // Enter takes it.
    let mut keys = typed("e0425");
    keys.push(KeyCode::Enter);
    press(&mut app, keys);
    let shown = bar(&mut app);
    assert!(shown.is_some(), "the bar is open");
    let Some((counter, current, context)) = shown else {
        return;
    };
    assert_eq!(counter, "1/1");
    assert!(current.is_some(), "a match is taken");
    let Some((block, hit)) = current else {
        return;
    };
    assert_eq!(block.source, Source::Output("shell".to_owned()));
    assert_eq!(block.turn, 2);
    assert!(hit.line >= RESULT_LINES, "past the lines shown: {hit:?}");
    let lit: Vec<String> = context
        .iter()
        .flat_map(|line| &line.spans)
        .filter(|span| span.style.bg == Some(Color::Yellow))
        .map(|span| span.content.to_string())
        .collect();
    assert_eq!(lit, ["E0425"]);

    // A regular expression counts what the same expression finds in the
    // texts of the session.
    let mut keys = vec![KeyCode::Backspace; 5];
    keys.extend(typed(r"re:fn \w+\("));
    press(&mut app, keys);
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
    let counter = bar(&mut app).map(|(counter, ..)| counter);
    assert_eq!(counter, Some(format!("{reference}/{reference}")));

    // Esc without Enter closes it and leaves the transcript as it was.
    press(&mut app, [KeyCode::Esc]);
    assert!(bar(&mut app).is_none());
    assert_eq!(app.world().resource::<Shown>().0, Some(agent));
}
