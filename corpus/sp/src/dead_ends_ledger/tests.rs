use std::fs;
use std::path::Path;
use std::sync::mpsc::Receiver;

use rig_basics::coding_tools::{EditTool, ShellTool};
use rig_basics::compaction::CompactionPlugin;
use rig_core::catalog::{Catalog, Connector};
use rig_core::completion::{CompletionRequest, Message};
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness::harness_protocol::SessionId;
use rig_harness_test_support::{
    Ended, ToolOutputs, app_in, calls, connect, first_agent, log_ended, log_tool_outputs, reply,
    run_until, scratch_home,
};

use super::*;

/// The fixture: `make test` prints `E42 boom` and exits 2.
const MAKEFILE: &str = "test:\n\t@echo 'E42 boom'; exit 2\n";

/// The kernel, the shell and edit tools, compaction and the ledger on
/// session `id` of `home`, in `project`, after its first frame: for a new
/// session with an agent `main`.
fn start(
    home: &Home,
    id: &SessionId,
    project: &Path,
    new: bool,
) -> Option<(App, Receiver<()>, Entity)> {
    let (mut app, wakes) = app_in(home, id);
    if new {
        app.world_mut()
            .spawn((Name::new("agent"), Agent, AgentId("main".to_owned())));
    }
    app.insert_resource(ProjectRoot(project.to_path_buf()))
        .insert_resource(Models(Connector::new(Catalog::default())))
        .add_plugins((
            AgentPlugin,
            SessionLogPlugin,
            EditTool,
            ShellTool,
            CompactionPlugin,
            DeadEndsLedgerPlugin,
        ))
        .init_resource::<Ended>()
        .add_observer(log_ended)
        .init_resource::<ToolOutputs>()
        .add_observer(log_tool_outputs);
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut())?;
    Some((app, wakes, agent))
}

/// The user says `text`; the turn runs to its end.
fn say(app: &mut App, wakes: &Receiver<()>, agent: Entity, text: &str) {
    let ended = app.world().resource::<Ended>().0.len();
    app.world_mut()
        .trigger(Deliver::new(agent, text, DeliveryMode::Steer));
    run_until(app, wakes, |world| {
        world.resource::<Ended>().0.len() > ended
    });
    app.update();
}

/// A `shell` call of `make test`.
fn make_test(id: &str) -> Vec<MockStreamEvent> {
    calls(id, "shell", serde_json::json!({ "command": "make test" }))
}

/// A request's system prompt and the rest of its messages, as JSON text.
fn parts(request: Option<&CompletionRequest>) -> (String, String) {
    let Some(request) = request else {
        return (String::new(), String::new());
    };
    let (system, rest): (Vec<&Message>, Vec<&Message>) = request
        .chat_history
        .iter()
        .map(|message| &**message)
        .partition(|message| matches!(message, Message::System { .. }));
    let text = |messages: Vec<&Message>| serde_json::to_string(&messages).unwrap_or_default();
    (text(system), text(rest))
}

/// The agent's entries' sources and approaches.
fn ledger(app: &App, agent: Entity) -> Vec<(Source, String, u64)> {
    let entries = app
        .world()
        .get::<DeadEnds>(agent)
        .map(|ledger| ledger.entries.clone());
    let entries = entries.unwrap_or_default().into_iter();
    entries
        .map(|entry| (entry.source, entry.approach, entry.turn))
        .collect()
}

/// The output of tool call `id`.
fn output(app: &App, id: &str) -> String {
    let outputs = &app.world().resource::<ToolOutputs>().0;
    outputs.get(id).cloned().unwrap_or_default()
}

#[test]
fn dead_ends_are_recorded_warned_kept_through_compaction_and_resumed() {
    let home = scratch_home("corpus-sp-dead-ends-ledger");
    let project = scratch_home("corpus-sp-dead-ends-ledger-project")
        .root()
        .to_path_buf();
    assert!(fs::create_dir_all(&project).is_ok(), "a scratch project");
    assert!(fs::write(project.join("Makefile"), MAKEFILE).is_ok());
    let source = project.join("lib.c");
    assert!(fs::write(&source, "int x = 1;\n").is_ok());
    let edit = serde_json::json!({
        "path": source.display().to_string(),
        "edits": [{ "old_text": "int x = 1;", "new_text": "int x = 2;" }],
    });
    let pin = serde_json::json!({ "approach": "pin libfoo 0.3", "why": "API removed" });
    let model = MockCompletionModel::from_stream_turns([
        make_test("1"),
        reply("It fails with E42."),
        make_test("2"),
        reply("Same failure."),
        make_test("3"),
        reply("Still failing."),
        calls("4", "edit", edit),
        make_test("5"),
        reply("Edited; still failing."),
        calls("6", DEAD_END, pin),
        reply("Recorded."),
        reply("## Goal\nMake the tests pass."),
        reply("I will try another route."),
    ]);
    let id: SessionId = match "1000-4".parse() {
        Ok(id) => id,
        Err(_) => return,
    };
    let started = start(&home, &id, &project, true);
    assert!(started.is_some(), "a new session with an agent");
    let Some((mut app, wakes, agent)) = started else {
        return;
    };
    let connection = connect(&model);
    assert!(connection.is_some(), "the scripted model");
    let Some(connection) = connection else {
        return;
    };
    app.world_mut().entity_mut(agent).insert(connection);

    // Turns 1 and 2: the same failure twice, no edit between.
    say(&mut app, &wakes, agent, "Run the tests.");
    assert!(ledger(&app, agent).is_empty(), "one failure is no dead end");
    say(&mut app, &wakes, agent, "Run them again.");
    assert_eq!(
        ledger(&app, agent),
        [(Source::Auto, "make test".to_owned(), 2)]
    );

    // Turn 3: the request carries the section; the repeat is warned about.
    say(&mut app, &wakes, agent, "Once more.");
    let requests = model.requests();
    let (system, messages) = parts(requests.get(4));
    assert!(system.contains("<known_dead_ends>"), "{system}");
    assert!(
        system.contains("make test") && system.contains("E42 boom"),
        "{system}"
    );
    assert!(!messages.contains("known_dead_ends"), "{messages}");
    let warned = output(&app, "3");
    assert!(
        warned.starts_with("Warning: this repeats known dead end 1"),
        "{warned}"
    );
    assert_eq!(
        warned
            .lines()
            .filter(|line| line.starts_with("Warning"))
            .count(),
        1
    );
    assert_eq!(
        ledger(&app, agent).len(),
        1,
        "the repeat adds no second entry"
    );

    // Turn 4: an edit, then the same command: no warning.
    say(&mut app, &wakes, agent, "Fix it and run them.");
    let after_edit = output(&app, "5");
    assert!(
        after_edit.contains("E42 boom") && !after_edit.contains("Warning"),
        "{after_edit}"
    );

    // The model's dead end and the user's.
    say(&mut app, &wakes, agent, "Try pinning libfoo.");
    app.world_mut()
        .trigger(RunCommand::new(agent, "dead-end avoid global mutex"));
    app.update();
    let sources: Vec<Source> = ledger(&app, agent)
        .into_iter()
        .map(|(source, ..)| source)
        .collect();
    assert_eq!(sources, [Source::Auto, Source::Agent, Source::User]);

    // A compaction: the next request still has all three, in the section
    // alone.
    app.world_mut().trigger(RunCommand::new(agent, "compact"));
    run_until(&mut app, &wakes, |world| {
        world
            .get::<Conversation>(agent)
            .is_some_and(|conversation| rig_basics::compaction::summarized_prefix(conversation) > 0)
    });
    say(&mut app, &wakes, agent, "What next?");
    let requests = model.requests();
    assert_eq!(requests.len(), 13, "every scripted request was made");
    let (system, messages) = parts(requests.last());
    for text in [
        "make test",
        "E42 boom",
        "pin libfoo 0.3",
        "API removed",
        "avoid global mutex",
    ] {
        assert!(system.contains(text), "{text} in {system}");
    }
    assert!(!messages.contains("known_dead_ends"), "{messages}");
    assert!(!messages.contains("avoid global mutex"), "{messages}");

    drop(app);

    // Resumed, the original has all three back.
    let resumed_model = MockCompletionModel::from_stream_turns([reply("Resumed.")]);
    let resumed = start(&home, &id, &project, false);
    assert!(resumed.is_some(), "the session was restored");
    let Some((mut app, wakes, agent)) = resumed else {
        return;
    };
    if let Some(connection) = connect(&resumed_model) {
        app.world_mut().entity_mut(agent).insert(connection);
    }
    assert_eq!(ledger(&app, agent).len(), 3, "the entries came back");
    say(&mut app, &wakes, agent, "Go on.");
    let (system, _) = parts(resumed_model.requests().first());
    assert!(
        system.contains("pin libfoo 0.3") && system.contains("avoid global mutex"),
        "{system}"
    );

    // `/dead-ends drop 3` removes the user's entry.
    app.world_mut()
        .trigger(RunCommand::new(agent, "dead-ends drop 3"));
    app.update();
    let sources: Vec<Source> = ledger(&app, agent)
        .into_iter()
        .map(|(source, ..)| source)
        .collect();
    assert_eq!(sources, [Source::Auto, Source::Agent]);
    drop(app);
    fs::remove_dir_all(home.root()).ok();
    fs::remove_dir_all(&project).ok();
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(DeadEndsLedgerPlugin);
}
