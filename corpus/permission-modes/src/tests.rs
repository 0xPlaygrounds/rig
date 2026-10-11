use std::path::PathBuf;
use std::sync::mpsc::Receiver;

use rig_cassette::journal::MemoryStore;
use rig_coding_tools::{EditTool, ReadTool, ShellTool, WriteTool};
use rig_harness::rig_core::catalog::{Catalog, Connector};
use rig_harness::rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness_test_support::{app_on, calls, connect, reply, run_until, start_call};
use rig_subagents::{SubagentsPlugin, TASK};
use rig_tui::{KeyPressed, Keys};
use serde_json::json;

use super::*;

/// An app with the coding tools, subagents and permission modes, someone
/// at it who answers questions, after its first frame, and an agent in
/// `mode` on `model`.
fn session(mode: PermissionMode, model: &MockCompletionModel) -> (App, Receiver<()>, Entity) {
    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.insert_resource(Models(Connector::new(Catalog::default())))
        .add_plugins((AgentPlugin, JournalPlugin, SubagentsPlugin))
        .add_plugins((ReadTool, WriteTool, EditTool, ShellTool))
        .add_plugins(PermissionModesPlugin);
    app.world_mut().spawn(Answerer::default());
    let mut agent = app.world_mut().spawn((Agent, mode));
    if let Some(connection) = connect(model) {
        let choice = ModelChoice(connection.spec.reference());
        agent.insert((connection, choice));
    }
    let agent = agent.id();
    app.finish();
    app.update();
    (app, wakes, agent)
}

/// A fresh directory for a test's files, in this crate's `target`.
fn scratch(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join(format!("{name}-{}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
    std::fs::create_dir_all(&dir).ok();
    dir
}

fn tell(app: &mut App, agent: Entity, text: &str) {
    app.world_mut()
        .trigger(Deliver::new(agent, text, DeliveryMode::Queue));
}

fn idle(world: &World, agent: Entity) -> bool {
    world.get::<ActiveTurn>(agent).is_none()
}

/// The open approval question about `agent`'s call, and its title.
fn question(world: &mut World, agent: Entity) -> Option<(Entity, String)> {
    let mut asks =
        world.query_filtered::<(Entity, &Ask, &ForAgent), (With<Approval>, Without<Outcome>)>();
    asks.iter(world)
        .find(|(.., to)| to.0 == agent)
        .map(|(ask, question, _)| (ask, question.title.clone()))
}

#[test]
fn read_only_offers_and_runs_only_tools_that_read() {
    let dir = scratch("read-only");
    let file = dir.join("new.txt");
    let write = json!({ "path": file.display().to_string(), "content": "hi" });
    let model = MockCompletionModel::from_stream_turns([calls("w1", "write", write), reply("Ok.")]);
    let (mut app, wakes, agent) = session(PermissionMode::ReadOnly, &model);
    tell(&mut app, agent, "Write a file.");
    run_until(&mut app, &wakes, |world| {
        model.request_count() == 2 && idle(world, agent)
    });

    let requests = model.requests();
    let offered: Vec<String> = requests.first().map_or_else(Vec::new, |request| {
        let names = request
            .tools
            .iter()
            .map(|tool| tool.name.as_str().to_owned());
        names.collect()
    });
    assert!(offered.iter().any(|name| name == "read"), "{offered:?}");
    for changes in ["write", "edit", "shell"] {
        assert!(!offered.iter().any(|name| name == changes), "{offered:?}");
    }
    let answered = format!("{:?}", requests.get(1));
    assert!(answered.contains("read-only mode"), "{answered}");
    assert!(!file.exists());

    // shift+tab cycles the shown agent's mode.
    let world = app.world_mut();
    let shift_tab = Keys::from(KeyPress::key(KeyCode::BackTab));
    let mut bindings = world.query::<(Entity, &KeyBinding)>();
    let binding = bindings
        .iter(world)
        .find(|(_, binding)| binding.keys == shift_tab);
    let binding = binding.map_or(Entity::PLACEHOLDER, |(binding, _)| binding);
    let pressed = KeyPressed {
        entity: binding,
        agent: Some(agent),
        released: false,
    };
    world.trigger(pressed);
    app.update();
    let mode = app.world().get::<PermissionMode>(agent).copied();
    assert_eq!(mode, Some(PermissionMode::Ask));
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn edits_runs_edits_and_asks_before_shell_also_for_a_subagent() {
    let dir = scratch("edits");
    let (file, marker) = (dir.join("notes.txt"), dir.join("ran"));
    std::fs::write(&file, "hello\n").ok();
    let edit = json!({
        "path": file.display().to_string(),
        "edits": [{ "old_text": "hello", "new_text": "goodbye" }],
    });
    let shell = json!({ "command": format!("touch {}", marker.display()) });
    // The subagent shares its parent's model: its first reply is the third.
    let model = MockCompletionModel::from_stream_turns([
        vec![
            MockStreamEvent::tool_call("e1", "edit", edit),
            MockStreamEvent::tool_call("s1", "shell", shell.clone()),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        reply("Edited; the command was denied."),
        calls("s2", "shell", shell),
        reply("Ran."),
    ]);
    let (mut app, wakes, agent) = session(PermissionMode::Edits, &model);
    tell(&mut app, agent, "Edit, then run.");
    run_until(&mut app, &wakes, |world| {
        let asks = world.try_query_filtered::<&ForAgent, With<Approval>>();
        asks.is_some_and(|mut asks| asks.iter(world).any(|to| to.0 == agent))
    });

    // The edit ran at once; the shell call waits for the user's answer.
    let edited = std::fs::read_to_string(&file).unwrap_or_default();
    assert_eq!(edited, "goodbye\n");
    let asked = question(app.world_mut(), agent);
    assert!(
        asked
            .as_ref()
            .is_some_and(|(_, title)| title == "The agent wants to run `shell`")
    );
    assert!(!marker.exists());
    assert_eq!(model.request_count(), 1);

    // The user answers Deny.
    if let Some((ask, _)) = asked {
        let deny = Answer::Choice("deny".to_owned()).outcome();
        app.world_mut().entity_mut(ask).insert(deny);
    }
    run_until(&mut app, &wakes, |world| {
        model.request_count() == 2 && idle(world, agent)
    });
    let answered = format!("{:?}", model.requests().get(1));
    assert!(
        answered.contains("Denied: the user did not allow"),
        "{answered}"
    );
    assert!(!marker.exists());

    // A subagent of an agent in `edits` starts in `edits`, and its first
    // shell call waits too, asked with its name.
    let task = json!({ "description": "Run it", "prompt": "Run the command." });
    start_call(app.world_mut(), agent, (TASK, "t1"), task);
    run_until(&mut app, &wakes, |world| {
        let child = world
            .get::<Spawned>(agent)
            .and_then(|spawned| spawned.iter().last());
        let mut asks = world.try_query_filtered::<&ForAgent, With<Approval>>();
        let asked = |child| {
            asks.as_mut()
                .is_some_and(|asks| asks.iter(world).any(|to| to.0 == child))
        };
        child.is_some_and(asked)
    });
    let child = app
        .world()
        .get::<Spawned>(agent)
        .and_then(|spawned| spawned.iter().last());
    let child = child.unwrap_or(Entity::PLACEHOLDER);
    let mode = app.world().get::<PermissionMode>(child).copied();
    assert_eq!(mode, Some(PermissionMode::Edits));
    let asked = question(app.world_mut(), child).map(|(_, title)| title);
    assert_eq!(
        asked.as_deref(),
        Some("Subagent \"Run it\" wants to run `shell`")
    );
    assert!(!marker.exists());
    std::fs::remove_dir_all(&dir).ok();
}
