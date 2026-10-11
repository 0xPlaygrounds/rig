use std::path::PathBuf;
use std::sync::mpsc::Receiver;

use rig_cassette::journal::MemoryStore;
use rig_coding_tools::{EditTool, ReadTool, ShellTool, WriteTool};
use rig_harness::rig_core::catalog::{Catalog, Connector};
use rig_harness::rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness_test_support::{app_on, calls, connect, reply, run_until, start_call};
use rig_subagents::{SubagentsPlugin, TASK};
use serde_json::json;

use super::*;

/// The titles of the pickers asked for, with the agent each answers for.
#[derive(Resource, Default)]
struct Picks(Vec<(Entity, String)>);

fn record_picks(mut picks: MessageReader<PickRequest>, mut seen: ResMut<Picks>) {
    for pick in picks.read() {
        seen.0.push((pick.agent, pick.title.clone()));
    }
}

/// An interactive app with the coding tools, subagents and permission
/// modes, after its first frame, and an agent in `mode` on `model`.
fn session(mode: PermissionMode, model: &MockCompletionModel) -> (App, Receiver<()>, Entity) {
    let (mut app, wakes) = app_on(&MemoryStore::default());
    let interactive = Invoked {
        terminal: true,
        ..Invoked::default()
    };
    app.insert_resource(Models(Connector::new(Catalog::default())))
        .insert_resource(interactive)
        .insert_resource(PermissionConfig::default())
        .init_resource::<Picks>()
        .add_plugins((AgentPlugin, JournalPlugin, SubagentsPlugin))
        .add_plugins((ReadTool, WriteTool, EditTool, ShellTool))
        .add_plugins(PermissionModesPlugin)
        .add_systems(Update, record_picks);
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
    let message = Deliver::new(agent, text, DeliveryMode::Queue);
    app.world_mut().trigger(message);
}

/// Whether `agent` has no running turn.
fn idle(world: &World, agent: Entity) -> bool {
    world.get::<ActiveTurn>(agent).is_none()
}

/// The id of the call of `agent` waiting for approval.
fn waiting(world: &World, agent: Entity) -> Option<u64> {
    let mut waiting = world.try_query::<&Awaiting>()?;
    let mut waiting = waiting.iter(world);
    waiting
        .find(|awaiting| awaiting.agent == agent)
        .map(|awaiting| awaiting.id)
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
    let model = MockCompletionModel::from_stream_turns([
        vec![
            MockStreamEvent::tool_call("e1", "edit", edit),
            MockStreamEvent::tool_call("s1", "shell", shell.clone()),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        reply("Edited; the command was denied."),
    ]);
    let (mut app, wakes, agent) = session(PermissionMode::Edits, &model);
    tell(&mut app, agent, "Edit, then run.");
    run_until(&mut app, &wakes, |world| waiting(world, agent).is_some());

    // The edit ran at once; the shell call waits for the user's answer.
    let edited = std::fs::read_to_string(&file).unwrap_or_default();
    assert_eq!(edited, "goodbye\n");
    let id = waiting(app.world(), agent);
    assert!(id.is_some());
    assert!(!marker.exists());
    assert_eq!(model.request_count(), 1);
    app.update();
    let asked = app.world().resource::<Picks>().0.last().cloned();
    assert!(asked.is_some_and(|(to, title)| to == agent && title.contains("shell")));

    // The picker's "Deny" runs this command for the agent.
    let line = format!("permit {} deny", id.unwrap_or_default());
    app.world_mut().trigger(RunCommand {
        entity: agent,
        line,
    });
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
    // shell call waits too, asked with its name, for it.
    let task = json!({ "description": "Run it", "prompt": "Run the command." });
    start_call(app.world_mut(), agent, (TASK, "t1"), task);
    let spawned = app.world().get::<Spawned>(agent);
    let child = spawned.and_then(|spawned| spawned.iter().last());
    let child = child.unwrap_or(Entity::PLACEHOLDER);
    let mode = app.world().get::<PermissionMode>(child).copied();
    assert_eq!(mode, Some(PermissionMode::Edits));
    let sub = MockCompletionModel::from_stream_turns([calls("s2", "shell", shell), reply("Ran.")]);
    if let Some(connection) = connect(&sub) {
        app.world_mut().entity_mut(child).insert(connection);
    }
    run_until(&mut app, &wakes, |world| waiting(world, child).is_some());
    assert!(waiting(app.world(), child).is_some());
    assert!(!marker.exists());
    app.update();
    let asked = app.world().resource::<Picks>().0.last().cloned();
    assert!(asked.is_some_and(|(to, title)| to == child && title.starts_with("Subagent")));
    std::fs::remove_dir_all(&dir).ok();
}
