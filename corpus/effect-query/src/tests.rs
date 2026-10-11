use std::fs;
use std::path::{Path, PathBuf};
use std::sync::mpsc::Receiver;

use rig_cassette::journal::JsonlDirStore;
use rig_coding_tools::ReadTool;
use rig_core::test_utils::MockCompletionModel;
use rig_ecs::journal::SessionStore;
use rig_harness::harness_protocol::{SessionDir, SessionId, env};
use rig_harness_test_support::{Ended, calls, connect, log_ended, reply, run_until, start_call};
use rig_telemetry::EffectLogPlugin;

use super::*;

/// Each finished tool call's output text, by the model's call id.
#[derive(Resource, Default)]
struct Outputs(HashMap<String, String>);

fn keep_output(
    output: On<Insert<Outcome>>,
    calls: Query<(&ToolRequest, &Outcome)>,
    mut outputs: ResMut<Outputs>,
) {
    if let Ok((request, outcome)) = calls.get(output.entity) {
        let result = outcome.tool_result();
        outputs
            .0
            .insert(request.call.id.to_string(), result.output().render());
    }
}

/// The kernel, the effect log, `read` and this plugin on the session in
/// `dir`, after its first frame restored it.
fn start(dir: &SessionDir) -> (App, Receiver<()>) {
    let (sender, wakes) = std::sync::mpsc::channel();
    let mut app = App::new();
    app.add_plugins(TaskPoolPlugin::default())
        .insert_resource(Wake::new(move || {
            sender.send(()).ok();
        }))
        .insert_resource(SessionPaths(dir.clone()))
        .insert_resource(SessionStore::new(JsonlDirStore::new(dir.path())))
        .set_error_handler(bevy_ecs::error::warn)
        .add_plugins((
            AgentPlugin,
            JournalPlugin,
            EffectLogPlugin,
            ReadTool,
            EffectQueryPlugin,
        ))
        .init_resource::<Outputs>()
        .init_resource::<Ended>()
        .add_observer(keep_output)
        .add_observer(log_ended);
    app.finish();
    app.update();
    (app, wakes)
}

fn first_agent(app: &mut App) -> Option<Entity> {
    let world = app.world_mut();
    world
        .query_filtered::<Entity, With<Agent>>()
        .iter(world)
        .next()
}

/// A `RIG_HOME` of the test's own beside its binary, emptied.
fn home() -> PathBuf {
    let exe = std::env::current_exe().unwrap_or_default();
    let root = exe
        .parent()
        .unwrap_or(Path::new("."))
        .join("effect-query-home");
    fs::remove_dir_all(&root).ok();
    // SAFETY: this crate has one test, and nothing else reads the
    // environment while it is set.
    unsafe { std::env::set_var(env::HOME, &root) };
    root
}

/// Runs `tool` with `args` as `agent`'s call `id` outside a turn, and
/// returns its output.
fn call_tool(
    app: &mut App,
    wakes: &Receiver<()>,
    agent: Entity,
    (tool, id): (&str, &str),
    args: serde_json::Value,
) -> String {
    start_call(app.world_mut(), agent, (tool, id), args);
    run_until(app, wakes, |world| {
        world.resource::<Outputs>().0.contains_key(id)
    });
    app.world()
        .resource::<Outputs>()
        .0
        .get(id)
        .cloned()
        .unwrap_or_default()
}

/// The table's data rows.
fn rows(table: &str) -> Vec<Vec<&str>> {
    table
        .lines()
        .skip(2)
        .map(|line| line.split('\t').collect())
        .collect()
}

#[test]
fn finds_the_failed_read_and_still_does_after_a_restart() {
    let root = home();
    let id = "1000-1".parse::<SessionId>();
    assert!(id.is_ok(), "a session id");
    let Ok(id) = id else {
        return;
    };
    let dir = Home::from_env().session(&id);
    assert!(fs::create_dir_all(dir.path()).is_ok());
    let present = root.join("a.txt");
    assert!(fs::write(&present, "hello\n").is_ok());
    let missing = root.join("missing.txt");

    let (mut app, wakes) = start(&dir);
    let agent = first_agent(&mut app);
    assert!(agent.is_some(), "the first agent");
    let Some(agent) = agent else {
        return;
    };
    let model = MockCompletionModel::from_stream_turns([
        calls("c1", "read", serde_json::json!({ "path": present })),
        calls("c2", "read", serde_json::json!({ "path": missing })),
        reply("Read them."),
        calls(
            "c3",
            "effects",
            serde_json::json!({ "kind": "tool", "status": "error" }),
        ),
        reply("One failed."),
    ]);
    if let Some(connection) = connect(&model) {
        app.world_mut().entity_mut(agent).insert(connection);
    }
    for (turn, text) in ["read both", "which failed?"].into_iter().enumerate() {
        app.world_mut()
            .trigger(Deliver::new(agent, text, DeliveryMode::Queue));
        run_until(&mut app, &wakes, |world| {
            world.resource::<Ended>().0.len() == turn + 1
        });
    }

    let outputs = &app.world().resource::<Outputs>().0;
    let table = outputs.get("c3").cloned().unwrap_or_default();
    let found = rows(&table);
    assert_eq!(found.len(), 1, "one failed tool call:\n{table}");
    let row = found.first().cloned().unwrap_or_default();
    assert_eq!(row.get(4), Some(&"read"), "{table}");
    assert_eq!(row.get(5), Some(&"error"), "{table}");
    let failed = row.first().copied().unwrap_or_default().to_owned();
    let read_error = outputs.get("c2").cloned().unwrap_or_default();
    assert!(!read_error.is_empty());

    let detail = call_tool(
        &mut app,
        &wakes,
        agent,
        ("effects_detail", "d1"),
        serde_json::json!({ "id": failed.parse::<u64>().unwrap_or_default() }),
    );
    assert!(detail.contains("missing.txt"), "{detail}");
    assert!(detail.contains("result:\nerror: "), "{detail}");
    // The model saw the error's message, which the detail repeats.
    let message = read_error.lines().next().unwrap_or_default();
    assert!(detail.contains(message), "{message:?} in {detail}");
    // Let the effect log write the last calls.
    app.update();
    drop(app);

    let (mut app, wakes) = start(&dir);
    let agent = first_agent(&mut app);
    assert!(agent.is_some(), "the restored agent");
    let Some(agent) = agent else {
        return;
    };
    let table = call_tool(
        &mut app,
        &wakes,
        agent,
        ("effects", "e1"),
        serde_json::json!({ "kind": "tool" }),
    );
    let found: Vec<(String, String)> = rows(&table)
        .iter()
        .map(|row| {
            let cell = |at: usize| row.get(at).copied().unwrap_or_default().to_owned();
            (cell(4), cell(5))
        })
        .collect();
    let expected = [
        ("read", "ok"),
        ("read", "error"),
        ("effects", "ok"),
        ("effects_detail", "ok"),
    ]
    .map(|(name, status)| (name.to_owned(), status.to_owned()));
    assert_eq!(found, expected, "{table}");
    let models = call_tool(
        &mut app,
        &wakes,
        agent,
        ("effects", "e2"),
        serde_json::json!({ "kind": "model", "since_turn": 2 }),
    );
    assert_eq!(
        rows(&models).len(),
        2,
        "turn 2's two model calls:\n{models}"
    );
}
