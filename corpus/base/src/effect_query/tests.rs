use std::fs;
use std::sync::mpsc::Receiver;

use rig_basics::coding_tools::ReadTool;
use rig_basics::telemetry::effect_log::EffectLogPlugin;
use rig_core::test_utils::MockCompletionModel;
use rig_harness::harness_protocol::{Home, SessionId};
use rig_harness_test_support::{
    Ended, ToolOutputs, app_in, calls, connect, first_agent, log_ended, log_tool_outputs, reply,
    run_until, scratch_home, start_call,
};

use super::*;

/// The kernel, the effect log, `read` and this plugin on session `id` of
/// `home`, after its first frame restored it, and its first agent.
fn start(home: &Home, id: &SessionId) -> (App, Receiver<()>, Entity) {
    let (mut app, wakes) = app_in(home, id);
    app.add_plugins((
        AgentPlugin,
        SessionLogPlugin,
        EffectLogPlugin,
        ReadTool,
        EffectQueryPlugin,
    ))
    .init_resource::<ToolOutputs>()
    .init_resource::<Ended>()
    .add_observer(log_tool_outputs)
    .add_observer(log_ended);
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    (app, wakes, agent)
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
        world.resource::<ToolOutputs>().0.contains_key(id)
    });
    app.world()
        .resource::<ToolOutputs>()
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
    let home = scratch_home("effect-query-home");
    let id = SessionId::generate();
    let present = home.root().join("a.txt");
    assert!(fs::create_dir_all(home.root()).is_ok());
    assert!(fs::write(&present, "hello\n").is_ok());
    let missing = home.root().join("missing.txt");

    let (mut app, wakes, agent) = start(&home, &id);
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

    let outputs = &app.world().resource::<ToolOutputs>().0;
    let table = outputs.get("c3").cloned().unwrap_or_default();
    let found = rows(&table);
    assert_eq!(found.len(), 1, "one failed tool call:\n{table}");
    let row = found.first().cloned().unwrap_or_default();
    assert_eq!(row.get(4), Some(&"read"), "{table}");
    assert_eq!(row.get(5), Some(&"error"), "{table}");
    assert_eq!(row.get(2), Some(&"0"), "the first turn's: {table}");
    // When it started, and how long it took.
    assert!(row.get(6).is_some_and(|start| start.len() == 8), "{table}");
    assert!(
        row.get(7).is_some_and(|took| took.ends_with('s')),
        "{table}"
    );
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

    let (mut app, wakes, agent) = start(&home, &id);
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
        serde_json::json!({ "kind": "model", "since_turn": 1 }),
    );
    assert_eq!(
        rows(&models).len(),
        2,
        "turn 2's two model calls:\n{models}"
    );
    // `/effects slow` lists the slowest first.
    let slow = Filter::parse("slow kind=tool");
    assert!(slow.as_ref().is_ok_and(|filter| filter.slowest));
    let records = app
        .world()
        .resource::<EffectLog>()
        .read()
        .unwrap_or_default();
    let slowest = query::table(&records, &HashMap::new(), &slow.unwrap_or_default());
    let took: Vec<f64> = rows(&slowest)
        .iter()
        .filter_map(|row| row.get(7)?.strip_suffix('s')?.parse().ok())
        .collect();
    assert_eq!(took.len(), 6, "every tool call so far: {slowest}");
    assert!(took.is_sorted_by(|a, b| a >= b), "{slowest}");
}
