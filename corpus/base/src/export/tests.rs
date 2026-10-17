use rig_basics::coding_tools::ReadTool;
use rig_basics::subagents::SubagentsPlugin;
use rig_basics::telemetry::EffectLogPlugin;
use rig_core::catalog::{Catalog, Connector};
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness_test_support::{app_in, connect, run_until, scratch_home};
use serde_json::{Value, json};

use super::*;

/// The end of a reply of a million output tokens, which the catalog
/// prices.
fn spent() -> MockStreamEvent {
    let mut usage = Usage::default();
    usage.input_tokens = Some(0);
    usage.output_tokens = Some(1_000_000);
    MockStreamEvent::final_response(usage)
}

/// A reply of `text`, then a priced end.
fn says(text: &str) -> Vec<MockStreamEvent> {
    vec![MockStreamEvent::text(text), spent()]
}

/// `model`, pricing its replies as the catalog model [`connect`] names.
fn priced(model: MockCompletionModel) -> MockCompletionModel {
    match Catalog::builtin().resolve("deepseek/deepseek-flash") {
        Ok(spec) => model.priced_as(spec.shared()),
        Err(_) => model,
    }
}

#[test]
fn exports_the_saved_session_with_the_subagent_under_its_task_call() {
    // The session is kept in its directory under a RIG_HOME of the test's.
    let home = scratch_home("corpus-export-test");
    let id = SessionId::generate();
    let (mut app, wakes) = app_in(&home, &id);
    let project = home.root().join("work");
    std::fs::create_dir_all(&project).ok();
    let notes = project.join("notes.txt");
    std::fs::write(&notes, "The answer is in the subagent.").ok();

    // The parent reads a file and starts a subagent in one reply, waits,
    // and answers once the subagent reported; the subagent answers by a
    // model of its own.
    let read = json!({ "path": notes.display().to_string() });
    let task = json!({ "description": "Find the answer", "prompt": "Answer briefly." });
    let parent_model = priced(MockCompletionModel::from_stream_turns([
        vec![
            MockStreamEvent::tool_call("r1", "read", read),
            MockStreamEvent::tool_call("t1", "task", task),
            spent(),
        ],
        says("Waiting for the subagent."),
        says("Done."),
    ]));
    let child_model = priced(MockCompletionModel::from_stream_turns([says("sub-answer")]));
    app.insert_resource(ProjectRoot(project.clone()))
        .insert_resource(Models(Connector::new(Catalog::default())));
    let child_connection = connect(&child_model);
    app.add_plugins((
        AgentPlugin,
        SessionLogPlugin,
        EffectLogPlugin,
        ReadTool,
        SubagentsPlugin,
        ExportPlugin,
    ))
    .add_observer(move |spawned: On<Add<SpawnedBy>>, mut commands: Commands| {
        if let Some(connection) = child_connection.clone() {
            commands.entity(spawned.entity).insert(connection);
        }
    });
    let connection = connect(&parent_model);
    assert!(connection.is_some(), "the scripted model does not connect");
    let Some(connection) = connection else {
        return;
    };
    let choice = ModelChoice(connection.spec.reference());
    let parent = app.world_mut().spawn((Agent, connection, choice)).id();
    app.finish();
    app.update();

    let asked = "Read the notes and ask a subagent.";
    app.world_mut()
        .trigger(Deliver::new(parent, asked, DeliveryMode::Queue));
    run_until(&mut app, &wakes, |world| {
        let idle = world
            .try_query::<&TurnOf>()
            .is_some_and(|mut turns| turns.iter(world).next().is_none());
        parent_model.request_count() == 3 && idle
    });
    assert_eq!(parent_model.request_count(), 3);

    let export = |app: &mut App, line: String| {
        app.world_mut()
            .trigger(Deliver::new(parent, line, DeliveryMode::Queue));
        app.update();
    };
    // This session, from its logs.
    export(&mut app, "/export md out.md".to_owned());
    let text = std::fs::read_to_string(project.join("out.md")).unwrap_or_default();
    assert!(text.contains(asked), "{text}");
    assert!(text.contains("Tool call `read`"), "{text}");
    assert!(text.contains("Done."), "{text}");
    let after_task = text.split("Tool call `task`").nth(1).unwrap_or_default();
    let subagent = after_task.split("### Subagent").nth(1).unwrap_or_default();
    assert!(subagent.contains("sub-answer"), "{text}");

    // The same session by its id, as one that is not running is read.
    export(&mut app, format!("/export json out.json --session {id}"));
    let exported = std::fs::read_to_string(project.join("out.json")).unwrap_or_default();
    let exported: Value = serde_json::from_str(&exported).unwrap_or_default();
    let agents = exported
        .get("agents")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    assert_eq!(agents.len(), 2, "{exported}");
    let calls: Vec<usize> = agents
        .iter()
        .map(|agent| {
            agent
                .get("calls")
                .and_then(Value::as_array)
                .map_or(0, Vec::len)
        })
        .collect();
    assert_eq!(calls, [3, 1], "each model call with its cost: {exported}");
    let turns = agents
        .iter()
        .filter_map(|agent| agent.get("calls").and_then(Value::as_array))
        .flatten()
        .all(|call| call.get("turn").is_some_and(Value::is_u64));
    assert!(turns, "each call names its turn: {exported}");
    let cost = agents
        .first()
        .and_then(|parent| parent.pointer("/usage/cost"));
    let cost = cost.and_then(Value::as_f64).unwrap_or_default();
    assert!(cost > 0.0, "{exported}");

    std::fs::remove_dir_all(home.root()).ok();
}
