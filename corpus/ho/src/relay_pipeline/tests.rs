use std::sync::mpsc::Receiver;

use rig_core::message::Message;
use rig_core::test_utils::MockCompletionModel;
use rig_harness::harness_protocol::Home;

use super::*;
use rig_harness_test_support::{
    MemoryStore, app_on, commit_message, reply, run_until, scratch_home, scripted_models,
};

const PIPELINES: &str = r#"
[[feature.stage]]
name = "plan"
prompt = "Plan: {{task}}"
model = "pipe-stage/m"

[[feature.stage]]
name = "implement"
prompt = "Implement {{prev}}. {{feedback}}"
model = "pipe-stage/m"

[[feature.stage]]
name = "test"
prompt = "Test {{prev}}"
model = "pipe-stage/m"
tools = ["read"]
loop_to = "implement"
pass_when = "^PASS"
max_loops = 1
"#;

/// A tool the stages are offered, never called.
fn unused(_: On<ToolCalled<serde_json::Value>>) {}

/// On a call [`hold_implement`] holds.
#[derive(Component)]
struct Held;

/// Whether [`hold_implement`] holds a call.
#[derive(Resource, Default)]
struct Holding(bool);

/// Holds the implement stage's calls, so the stage runs at a restart.
fn hold_implement(
    calls: Query<(Entity, &ForAgent), (With<TurnCall>, ReadyToStart, Without<Held>)>,
    helpers: Query<&AgentTitle, With<rig_basics::subagents::Helper>>,
    mut holding: ResMut<Holding>,
    mut commands: Commands,
) {
    for (call, &ForAgent(agent)) in &calls {
        if helpers
            .get(agent)
            .is_ok_and(|title| title.0.ends_with("implement"))
        {
            commands.spawn(HoldOf(call));
            commands.entity(call).insert(Held);
            holding.0 = true;
        }
    }
}

/// An app on `store` with the pipelines of `home`, its user's agent on
/// `pipe-main/m` answered by `main` and the stages on `pipe-stage/m`
/// answered by `stages`, after its first frame restored the session; its
/// user's agent is new unless the session has one.
fn session(
    store: &MemoryStore,
    home: &Home,
    main: &MockCompletionModel,
    stages: &MockCompletionModel,
) -> (App, Receiver<()>, Entity) {
    let (mut app, wakes) = app_on(store);
    app.insert_resource(scripted_models([
        ("pipe-main/m", main),
        ("pipe-stage/m", stages),
    ]))
    .insert_resource(RigHome(home.clone()))
    .insert_resource(ProjectRoot(home.root().to_owned()))
    .add_plugins((AgentPlugin, SessionLogPlugin, RelayPipelinePlugin))
    .add_open_tool("read", "Read a file.", ToolOptions::default(), unused)
    .add_open_tool("write", "Write a file.", ToolOptions::default(), unused);
    app.finish();
    app.update();
    let world = app.world_mut();
    let restored = world
        .query_filtered::<Entity, (With<Agent>, Without<SpawnedBy>)>()
        .iter(world)
        .next();
    let host = restored.unwrap_or_else(|| world.spawn(Agent).id());
    // The model is chosen afresh: choosing one is `rig_basics::models`' job.
    world
        .entity_mut(host)
        .insert(ModelChoice("pipe-main/m".to_owned()));
    app.update();
    (app, wakes, host)
}

/// The scratch home `name` with the test's pipelines in its project.
fn home(name: &str) -> Home {
    let home = scratch_home(name);
    let rig = home.root().join(".rig");
    std::fs::create_dir_all(&rig).ok();
    std::fs::write(rig.join(FILE), PIPELINES).ok();
    home
}

/// Everything in `host`'s conversation.
fn said(world: &World, host: Entity) -> String {
    let conversation = world.get::<Conversation>(host);
    format!(
        "{:?}",
        conversation.map(|conversation| conversation.messages())
    )
}

/// Whether `host`'s pipeline ended and its report was answered.
fn reported(world: &World, host: Entity) -> bool {
    !world.entity(host).contains::<PipelineRun>() && said(world, host).contains("Noted.")
}

#[test]
fn a_failed_test_sends_the_pipeline_back_to_implement_and_the_agent_gets_the_report() {
    let main = MockCompletionModel::from_stream_turns([reply("Noted.")]);
    let stages =
        MockCompletionModel::from_stream_turns(["P", "I1", "FAIL: x", "I2", "PASS"].map(reply));
    let home = home("relay-pipeline");
    let (mut app, wakes, host) = session(&MemoryStore::default(), &home, &main, &stages);

    let line = "pipe feature add retry to the client".to_owned();
    app.world_mut().trigger(RunCommand::new(host, line));
    // Interrupting the user's agent leaves its pipeline running.
    app.world_mut().trigger(StopTurn::new(host));
    run_until(&mut app, &wakes, |world| reported(world, host));

    let asked: Vec<String> = stages
        .requests()
        .iter()
        .map(|request| format!("{request:?}"))
        .collect();
    let starts = [
        "Plan: add retry",
        "Implement P.",
        "Test I1",
        "Implement P. FAIL: x",
        "Test I2",
    ];
    assert_eq!(asked.len(), starts.len(), "{asked:#?}");
    for (request, start) in asked.iter().zip(starts) {
        assert!(request.contains(start), "{start}: {request}");
    }
    for request in stages.requests().iter().skip(2).step_by(2) {
        let offered: Vec<String> = request
            .tools
            .iter()
            .map(|tool| tool.name.to_string())
            .collect();
        assert_eq!(offered, ["read"]);
    }
    let report = said(app.world(), host);
    for line in [
        "passed on: add retry to the client",
        "1. plan: answered: P",
        "2. implement: answered: I1",
        "3. test: missed `^PASS`, back to implement (loop 1 of 1): FAIL: x",
        "4. implement: answered: I2",
        "5. test: passed: PASS",
    ] {
        assert!(report.contains(line), "{line}: {report}");
    }
}

#[test]
fn a_pipeline_running_at_a_restart_starts_its_current_stage_over() {
    let store = MemoryStore::default();
    let home = home("relay-pipeline-restart");
    let main = MockCompletionModel::from_stream_turns([reply("unused")]);
    let before = MockCompletionModel::from_stream_turns([reply("P")]);
    let (mut app, wakes, host) = session(&store, &home, &main, &before);
    app.init_resource::<Holding>()
        .add_systems(AgentPass, hold_implement.in_set(Stage::Gate));
    // A session is saved once its agent has a message.
    for earlier in [Message::user("Hello."), Message::assistant("Hi.")] {
        assert_eq!(commit_message(app.world_mut(), host, earlier), Ok(()));
    }
    let line = "pipe feature add retry to the client".to_owned();
    app.world_mut().trigger(RunCommand::new(host, line));
    let implementing = |world: &World| {
        let run = world.get::<PipelineRun>(host);
        run.is_some_and(|run| run.at == 1) && world.resource::<Holding>().0
    };
    run_until(&mut app, &wakes, implementing);
    assert!(implementing(app.world()));
    drop(app);

    let main = MockCompletionModel::from_stream_turns([reply("Noted.")]);
    let stages = MockCompletionModel::from_stream_turns(["I1", "FAIL: x", "I2", "PASS"].map(reply));
    let (mut app, wakes, host) = session(&store, &home, &main, &stages);
    run_until(&mut app, &wakes, |world| reported(world, host));
    let first = format!("{:?}", stages.requests().first());
    assert!(first.contains("Implement P."), "{first}");
    assert_eq!(stages.request_count(), 4);
    let report = said(app.world(), host);
    for line in [
        "1. plan: answered: P",
        "2. implement: answered: I1",
        "5. test: passed: PASS",
    ] {
        assert!(report.contains(line), "{line}: {report}");
    }
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(RelayPipelinePlugin);
}
