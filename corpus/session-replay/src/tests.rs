use std::sync::mpsc::Receiver;

use rig_cassette::journal::MemoryStore;
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness::harness_protocol::{Home, Invocation, SessionId};
use rig_harness::host::exit::ExitPlugin;
use rig_harness_test_support::{
    app_in, app_on, calls, connect, first_agent, reply, run_until, scratch_home,
};
use rig_telemetry::EffectLogPlugin;
use rig_tools::Write;

use super::*;

/// The kernel and the `write` tool, every agent answered by `model`.
fn with_agents(app: &mut App, model: &MockCompletionModel) {
    let connection = connect(model);
    app.set_error_handler(bevy_ecs::error::warn)
        .add_plugins((AgentPlugin, JournalPlugin))
        .add_tool_with(
            Write,
            ToolOptions {
                rules: Write::RULES,
                footprint: Some(Footprint::Exclusive),
            },
        )
        .add_observer(move |agent: On<Add<Agent>>, mut commands: Commands| {
            if let Some(connection) = connection.clone() {
                commands.entity(agent.entity).insert(connection);
            }
        });
}

/// The primary agent's conversation, as JSON.
fn conversation(app: &mut App) -> Vec<serde_json::Value> {
    let world = app.world_mut();
    let mut agents = world.query_filtered::<&Conversation, Without<SpawnedBy>>();
    agents
        .iter(world)
        .flat_map(|conversation| conversation.messages().iter())
        .filter_map(|message| serde_json::to_value(message).ok())
        .collect()
}

/// Runs `app`'s replay to its end; its status, divergences and answers.
fn run_replay(
    app: &mut App,
    wakes: &Receiver<()>,
) -> Option<(ReplayStatus, Vec<Divergence>, (usize, usize))> {
    app.finish();
    run_until(app, wakes, |world| {
        world.get_resource::<Replay>().is_some_and(|replay| {
            matches!(
                replay.status(),
                ReplayStatus::Finished | ReplayStatus::Stopped
            )
        })
    });
    let replay = app.world().get_resource::<Replay>()?;
    Some((
        replay.status(),
        replay.divergences().to_vec(),
        replay.answered(),
    ))
}

/// Records one turn that writes `path`, then answers "done", in the session
/// `id` of `home`; its conversation.
fn record(home: &Home, id: &SessionId, path: &str) -> Vec<serde_json::Value> {
    let model = MockCompletionModel::from_stream_turns([
        calls(
            "call-1",
            "write",
            serde_json::json!({"path": path, "content": "x"}),
        ),
        reply("done"),
    ]);
    let (mut app, wakes) = app_in(home, id);
    with_agents(&mut app, &model);
    app.add_plugins(EffectLogPlugin);
    app.finish();
    app.update();
    if let Some(agent) = first_agent(app.world_mut()) {
        let ask = Deliver::new(agent, "write a.txt \"x\"", DeliveryMode::Queue);
        app.world_mut().trigger(ask);
    }
    run_until(&mut app, &wakes, |world| {
        let mut agents = world.try_query::<&Conversation>();
        agents.as_mut().is_some_and(|agents| {
            let mut conversations = agents.iter(world);
            conversations.any(|conversation| conversation.messages().len() == 4)
        })
    });
    // The effect log is written at the end of the frame.
    app.update();
    conversation(&mut app)
}

#[test]
fn a_recorded_session_replays_without_model_or_tools_and_a_changed_prompt_diverges() {
    let home = scratch_home("session-replay");
    let file = home.root().join("a.txt");
    // Two ids of one second: `SessionId::generate` would give the same.
    let id = |id: &str| id.parse().unwrap_or_else(|_| SessionId::generate());
    let (recorded_id, replay_id) = (id("1-1"), id("1-2"));
    let recorded = record(&home, &recorded_id, &file.to_string_lossy());
    assert_eq!(recorded.len(), 4, "{recorded:?}");
    assert_eq!(std::fs::read_to_string(&file).ok().as_deref(), Some("x"));
    std::fs::remove_file(&file).ok();
    let recorded_dir = home.session(&recorded_id).path().to_owned();

    // `--replay=DIR`, on a model that has no reply: the same conversation,
    // with no model call and no write, in a session that is gone after.
    let model = MockCompletionModel::from_stream_turns(Vec::<Vec<MockStreamEvent>>::new());
    let (mut app, wakes) = app_in(&home, &replay_id);
    app.insert_resource(Invoked {
        args: Invocation {
            flags: vec![(
                REPLAY.to_owned(),
                Some(recorded_dir.to_string_lossy().into()),
            )],
            ..Invocation::default()
        },
        terminal: false,
    })
    .add_plugins((ExitPlugin, SessionReplayPlugin));
    with_agents(&mut app, &model);
    let outcome = run_replay(&mut app, &wakes);
    // Two model calls and the write, all answered from the record.
    assert_eq!(outcome, Some((ReplayStatus::Finished, Vec::new(), (2, 1))));
    assert_eq!(conversation(&mut app), recorded);
    assert!(!file.exists(), "the write tool ran during the replay");
    assert!(model.requests().is_empty(), "the model was called");
    run_until(&mut app, &wakes, |world| {
        world.contains_resource::<ExitReason>()
    });
    app.update();
    assert!(
        !home.session(&replay_id).path().exists(),
        "the replay kept its session"
    );
    assert!(recorded_dir.exists());

    // A prompt section added: the first request diverges on the system prompt.
    let source = ReplaySource::from_dir(&recorded_dir, ReplayFlags::default());
    let (mut app, wakes) = app_on(&MemoryStore::default());
    with_agents(&mut app, &model);
    app.world_mut().spawn(PromptSection::new(
        50,
        "replay-test",
        "Always answer in French.",
    ));
    if let Ok(source) = source {
        app.insert_resource(source);
    }
    app.add_plugins(SessionReplayPlugin);
    let (status, divergences, _) = run_replay(&mut app, &wakes).unwrap_or_default();
    assert_eq!(status, ReplayStatus::Stopped);
    let report = divergences
        .first()
        .map(ToString::to_string)
        .unwrap_or_default();
    assert!(report.contains("system prompt differs"), "{report}");
    assert!(report.contains("+ Always answer in French."), "{report}");
    assert!(model.requests().is_empty(), "the model was called");
    assert!(!file.exists());
}
