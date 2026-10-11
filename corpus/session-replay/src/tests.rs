use std::path::PathBuf;
use std::sync::mpsc::Receiver;

use rig_cassette::effect_log::EffectLogRecorder;
use rig_cassette::journal::MemoryStore;
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness::rig_ecs::effects::Effects;
use rig_harness_test_support::{app_on, calls, connect, reply, run_until};
use rig_tools::Write;

use super::*;

/// The kernel and the `write` tool on `store`, every agent answered by
/// `model`.
fn app(store: &MemoryStore, model: &MockCompletionModel) -> (App, Receiver<()>) {
    let (mut app, wakes) = app_on(store);
    let connection = connect(model);
    app.set_error_handler(bevy_ecs::error::warn)
        .add_plugins((AgentPlugin, JournalPlugin))
        .add_tool_with(
            Write,
            ToolOptions {
                rules: Write::RULES,
                footprint: Footprint::Exclusive,
            },
        )
        .add_observer(move |agent: On<Add<Agent>>, mut commands: Commands| {
            if let Some(connection) = connection.clone() {
                commands.entity(agent.entity).insert(connection);
            }
        });
    (app, wakes)
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

/// Replays the recorded session with `section` added to the system prompt,
/// on a model that has no reply; that model, the app's conversation and
/// its replay.
fn replay(
    recorded: &ReplaySource,
    section: Option<&str>,
) -> (
    MockCompletionModel,
    Vec<serde_json::Value>,
    Option<(ReplayStatus, Vec<Divergence>, (usize, usize))>,
) {
    let model = MockCompletionModel::from_stream_turns(Vec::<Vec<MockStreamEvent>>::new());
    let (mut app, wakes) = app(&MemoryStore::default(), &model);
    if let Some(section) = section {
        app.world_mut()
            .spawn(PromptSection::new(50, "replay-test", section));
    }
    app.insert_resource(recorded.clone())
        .add_plugins(SessionReplayPlugin);
    app.finish();
    run_until(&mut app, &wakes, |world| {
        world.get_resource::<Replay>().is_some_and(|replay| {
            matches!(
                replay.status(),
                ReplayStatus::Finished | ReplayStatus::Stopped
            )
        })
    });
    let outcome = app
        .world()
        .get_resource::<Replay>()
        .map(|replay| (replay.status(), replay.divergences(), replay.answered()));
    (model.clone(), conversation(&mut app), outcome)
}

#[test]
fn a_recorded_session_replays_without_model_or_tools_and_a_changed_prompt_diverges() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target-test-replay");
    std::fs::create_dir_all(&dir).ok();
    let file = dir.join("a.txt");
    let path = file.to_string_lossy().into_owned();

    // Record: one turn that writes a.txt, then answers "done".
    let journal = MemoryStore::default();
    let model = MockCompletionModel::from_stream_turns([
        calls(
            "call-1",
            "write",
            serde_json::json!({"path": path, "content": "x"}),
        ),
        reply("done"),
    ]);
    let (mut app, wakes) = app(&journal, &model);
    let recorder = EffectLogRecorder::new();
    app.insert_resource(Effects::recorded_by(Arc::new(recorder.clone()), None));
    app.finish();
    app.update();
    let ended = |world: &World| {
        let mut agents = world.try_query::<&Conversation>();
        agents.as_mut().is_some_and(|agents| {
            agents
                .iter(world)
                .any(|conversation| conversation.messages().len() == 4)
        })
    };
    let agent = {
        let world = app.world_mut();
        world
            .query_filtered::<Entity, With<Agent>>()
            .iter(world)
            .next()
    };
    if let Some(agent) = agent {
        app.world_mut().trigger(Deliver::new(
            agent,
            "write a.txt \"x\"",
            DeliveryMode::Queue,
        ));
    }
    run_until(&mut app, &wakes, ended);
    // The journal is written at the end of the frame.
    app.update();
    let recorded_conversation = conversation(&mut app);
    assert_eq!(recorded_conversation.len(), 4, "{recorded_conversation:?}");
    assert_eq!(std::fs::read_to_string(&file).ok().as_deref(), Some("x"));
    std::fs::remove_file(&file).ok();
    let recorded = ReplaySource {
        effects: recorder.take(),
        journal: Arc::new(journal),
        flags: ReplayFlags::default(),
    };

    // Replay: the same conversation, with no model call and no write.
    let (model, replayed, outcome) = replay(&recorded, None);
    // Two model calls and the write, all answered from the record.
    assert_eq!(outcome, Some((ReplayStatus::Finished, Vec::new(), (2, 1))));
    assert_eq!(replayed, recorded_conversation);
    assert!(!file.exists(), "the write tool ran during the replay");
    assert!(model.requests().is_empty(), "the model was called");

    // A prompt section added: the first request diverges on the system prompt.
    let (model, _, outcome) = replay(&recorded, Some("Always answer in French."));
    let (status, divergences, _) = outcome.unwrap_or_default();
    assert_eq!(status, ReplayStatus::Stopped);
    let report = divergences
        .first()
        .map(ToString::to_string)
        .unwrap_or_default();
    assert!(report.contains("system prompt differs"), "{report}");
    assert!(report.contains("+ Always answer in French."), "{report}");
    assert!(model.requests().is_empty(), "the model was called");
    assert!(!file.exists());
    std::fs::remove_dir_all(&dir).ok();
}
