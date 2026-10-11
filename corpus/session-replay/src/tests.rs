use std::borrow::Cow;
use std::sync::mpsc::Receiver;

use rig_cassette::journal::MemoryStore;
use rig_core::catalog::{Catalog, Connector, ModelSpec};
use rig_core::operation::Completion;
use rig_core::providers::registry::{ConnectError, ProviderId};
use rig_core::serve::adapters::ModelAdapter;
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness::harness_protocol::{Home, Invocation, SessionId};
use rig_harness::host::exit::ExitPlugin;
use rig_harness_test_support::{
    app_in, app_on, calls, connect, first_agent, reply, run_until, scratch_home,
};
use rig_telemetry::EffectLogPlugin;
use rig_tools::Write;

use super::*;

/// Every call goes to its model as to one that calls no tools, so Start
/// takes the tools and their rules out of the request: what is sent, and
/// recorded, is not what Gate sees.
fn without_tools(mut calls: Query<&mut ModelRequest, Added<Admitted>>) {
    for mut call in &mut calls {
        let adapter = Adapter::new("no-tools", |mut connection: Connection| {
            let mut spec = (*connection.spec).clone();
            spec.tools = false;
            connection.spec = Arc::new(spec);
            connection
        });
        call.adapters.push(adapter);
    }
}

/// The kernel and the `write` tool, every agent connected to `connection`
/// (when there is one) as [`without_tools`] adapts it.
fn with_agents(app: &mut App, connection: Option<Connection>) {
    app.set_error_handler(bevy_ecs::error::warn)
        .add_plugins((AgentPlugin, JournalPlugin))
        .add_systems(AgentPass, without_tools.in_set(Stage::Prepare))
        .add_tool_with(
            Write::default(),
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
    Some((replay.status(), replay.divergences(), replay.answered()))
}

/// Records one turn that writes `path`, then answers "done", in the session
/// `id` of `home`, the model connected by `to`; its conversation.
fn record(
    home: &Home,
    id: &SessionId,
    path: &str,
    to: fn(&MockCompletionModel) -> Option<Connection>,
) -> Vec<serde_json::Value> {
    let model = MockCompletionModel::from_stream_turns([
        calls(
            "call-1",
            "write",
            serde_json::json!({"path": path, "content": "x"}),
        ),
        reply("done"),
    ]);
    let (mut app, wakes) = app_in(home, id);
    with_agents(&mut app, to(&model));
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
    let recorded = record(&home, &recorded_id, &file.to_string_lossy(), connect);
    assert_eq!(recorded.len(), 4, "{recorded:?}");
    assert_eq!(std::fs::read_to_string(&file).ok().as_deref(), Some("x"));
    std::fs::remove_file(&file).ok();
    let recorded_dir = home.session(&recorded_id).path().to_owned();
    // The calls were recorded as sent, without the tools Gate saw.
    let log = ReplaySource::from_dir(&recorded_dir, ReplayFlags::default());
    let sent = log.iter().flat_map(|source| &source.effects.records);
    let tools: Vec<usize> = sent
        .filter_map(|record| match &record.kind {
            EffectKind::Completion { request, .. } => Some(request.tools.len()),
            _ => None,
        })
        .collect();
    assert_eq!(tools, [0, 0]);

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
    with_agents(&mut app, connect(&model));
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
    with_agents(&mut app, connect(&model));
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

/// The catalog model `replay-corp/big`, whose provider has no key.
fn keyless() -> Option<Models> {
    let corp = ProviderId::declare("replay-corp").ok()?;
    let mut catalog = Catalog::default();
    catalog.insert(ModelSpec::new(corp, "big").with_tools(true));
    let mut connector = Connector::new(catalog);
    connector.provide(corp, |_: &ModelSpec, _: &Catalog| {
        Err(ConnectError::MissingKey {
            vendor: "replay-corp".to_owned(),
            tried: vec![Cow::Borrowed("REPLAY_CORP_KEY")],
        })
    });
    Some(Models(connector))
}

/// `model` as `replay-corp/big`, as if its key were there.
fn keyed(model: &MockCompletionModel) -> Option<Connection> {
    let spec = keyless()?
        .0
        .catalog()
        .resolve("replay-corp/big")
        .ok()?
        .shared();
    let handler = ModelAdapter::<Completion>::new(spec.reference(), model.clone());
    Some(Connection {
        spec,
        handler: Handler(ErasedHandler::new(handler)),
    })
}

#[test]
fn a_model_without_its_key_replays_from_the_record_and_a_turn_without_a_model_fails_the_replay() {
    let home = scratch_home("session-replay-keyless");
    let file = home.root().join("b.txt");
    let id = |id: &str| id.parse().unwrap_or_else(|_| SessionId::generate());
    let recorded_id = id("2-1");
    let recorded = record(&home, &recorded_id, &file.to_string_lossy(), keyed);
    assert_eq!(recorded.len(), 4, "{recorded:?}");
    std::fs::remove_file(&file).ok();
    let recorded_dir = home.session(&recorded_id).path().to_owned();
    // `--replay=DIR` with the key gone; the agent chooses `choice`.
    let replay = |id: &SessionId, choice: Option<&'static str>| {
        let (mut app, wakes) = app_in(&home, id);
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
        with_agents(&mut app, None);
        if let Some(models) = keyless() {
            app.insert_resource(models);
        }
        if let Some(choice) = choice {
            app.add_observer(move |agent: On<Add<Agent>>, mut commands: Commands| {
                let choice = ModelChoice(choice.to_owned());
                commands.entity(agent.entity).insert(choice);
            });
        }
        let outcome = run_replay(&mut app, &wakes);
        run_until(&mut app, &wakes, |world| {
            world.contains_resource::<ExitReason>()
        });
        let exit = app.world().get_resource::<ExitReason>().cloned();
        (outcome, exit, conversation(&mut app))
    };

    // The chosen model does not connect: the record answers it all the same.
    let (outcome, exit, replayed) = replay(&id("2-2"), Some("replay-corp/big"));
    assert_eq!(outcome, Some((ReplayStatus::Finished, Vec::new(), (2, 1))));
    assert_eq!(exit, Some(ExitReason::Quit));
    assert_eq!(replayed, recorded);
    assert!(!file.exists(), "the write tool ran during the replay");

    // No model at all: the turn fails before its call reaches the replay,
    // which stops, says why, and exits with 1.
    let (outcome, exit, _) = replay(&id("2-3"), None);
    let (status, divergences, answered) = outcome.unwrap_or_default();
    assert_eq!((status, answered), (ReplayStatus::Stopped, (0, 0)));
    let report = divergences
        .first()
        .map(ToString::to_string)
        .unwrap_or_default();
    assert!(
        report.contains("2 of the 2 recorded model calls were not replayed"),
        "{report}"
    );
    assert!(report.contains("No model is connected"), "{report}");
    assert_eq!(exit, Some(ExitReason::Failed(1)));
}
