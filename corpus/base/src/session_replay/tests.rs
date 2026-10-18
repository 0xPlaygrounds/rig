use std::sync::mpsc::Receiver;

use rig_harness_test_support::MemoryStore;
use std::borrow::Cow;

use corpus_ho::exit_notes::{ExitNotesPlugin, NotesCall};
use rig_basics::BasicCommandsPlugin;
use rig_basics::telemetry::effect_log::EffectLogPlugin;
use rig_core::catalog::{Catalog, Connector, ModelSpec, Transport};
use rig_core::operation::Completion;
use rig_core::providers::registry::{ConnectError, ProviderId};
use rig_core::serve::adapters::ModelAdapter;
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness::harness_protocol::{Home, Invocation, SessionId};
use rig_harness::host::exit::ExitPlugin;
use rig_harness_test_support::{
    app_in, app_on, calls, connect, first_agent, reply, run_until, scratch_home,
};
use rig_tools::Write;

use super::*;

/// The session recorded in `dir`, read as the app's stores read it.
fn recorded_source(dir: &std::path::Path) -> io::Result<ReplaySource> {
    let store = StoreOpener::default().open(&SessionDir::at(dir))?;
    ReplaySource::from_store(&*store, ReplayFlags::default())
}

/// Every call goes to its model as to one that calls no tools, so Start
/// takes the tools out of the request: what is sent, and recorded, is not
/// what Gate sees.
fn without_tools(mut calls: Query<&mut Adapters, Added<Admitted>>) {
    for mut adapters in &mut calls {
        let adapter = Adapter::new("no-tools", |mut connection: Connection| {
            let mut spec = (*connection.spec).clone();
            spec.tools = false;
            connection.spec = Arc::new(spec);
            connection
        });
        adapters.0.push(adapter);
    }
}

/// The kernel and the `write` tool, every agent connected to `connection`
/// (when there is one) as [`without_tools`] adapts it.
fn with_agents(app: &mut App, connection: Option<Connection>) {
    app.set_error_handler(bevy_ecs::error::warn)
        .add_plugins((AgentPlugin, SessionLogPlugin))
        .add_systems(AgentPass, without_tools.in_set(Stage::Prepare))
        .add_tool_with(
            Write::default(),
            ToolOptions {
                footprint: Some(Touches::Writes(Scope::Anything)),
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

/// The summary exit-notes' call answers.
const SUMMARY: &str = "## Goal\nWrite a.txt.";

/// Exit-notes, and `/quit`, writing their notes in the project `root`.
fn with_exit_notes(app: &mut App, root: &std::path::Path) {
    app.insert_resource(ProjectRoot(root.to_path_buf()))
        .add_plugins((ExitPlugin, BasicCommandsPlugin, ExitNotesPlugin));
}

/// The notes exit-notes wrote in the project `root`.
fn notes_in(root: &std::path::Path) -> Vec<String> {
    let notes = std::fs::read_dir(root.join(".rig").join("notes"));
    let notes = notes.into_iter().flatten().flatten();
    notes
        .filter_map(|note| std::fs::read_to_string(note.path()).ok())
        .collect()
}

/// Records one turn that writes `path`, then answers "done", in the session
/// `id` of `home`, the model connected by `to`; its conversation. Under
/// exit-notes for the project `quit_in`, the session then ends with
/// `/quit`, whose summary call answers [`SUMMARY`].
fn record(
    home: &Home,
    id: &SessionId,
    path: &str,
    to: fn(&MockCompletionModel) -> Option<Connection>,
    quit_in: Option<&std::path::Path>,
) -> Vec<serde_json::Value> {
    let model = MockCompletionModel::from_stream_turns([
        calls(
            "call-1",
            "write",
            serde_json::json!({"path": path, "content": "x"}),
        ),
        reply("done"),
        reply(SUMMARY),
    ]);
    let (mut app, wakes) = app_in(home, id);
    with_agents(&mut app, to(&model));
    app.add_plugins(EffectLogPlugin);
    if let Some(root) = quit_in {
        with_exit_notes(&mut app, root);
    }
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
    if let (Some(agent), Some(_)) = (first_agent(app.world_mut()), quit_in) {
        let quit = Deliver::new(agent, "/quit", DeliveryMode::Queue).with_author(Author::user());
        app.world_mut().trigger(quit);
        run_until(&mut app, &wakes, |world| {
            !world.resource::<Messages<AppExit>>().is_empty()
        });
    }
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
    let recorded = record(&home, &recorded_id, &file.to_string_lossy(), connect, None);
    assert_eq!(recorded.len(), 4, "{recorded:?}");
    assert_eq!(std::fs::read_to_string(&file).ok().as_deref(), Some("x"));
    std::fs::remove_file(&file).ok();
    let recorded_dir = home.session(&recorded_id).path().to_owned();
    // The calls were recorded as sent, without the tools Gate saw.
    let log = recorded_source(&recorded_dir);
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
    let source = recorded_source(&recorded_dir);
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

/// The model the [`keyless`] catalog holds.
const KEYLESS: &str = "replay-corp/big";

/// A catalog of [`KEYLESS`], whose provider has no key for it.
fn keyless() -> Option<Models> {
    let corp = ProviderId::declare("replay-corp").ok()?;
    let mut catalog = Catalog::default();
    catalog.insert(ModelSpec::new(corp, "big").with_tools(true));
    let mut connector = Connector::new(catalog);
    connector.provide(corp, |_: &ModelSpec, _: &Catalog, _: &Transport| {
        Err(ConnectError::MissingKey {
            vendor: "replay-corp".to_owned(),
            tried: vec![Cow::Borrowed("REPLAY_CORP_KEY")],
        })
    });
    Some(Models(connector))
}

/// `model` as [`KEYLESS`], as if its key were there.
fn keyed(model: &MockCompletionModel) -> Option<Connection> {
    let spec = keyless()?.0.catalog().resolve(KEYLESS).ok()?.shared();
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
    let recorded = record(&home, &recorded_id, &file.to_string_lossy(), keyed, None);
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
        let replayed = conversation(&mut app);
        app.update();
        // Whatever the exit, the replay's own session is gone.
        assert!(
            !home.session(id).path().exists(),
            "the replay kept its session"
        );
        (outcome, exit, replayed)
    };

    // The chosen model's provider has no key: the record answers it all the
    // same.
    let (outcome, exit, replayed) = replay(&id("2-2"), Some(KEYLESS));
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

/// A session that ended with `/quit` under exit-notes replays whole: the
/// replay quits where the recording did, and exit-notes' summary call made
/// then is answered from the record too.
#[test]
fn a_session_quit_under_exit_notes_replays_its_summary_call() {
    let home = scratch_home("session-replay-quit");
    let (recorded_root, replay_root) = (home.root().join("recorded"), home.root().join("replay"));
    let file = home.root().join("a.txt");
    let id = |id: &str| id.parse().unwrap_or_else(|_| SessionId::generate());
    let (recorded_id, replay_id) = (id("4-1"), id("4-2"));
    let path = file.to_string_lossy();
    let recorded = record(&home, &recorded_id, &path, connect, Some(&recorded_root));
    assert_eq!(recorded.len(), 4, "{recorded:?}");
    std::fs::remove_file(&file).ok();
    let summarized = notes_in(&recorded_root);
    assert!(
        summarized.iter().any(|note| note.contains(SUMMARY)),
        "{summarized:?}"
    );

    let model = MockCompletionModel::from_stream_turns(Vec::<Vec<MockStreamEvent>>::new());
    let (mut app, wakes) = app_in(&home, &replay_id);
    let dir = home
        .session(&recorded_id)
        .path()
        .to_string_lossy()
        .into_owned();
    app.insert_resource(Invoked {
        args: Invocation {
            flags: vec![(REPLAY.to_owned(), Some(dir))],
            ..Invocation::default()
        },
        terminal: false,
    })
    .add_plugins(SessionReplayPlugin);
    with_agents(&mut app, connect(&model));
    with_exit_notes(&mut app, &replay_root);
    // The turn's two model calls and the summary call, and the write.
    let outcome = run_replay(&mut app, &wakes);
    assert_eq!(outcome, Some((ReplayStatus::Finished, Vec::new(), (3, 1))));
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Messages<AppExit>>().is_empty()
    });
    let exits = app.world().resource::<Messages<AppExit>>();
    let exits: Vec<AppExit> = exits.iter_current_update_messages().cloned().collect();
    assert_eq!(exits, [AppExit::Success]);
    assert_eq!(conversation(&mut app), recorded);
    assert!(model.requests().is_empty(), "the model was called");
    let replayed = notes_in(&replay_root);
    assert!(
        replayed.iter().any(|note| note.contains(SUMMARY)),
        "{replayed:?}"
    );
}

/// Exit-notes' summary calls started.
#[derive(Resource, Default)]
struct SummariesStarted(usize);

/// A session killed without an exit replays to its end, then exits as the
/// replay front does: exit-notes' summary call is not in the record, so it
/// is refused before it starts, and the replay stays clean.
#[test]
fn a_killed_session_replays_without_starting_an_exit_summary() {
    let home = scratch_home("session-replay-killed");
    let replay_root = home.root().join("replay");
    let file = home.root().join("a.txt");
    let id = |id: &str| id.parse().unwrap_or_else(|_| SessionId::generate());
    let (recorded_id, replay_id) = (id("5-1"), id("5-2"));
    let recorded = record(&home, &recorded_id, &file.to_string_lossy(), connect, None);
    std::fs::remove_file(&file).ok();

    let model = MockCompletionModel::from_stream_turns([reply(SUMMARY)]);
    let (mut app, wakes) = app_in(&home, &replay_id);
    let dir = home.session(&recorded_id).path().to_string_lossy().into();
    app.insert_resource(Invoked {
        args: Invocation {
            flags: vec![(REPLAY.to_owned(), Some(dir))],
            ..Invocation::default()
        },
        terminal: false,
    })
    .add_plugins(SessionReplayPlugin);
    with_agents(&mut app, connect(&model));
    with_exit_notes(&mut app, &replay_root);
    app.init_resource::<SummariesStarted>().add_observer(
        |started: On<Add<Started>>,
         summaries: Query<(), With<NotesCall>>,
         mut count: ResMut<SummariesStarted>| {
            count.0 += usize::from(summaries.contains(started.entity));
        },
    );
    let outcome = run_replay(&mut app, &wakes);
    assert_eq!(outcome, Some((ReplayStatus::Finished, Vec::new(), (2, 1))));
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Messages<AppExit>>().is_empty()
    });
    let exits = app.world().resource::<Messages<AppExit>>();
    let exits: Vec<AppExit> = exits.iter_current_update_messages().cloned().collect();
    assert_eq!(exits, [AppExit::Success]);
    assert_eq!(conversation(&mut app), recorded);
    assert_eq!(app.world().resource::<SummariesStarted>().0, 0);
    assert!(model.requests().is_empty(), "the model was called");
    let replay = app.world().get_resource::<Replay>();
    let divergences = replay.map(Replay::divergences).unwrap_or_default();
    assert_eq!(divergences, []);
    let notes = notes_in(&replay_root);
    assert!(
        notes
            .iter()
            .any(|note| note.contains("(summary unavailable)")),
        "{notes:?}"
    );
}

/// A call recorded without a question replays where a guard would now ask
/// and, with nobody to answer, refuse: the record answers it first.
#[test]
fn a_recorded_call_is_answered_before_a_guard_decides_it() {
    let home = scratch_home("session-replay-guarded");
    let file = home.root().join("a.txt");
    let id = "3-1".parse().unwrap_or_else(|_| SessionId::generate());
    let recorded = record(&home, &id, &file.to_string_lossy(), connect, None);
    std::fs::remove_file(&file).ok();
    let source = recorded_source(home.session(&id).path());
    let model = MockCompletionModel::from_stream_turns(Vec::<Vec<MockStreamEvent>>::new());
    let (mut app, wakes) = app_on(&MemoryStore::default());
    with_agents(&mut app, connect(&model));
    if let Ok(source) = source {
        app.insert_resource(source);
    }
    app.add_plugins((
        SessionReplayPlugin,
        corpus_shared::permission_modes::PermissionModesPlugin,
    ));
    let outcome = run_replay(&mut app, &wakes);
    assert_eq!(outcome, Some((ReplayStatus::Finished, Vec::new(), (2, 1))));
    assert_eq!(conversation(&mut app), recorded);
    assert!(!file.exists(), "the write tool ran during the replay");
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(|app: &mut App| {
        app.insert_resource(ReplaySource {
            effects: EffectLog::default(),
            flags: ReplayFlags::default(),
        })
        .add_plugins(SessionReplayPlugin);
    });
}

/// A replay runs with the plugins of the session it replays, the tool
/// guards among them: the record answers before any of them.
#[test]
fn stages_are_ordered_with_the_tool_guards() {
    rig_defaults_test_support::assert_stages_ordered(|app: &mut App| {
        app.insert_resource(ReplaySource {
            effects: EffectLog::default(),
            flags: ReplayFlags::default(),
        })
        .add_plugins((
            SessionReplayPlugin,
            corpus_shared::plan_mode::PlanModePlugin,
            corpus_shared::permission_modes::PermissionModesPlugin,
            corpus_shared::protected_paths::ProtectedPathsPlugin,
            corpus_shared::confirm_destructive::ConfirmDestructivePlugin,
        ));
    });
}

/// The subagent the recorded `task` call starts.
const SUBAGENT_ID: &str = "sub-1";

/// The kernel, the subagents and the write tool, as [`with_agents`] has
/// them: the primary agent on `main`, every subagent on `sub`.
fn with_subagents(app: &mut App, main: &MockCompletionModel, sub: &MockCompletionModel) {
    with_two_models(app, main, sub);
    rig_harness::load::<rig_basics::subagents::SubagentsPlugin>(app, "rig-basics", "");
}

/// [`with_subagents`] without the subagents.
fn with_two_models(app: &mut App, main: &MockCompletionModel, sub: &MockCompletionModel) {
    with_agents(app, None);
    app.insert_resource(Models(Connector::new(Catalog::default())));
    let (main, sub) = (connect(main), connect(sub));
    app.add_observer(
        move |agent: On<Add<Agent>>,
              spawned: Query<(), With<SpawnedBy>>,
              mut commands: Commands| {
            let connection = match spawned.contains(agent.entity) {
                true => sub.clone(),
                false => main.clone(),
            };
            if let Some(connection) = connection {
                let choice = ModelChoice(connection.spec.reference());
                commands.entity(agent.entity).insert((connection, choice));
            }
        },
    );
}

/// The recording's subagent gets a known id, which the scripted `wait`
/// names.
fn known_id(mut spawns: Query<&mut rig_basics::subagents::SpawnSubagent, Added<Admitted>>) {
    for mut spawn in &mut spawns {
        spawn.id = Some(AgentId(SUBAGENT_ID.to_owned()));
    }
}

/// Records, as `recorded_id` in `home`, a session where the main agent
/// starts a subagent with `task` and `wait`s for its report; the primary
/// agent's conversation and the session's directory.
fn record_a_subagent(
    home: &Home,
    recorded_id: &SessionId,
) -> (Vec<serde_json::Value>, std::path::PathBuf) {
    let main = MockCompletionModel::from_stream_turns([
        calls(
            "c1",
            rig_basics::subagents::TASK,
            serde_json::json!({ "description": "Say ok", "prompt": "Reply with ok." }),
        ),
        calls(
            "c2",
            rig_basics::subagents::WAIT,
            serde_json::json!({ "agent": SUBAGENT_ID }),
        ),
        reply("It said ok."),
    ]);
    let sub = MockCompletionModel::from_stream_turns([reply("ok")]);
    let (mut app, wakes) = app_in(home, recorded_id);
    with_subagents(&mut app, &main, &sub);
    app.add_plugins(EffectLogPlugin)
        .add_systems(AgentPass, known_id.in_set(Stage::Prepare));
    app.finish();
    app.update();
    if let Some(agent) = first_agent(app.world_mut()) {
        let ask = Deliver::new(agent, "Ask a subagent.", DeliveryMode::Queue);
        app.world_mut().trigger(ask);
    }
    run_until(&mut app, &wakes, |world| {
        let mut agents = world.try_query_filtered::<&Conversation, Without<SpawnedBy>>();
        agents.as_mut().is_some_and(|agents| {
            agents.iter(world).any(|conversation| {
                let last = conversation.messages().last();
                let text = last.and_then(|last| serde_json::to_string(last).ok());
                text.is_some_and(|text| text.contains("It said ok."))
            })
        })
    });
    app.update();
    assert!(sub.requests().len() == 1, "the subagent was not asked");
    let recorded = conversation(&mut app);
    (recorded, home.session(recorded_id).path().to_owned())
}

/// `--replay` of the session in `dir`, as `replay_id` in `home`.
fn replaying(home: &Home, replay_id: &SessionId, dir: &std::path::Path) -> (App, Receiver<()>) {
    let (mut app, wakes) = app_in(home, replay_id);
    app.insert_resource(Invoked {
        args: Invocation {
            flags: vec![(REPLAY.to_owned(), Some(dir.to_string_lossy().into()))],
            ..Invocation::default()
        },
        terminal: false,
    })
    .add_plugins((ExitPlugin, SessionReplayPlugin));
    (app, wakes)
}

/// A session where the main agent starts a subagent with `task` and
/// `wait`s for its report replays whole: the subagent is started again,
/// with its recorded id, its model call answered from the record, and the
/// wait ends when its report comes, as it did.
#[test]
fn a_session_that_waits_on_a_subagent_replays_whole() {
    let home = scratch_home("session-replay-subagent");
    let id = |id: &str| id.parse().unwrap_or_else(|_| SessionId::generate());
    let (recorded_id, replay_id) = (id("4-1"), id("4-2"));
    let (recorded, recorded_dir) = record_a_subagent(&home, &recorded_id);

    let (empty, none) = (
        MockCompletionModel::from_stream_turns(Vec::<Vec<MockStreamEvent>>::new()),
        MockCompletionModel::from_stream_turns(Vec::<Vec<MockStreamEvent>>::new()),
    );
    let (mut app, wakes) = replaying(&home, &replay_id, &recorded_dir);
    with_subagents(&mut app, &empty, &none);
    let (status, divergences, (models, _)) = run_replay(&mut app, &wakes).unwrap_or_default();
    let report: Vec<String> = divergences.iter().map(ToString::to_string).collect();
    assert_eq!((status, report), (ReplayStatus::Finished, Vec::new()));
    assert_eq!(models, 4);
    assert_eq!(conversation(&mut app), recorded);
    assert!(empty.requests().is_empty() && none.requests().is_empty());
}

/// A recording that started subagents, replayed by a build without
/// `SubagentsPlugin`, stops at once and says what is missing, rather than
/// diverging where the replayed `task` is refused as an unknown tool.
#[test]
fn a_subagent_recording_without_the_subagents_plugin_stops_and_names_it() {
    let home = scratch_home("session-replay-no-subagents");
    let id = |id: &str| id.parse().unwrap_or_else(|_| SessionId::generate());
    let (recorded_id, replay_id) = (id("5-1"), id("5-2"));
    let (_, recorded_dir) = record_a_subagent(&home, &recorded_id);

    let none = MockCompletionModel::from_stream_turns(Vec::<Vec<MockStreamEvent>>::new());
    let (mut app, wakes) = replaying(&home, &replay_id, &recorded_dir);
    with_two_models(&mut app, &none, &none);
    let (status, divergences, (models, _)) = run_replay(&mut app, &wakes).unwrap_or_default();
    let report: Vec<String> = divergences.iter().map(ToString::to_string).collect();
    assert_eq!(status, ReplayStatus::Stopped);
    assert_eq!(models, 0, "nothing was replayed");
    assert!(
        report.iter().any(|line| line.contains("SubagentsPlugin")),
        "{report:?}"
    );
}
