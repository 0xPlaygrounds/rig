use std::collections::HashMap;
use std::io;

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_ecs::reflect::{AppTypeRegistry, ReflectComponent};
use bevy_reflect::serde::TypedReflectSerializer;
use bevy_reflect::{Reflect, TypePath};
use rig_cassette::journal::{JournalStore, MemoryStore};
use rig_core::completion::{Message, Reasoning, Usage};
use rig_core::message::UserContent;
use serde_json::Value;

use crate::AgentPlugin;
use crate::agent::{
    Agent, AgentId, Conversation, Entry, Halt, Interrupt, LastUsage, Notice, STOPPED, SystemPrompt,
    ToolAccess, TurnOf, TurnOutcome,
};
use crate::effects::Handler;
use crate::inbox::Origin;
use crate::inbox::{Deliver, DeliveryMode};
use crate::journal::tests::commit_message;
use crate::journal::{
    Commit, Committed, JournalPlugin, ReflectSaved, SessionLog, SessionRestored, SessionStore,
};
use crate::model::{Connection, Effort, ModelChoice};
use crate::restore::{INTERRUPTED, Restored};
use crate::stage::{AgentPass, HoldOf, Settled, Stage, Undecided, Unreviewed};
use crate::tools::{AppToolsExt, Footprint, ToolOptions, ToolRequest};
use crate::turn::{RetryVerdict, TurnCall};
use rig_core::catalog::Catalog;
use rig_core::error::retry::Verdict;
use rig_core::operation::Completion;
use rig_core::serve::ErasedHandler;
use rig_core::serve::adapters::ModelAdapter;
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_core::tool::{PortableTool, ToolExecutionError};

/// An app on `store`, after its first frame restored the session. Like
/// the harness, it only warns about a command on a despawned entity. The
/// store comes after the plugins: any plugin may insert it.
fn app(store: &MemoryStore) -> App {
    let mut app = App::new();
    app.set_error_handler(bevy_ecs::error::warn)
        .add_plugins((AgentPlugin, JournalPlugin))
        .insert_resource(SessionStore::new(store.clone()));
    app.finish();
    app.update();
    app
}

fn first_agent(app: &mut App) -> Option<Entity> {
    let mut agents = app.world_mut().query_filtered::<Entity, With<Agent>>();
    agents.iter(app.world()).next()
}

fn say(app: &mut App, agent: Entity, message: Message) {
    commit_message(app.world_mut(), agent, message);
}

/// A plugin's per-agent counter, as the plugin guide's example keeps one.
#[derive(Component, Reflect, Default)]
#[reflect(Component, Saved)]
struct ToolCounts(HashMap<String, u32>);

/// A plugin's count across the session, saved with it.
#[derive(Resource, Reflect, Default)]
#[reflect(Resource, Saved)]
struct Compactions(u32);

/// Every saved component of the first agent, by type path, as reflection
/// writes it.
fn saved(app: &mut App) -> HashMap<String, Value> {
    let agent = first_agent(app);
    let world = app.world();
    let registry = world.resource::<AppTypeRegistry>().read();
    let entity = agent.and_then(|agent| world.get_entity(agent).ok());
    registry
        .iter_with_data::<ReflectSaved>()
        .filter_map(|(registration, _)| {
            let component = registration.data::<ReflectComponent>()?.reflect(entity?)?;
            let value = TypedReflectSerializer::new(component.as_partial_reflect(), &registry);
            let path = registration.type_info().type_path().to_owned();
            // A `null` is saved as absent.
            let value = serde_json::to_value(value)
                .ok()
                .filter(|value| !value.is_null())?;
            Some((path, value))
        })
        .collect()
}

#[test]
fn a_restored_session_has_its_saved_components_and_resources_and_the_messages_its_summary_kept() {
    let store = MemoryStore::default();
    let mut first = app(&store);
    first.world_mut().insert_resource(Compactions(1));
    first.update();
    assert_eq!(
        store.agents().ok(),
        Some(Vec::new()),
        "nothing said, nothing kept"
    );
    let agent = first_agent(&mut first);
    assert!(agent.is_some());
    let Some(agent) = agent else { return };
    let usage = Usage::new().input_tokens(10).output_tokens(5);
    let counts = |reads| ToolCounts(HashMap::from([("read".to_owned(), reads)]));
    first.world_mut().entity_mut(agent).insert((
        ModelChoice("ollama/deepseek-v4-flash".to_owned()),
        Effort(Some(Reasoning::Off)),
        SystemPrompt("You review code.".to_owned()),
        ToolAccess::Only(vec!["read".to_owned()]),
        LastUsage(Some(usage)),
        counts(1),
    ));
    for n in 0..4 {
        say(&mut first, agent, Message::user(format!("question {n}")));
        say(&mut first, agent, Message::assistant(format!("answer {n}")));
        // The newest value wins.
        first.world_mut().entity_mut(agent).insert(counts(n + 1));
        first.world_mut().insert_resource(Compactions(n + 2));
        first.update();
    }
    let summarize =
        |In(agent): In<Entity>, mut agents: Query<&mut Conversation>, mut commit: Commit| {
            if let Ok(mut conversation) = agents.get_mut(agent) {
                let summary = Message::user("The user asked three questions.");
                let summary = Entry::from(summary, Origin::plugin("compaction"));
                commit.splice(agent, &mut conversation, ..6, [summary]);
            }
        };
    assert!(
        first
            .world_mut()
            .run_system_cached_with(summarize, agent)
            .is_ok()
    );
    first.update();
    let live = first.world().get::<Conversation>(agent);
    let live = live.map(|conversation| conversation.messages().to_vec());
    let before = saved(&mut first);
    assert_eq!(before.len(), 6, "{before:?}");
    let newest = serde_json::json!({ "read": 4 });
    assert_eq!(before.get(ToolCounts::type_path()), Some(&newest));
    let id = first.world().get::<AgentId>(agent).cloned();
    drop(first);
    let gone =
        r#"{"seq":99,"n":99,"t":0,"type":"component","component":"a_plugin::Gone","value":1}"#;
    let id = id.unwrap_or_default().0;
    assert!(store.append(&id, format!("{gone}\n").as_bytes()).is_ok());

    let mut second = app(&store);
    // What was saved before the summary still comes back; the resource,
    // which this app had not inserted, too.
    assert_eq!(saved(&mut second), before);
    let compactions = second.world().get_resource::<Compactions>();
    assert_eq!(compactions.map(|count| count.0), Some(5));
    let agent = first_agent(&mut second);
    let world = second.world();
    let notices = world.resource::<Messages<Notice>>();
    let mut cursor = notices.get_cursor();
    let mut texts = cursor.read(notices).map(|notice| notice.text.as_str());
    assert!(texts.any(|text| text.contains("a_plugin::Gone")));
    let kept = agent.and_then(|agent| world.get::<Conversation>(agent));
    let kept = kept.map(|conversation| conversation.messages().to_vec());
    assert_eq!(kept, live);
    // The summary went into the first kept question, so user and model
    // still take turns.
    let summary = ["The user asked three questions.", "question 3"].map(UserContent::text);
    let summary = Message::User {
        content: summary.into(),
    };
    assert_eq!(
        kept.unwrap_or_default(),
        [summary, Message::assistant("answer 3")]
    );
}

#[test]
fn a_restored_model_that_cannot_be_used_is_reported_once() {
    let store = MemoryStore::default();
    let mut first = app(&store);
    let agent = first_agent(&mut first);
    assert!(agent.is_some(), "the first frame spawns an agent");
    let Some(agent) = agent else { return };
    let saved = (
        ModelChoice("nobody/none".to_owned()),
        Effort(Some(Reasoning::Off)),
    );
    first.world_mut().entity_mut(agent).insert(saved);
    say(&mut first, agent, Message::user("question"));
    first.update();
    drop(first);

    let mut second = app(&store);
    let agent = first_agent(&mut second);
    let world = second.world();
    let notices = world.resource::<Messages<Notice>>();
    let mut cursor = notices.get_cursor();
    let texts: Vec<_> = cursor.read(notices).map(|notice| &notice.text).collect();
    let refused = texts.iter().filter(|text| text.starts_with("Cannot use"));
    assert_eq!(refused.count(), 1, "{texts:?}");
    let connected = agent.and_then(|agent| world.get::<Connection>(agent));
    assert!(connected.is_none());
}

#[test]
fn every_change_the_log_records_is_committed() {
    let mut app = app(&MemoryStore::default());
    // No model answers it, and a reviewer takes that as final, so the
    // message is taken out; a note is left halted.
    let finally = |calls: Query<Entity, (Unreviewed, With<TurnCall>)>, mut commands: Commands| {
        for call in &calls {
            commands.entity(call).insert(RetryVerdict(Verdict::Final));
        }
    };
    app.add_systems(AgentPass, finally.in_set(Stage::Review));
    let agent = first_agent(&mut app);
    assert!(agent.is_some());
    let Some(agent) = agent else { return };
    let hello = Deliver::new(agent, "hello", DeliveryMode::Steer);
    app.world_mut().trigger(hello);
    app.update();
    let note = Deliver::new(agent, "a note", DeliveryMode::Note);
    app.world_mut().trigger(note);
    app.update();
    let committed = app.world().resource::<Messages<Committed>>();
    let mut cursor = committed.get_cursor();
    let changes = cursor.read(committed).map(|change| match change {
        Committed::Message { .. } => "message",
        Committed::Splice { .. } => "splice",
        Committed::Halt { .. } => "halt",
        Committed::Retired { .. } => "retired",
    });
    assert_eq!(
        changes.collect::<Vec<_>>(),
        ["message", "splice", "message", "halt"]
    );
}

/// Keeps every restored agent idle: its restored turn is stopped before
/// anything of it starts.
fn stay_idle(turns: Query<Entity, (With<Restored>, With<TurnOf>)>, mut commands: Commands) {
    for turn in &turns {
        let stopped = TurnOutcome::Stopped(Some("restart".to_owned()));
        commands.entity(turn).insert_if_new(stopped);
    }
}

/// The messages of the one agent of the session in `store`, restored and
/// kept idle, after `act` ran on it.
fn session(store: &MemoryStore, act: impl FnOnce(&mut World, Entity)) -> Vec<Message> {
    let mut app = App::new();
    app.insert_resource(SessionStore::new(store.clone()))
        .add_plugins((AgentPlugin, JournalPlugin))
        .add_systems(AgentPass, stay_idle.in_set(Stage::Gate));
    app.finish();
    app.update();
    let world = app.world_mut();
    let Ok(agent) = world.query_filtered::<Entity, With<Agent>>().single(world) else {
        return Vec::new();
    };
    act(world, agent);
    world.resource::<SessionLog>().flush();
    let conversation = world.get::<Conversation>(agent);
    conversation.map_or_else(Vec::new, |conversation| conversation.messages().to_vec())
}

#[test]
fn a_message_after_a_stopped_request_says_it_was_stopped_unless_it_was_retried() {
    let store = MemoryStore::default();
    let texts = [
        "run sleep 120 && echo finished",
        "what is 2 + 2?",
        "and 3 + 3?",
    ];
    // The second is asked for again, and that turn fails and keeps it.
    let keep = |In(agent): In<Entity>, mut agents: Query<&mut Conversation>, mut commit: Commit| {
        if let Ok(mut conversation) = agents.get_mut(agent) {
            assert!(commit.resume(agent, &mut conversation));
            commit.halt(agent, &mut conversation, Halt::Kept);
        }
    };
    let live = session(&store, |world, agent| {
        for (at, text) in texts.into_iter().enumerate() {
            world.trigger(Deliver::new(agent, text, DeliveryMode::Steer));
            // A pass accepts the message and starts the turn.
            world.run_schedule(AgentPass);
            world.trigger(Interrupt::new(agent));
            world.flush();
            if at == 1 {
                assert!(world.run_system_cached_with(keep, agent).is_ok());
            }
        }
    });
    let [first, second, third] = texts;
    let content = [first, STOPPED, second, third].map(UserContent::text);
    assert_eq!(
        live,
        [Message::User {
            content: content.into()
        }]
    );
    assert_eq!(session(&store, |_, _| {}), live);
}

/// A store on a full disk: it reads, but every log write fails.
struct Full;

impl JournalStore for Full {
    fn agents(&self) -> io::Result<Vec<String>> {
        Ok(Vec::new())
    }
    fn read(&self, _: &str) -> io::Result<Vec<u8>> {
        Ok(Vec::new())
    }
    fn truncate(&self, _: &str, _: u64) -> io::Result<()> {
        Ok(())
    }
    fn append(&self, _: &str, _: &[u8]) -> io::Result<()> {
        Err(io::Error::other("disk full"))
    }
    fn put_blob(&self, _: &str, _: &[u8]) -> io::Result<()> {
        Ok(())
    }
    fn blob(&self, _: &str) -> io::Result<Vec<u8>> {
        Ok(Vec::new())
    }
}

/// A failed log write stops the logging, with a notice, but the session
/// stays restored: what waits for that, such as remembering a model the
/// user picks, still runs.
#[test]
fn a_failed_log_write_leaves_the_session_restored() {
    let mut app = App::new();
    app.add_plugins((AgentPlugin, JournalPlugin))
        .insert_resource(SessionStore::new(Full));
    app.finish();
    app.update();
    let agent = first_agent(&mut app);
    assert!(agent.is_some());
    let Some(agent) = agent else { return };
    say(&mut app, agent, Message::user("hello"));
    app.update();
    let notices = app.world().resource::<Messages<Notice>>();
    let stopped = notices
        .iter_current_update_messages()
        .any(|notice| notice.text.starts_with("The session is no longer saved"));
    assert!(stopped);
    assert!(app.world().contains_resource::<SessionRestored>());
}

/// Reads, so a restart runs its unanswered calls again.
struct Look;

impl PortableTool for Look {
    const NAME: &'static str = "look";
    type Args = serde_json::Value;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        "Looks.".to_owned()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({ "type": "object" })
    }

    async fn call(&self, _: serde_json::Value) -> Result<String, ToolExecutionError> {
        Ok("seen".to_owned())
    }
}

/// May change anything, so a restart answers its unanswered calls as
/// interrupted.
struct Touch;

impl PortableTool for Touch {
    const NAME: &'static str = "touch";
    type Args = serde_json::Value;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        "Touches.".to_owned()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({ "type": "object" })
    }

    async fn call(&self, _: serde_json::Value) -> Result<String, ToolExecutionError> {
        Ok("touched".to_owned())
    }
}

/// Each tool call a Gate system saw, by tool, and whether a restart
/// carried it on.
#[derive(Resource, Default)]
struct Gated(Vec<(String, bool)>);

fn note_gated(calls: Query<(&ToolRequest, Has<Restored>), Undecided>, mut gated: ResMut<Gated>) {
    for (request, restored) in &calls {
        let name = request.call.function.name.as_str().to_owned();
        gated.0.push((name, restored));
    }
}

/// Holds every `touch` call, as an approval nobody gave.
fn hold_touches(calls: Query<(Entity, &ToolRequest), Undecided>, mut commands: Commands) {
    for (call, request) in &calls {
        if request.call.function.name.as_str() == "touch" {
            commands.spawn(HoldOf(call));
        }
    }
}

/// How each turn ended.
#[derive(Resource, Default)]
struct Ended(Vec<TurnOutcome>);

/// The session in `store` on `model`, with `look`, `touch`, a gate that
/// notes the calls it sees and what `configure` adds, after its first
/// frame.
fn tools_app(
    store: &MemoryStore,
    model: &MockCompletionModel,
    configure: impl FnOnce(&mut App),
) -> App {
    let spec = Catalog::builtin().resolve("deepseek/deepseek-flash");
    let connection = spec.ok().map(|spec| {
        let spec = spec.shared();
        let handler = ModelAdapter::<Completion>::new(spec.reference(), model.clone());
        let handler = Handler(ErasedHandler::new(handler));
        Connection { spec, handler }
    });
    let read_only = ToolOptions {
        footprint: Some(Footprint::ReadOnly),
        ..ToolOptions::default()
    };
    let mut app = App::new();
    app.add_plugins((TaskPoolPlugin::default(), AgentPlugin, JournalPlugin))
        .insert_resource(SessionStore::new(store.clone()))
        .init_resource::<Gated>()
        .init_resource::<Ended>()
        .add_tool_with(Look, read_only)
        .add_tool(Touch)
        .add_systems(AgentPass, note_gated.in_set(Stage::Gate))
        .add_observer(move |added: On<Add<Agent>>, mut commands: Commands| {
            if let Some(connection) = connection.clone() {
                commands.entity(added.entity).insert(connection);
            }
        })
        .add_observer(
            |settled: On<Add<Settled>>, turns: Query<&TurnOutcome>, mut ended: ResMut<Ended>| {
                ended.0.extend(turns.get(settled.entity).cloned());
            },
        );
    configure(&mut app);
    app.finish();
    app.update();
    app
}

/// Runs frames until `done`, for at most ten seconds; whether it was done.
fn run_until(app: &mut App, done: impl Fn(&World) -> bool) -> bool {
    let started = std::time::Instant::now();
    while started.elapsed() < std::time::Duration::from_secs(10) {
        app.update();
        if done(app.world()) {
            return true;
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    false
}

/// A session whose agent's model asked to `touch`, then `look`, and that
/// exited while a gate held the `touch` call, the `look` call waiting
/// behind it.
fn held_at_exit() -> MemoryStore {
    let store = MemoryStore::default();
    let call = |id: &str, tool: &str| MockStreamEvent::tool_call(id, tool, serde_json::json!({}));
    let model = MockCompletionModel::from_stream_turns([vec![
        call("t1", "touch"),
        call("l1", "look"),
        MockStreamEvent::final_response_with_default_usage(),
    ]]);
    let mut app = tools_app(&store, &model, |app| {
        app.add_systems(AgentPass, hold_touches.in_set(Stage::Gate));
    });
    if let Some(agent) = first_agent(&mut app) {
        app.world_mut()
            .trigger(Deliver::new(agent, "go", DeliveryMode::Steer));
    }
    let held = |world: &World| {
        let gated = &world.resource::<Gated>().0;
        gated.iter().any(|(tool, _)| tool == "touch")
    };
    assert!(run_until(&mut app, held));
    app.world_mut().write_message(AppExit::Success);
    app.update();
    store
}

/// The conversation of the first agent, as JSON.
fn said(app: &mut App) -> String {
    let agent = first_agent(app);
    let conversation = agent.and_then(|agent| app.world().get::<Conversation>(agent));
    let messages = conversation.map(|conversation| conversation.messages().to_vec());
    serde_json::to_string(&messages).unwrap_or_default()
}

#[test]
fn a_restart_gates_its_reruns_again_and_answers_a_held_change_as_interrupted() {
    let store = held_at_exit();
    let model = MockCompletionModel::from_stream_turns([vec![
        MockStreamEvent::text("done"),
        MockStreamEvent::final_response_with_default_usage(),
    ]]);
    let mut app = tools_app(&store, &model, |_| {});
    let ended = |world: &World| !world.resource::<Ended>().0.is_empty();
    assert!(run_until(&mut app, ended));
    // The held change was never gated again: it was answered as
    // interrupted. The read-only call was gated again, as restored work.
    let gated = app.world().resource::<Gated>().0.clone();
    assert_eq!(gated, [("look".to_owned(), true)]);
    let said = said(&mut app);
    assert!(
        said.contains(INTERRUPTED) && said.contains("seen"),
        "{said}"
    );
    assert_eq!(model.request_count(), 1);
    let answered = matches!(
        app.world().resource::<Ended>().0.as_slice(),
        [TurnOutcome::Answered(_)]
    );
    assert!(answered, "{:?}", app.world().resource::<Ended>().0);
}

#[test]
fn a_restored_turn_a_gate_stops_starts_nothing() {
    let store = held_at_exit();
    let model = MockCompletionModel::from_stream_turns(Vec::<Vec<MockStreamEvent>>::new());
    let mut app = tools_app(&store, &model, |app| {
        app.add_systems(AgentPass, stay_idle.in_set(Stage::Gate));
    });
    let ended = |world: &World| !world.resource::<Ended>().0.is_empty();
    assert!(run_until(&mut app, ended));
    let stopped = matches!(app.world().resource::<Ended>().0.as_slice(),
        [TurnOutcome::Stopped(Some(why))] if why == "restart");
    assert!(stopped, "{:?}", app.world().resource::<Ended>().0);
    // The rerun never ran, and the model was not asked.
    let said = said(&mut app);
    assert!(
        said.contains("stopped: restart") && !said.contains("seen"),
        "{said}"
    );
    assert_eq!(model.request_count(), 0);
}
