use std::collections::HashMap;
use std::io;

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_ecs::reflect::{AppTypeRegistry, ReflectComponent};
use bevy_reflect::prelude::{ReflectDefault, ReflectDeserialize, ReflectSerialize};
use bevy_reflect::serde::TypedReflectSerializer;
use bevy_reflect::{Reflect, TypePath};
use rig_core::completion::{Message, Reasoning, Usage};
use rig_core::conversation::{Author, Change, Entry, Halt, MessageId, STOPPED};
use rig_memory::{Files, LogWriteError, MemoryStore, Store};
use serde_json::Value;

use crate::AgentPlugin;
use crate::agent::{
    Agent, AgentId, Conversation, LastUsage, Notice, NoticeLevel, StopTurn, SystemPrompt,
    ToolAccess, Turn, TurnOf, TurnOutcome,
};
use crate::effects::Handler;
use crate::inbox::{Deliver, DeliveryMode, Incoming, Recalled, Unsent};
use crate::model::{Connection, ModelChoice, ReasoningSetting};
use crate::restore::{Interrupted, RESTART, Restored};
use crate::retry::Verdict;
use crate::session_log::tests::{edit, push_message};
use crate::session_log::{
    ConversationChanged, ReflectSaved, SessionLog, SessionLogPlugin, SessionRestored, SessionStore,
};
use crate::stage::{
    AgentPass, Decision, ForAgent, HoldOf, Outcome, PrepareSystems, ReadyToSettle, ReadyToStart,
    Settled, Stage, Started,
};
use crate::tools::{AppToolsExt, ToolRequest};
use crate::turn::{RetryVerdict, TurnCall};
use rig_core::catalog::Catalog;
use rig_core::effect::Decider;
use rig_core::operation::Completion;
use rig_core::serve::ErasedHandler;
use rig_core::serve::adapters::ModelAdapter;
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_core::tool::{Scope, Tool, ToolContext, ToolExecutionError, Touches};

/// An app on `store`, after its first frame restored the session. Like
/// the harness, it only warns about a command on a despawned entity. The
/// store comes after the plugins: any plugin may insert it.
fn app(store: &MemoryStore) -> App {
    let mut app = App::new();
    app.set_error_handler(bevy_ecs::error::warn)
        .add_plugins((AgentPlugin, SessionLogPlugin))
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
    push_message(app.world_mut(), agent, message);
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
        store.logs().ok(),
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
        ReasoningSetting(Some(Reasoning::Off)),
        SystemPrompt("You review code.".to_owned()),
        ToolAccess::Only(vec!["read".to_owned()]),
        LastUsage(Some(rig_memory::Anchor {
            usage,
            through: rig_core::conversation::MessageId::default(),
        })),
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
    edit(first.world_mut(), agent, |conversation| {
        let summary = Message::user("The user asked three questions.");
        let compaction = Author::plugin("compaction");
        let summary = Entry::new(summary).by(compaction.clone());
        conversation.splice(..6, [summary], compaction);
    });
    first.update();
    let live = first.world().get::<Conversation>(agent);
    let live = live.map(|conversation| conversation.messages().to_vec());
    let before = saved(&mut first);
    assert_eq!(before.len(), 6, "{before:?}");
    let newest = serde_json::json!({ "read": 4 });
    assert_eq!(before.get(ToolCounts::type_path()), Some(&newest));
    // The model is saved under the key a library program reads, as a string.
    assert_eq!(ModelChoice::type_path(), rig_memory::MODEL);
    let model = serde_json::json!("ollama/deepseek-v4-flash");
    assert_eq!(before.get(rig_memory::MODEL), Some(&model));
    let id = first.world().get::<AgentId>(agent).cloned();
    drop(first);
    let gone = r#"{"seq":99,"n":99,"t":0,"type":"state","key":"a_plugin::Gone","value":1}"#;
    let id = id.unwrap_or_default().0;
    let len = store.read_log(&id).map_or(0, |log| log.len() as u64);
    assert!(
        store
            .append_log(&id, len, format!("{gone}\n").as_bytes())
            .is_ok()
    );

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
    // The summary is its own entry, before the question it kept.
    let summary = Message::user("The user asked three questions.");
    assert_eq!(
        kept.unwrap_or_default(),
        [
            summary,
            Message::user("question 3"),
            Message::assistant("answer 3")
        ]
        .map(std::sync::Arc::new)
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
        ReasoningSetting(Some(Reasoning::Off)),
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
    // message is taken out, between the turn's record opening and ending;
    // a note is left halted.
    let finally = |calls: Query<Entity, (ReadyToSettle, With<TurnCall>)>,
                   mut commands: Commands| {
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
    let note = Deliver::new(agent, "a note", DeliveryMode::Inform);
    app.world_mut().trigger(note);
    app.update();
    let committed = app.world().resource::<Messages<ConversationChanged>>();
    let mut cursor = committed.get_cursor();
    let changes = cursor
        .read(committed)
        .map(|change| match change.change.change {
            Change::Push { .. } => "push",
            Change::Splice { .. } => "splice",
            Change::Halt { .. } => "halt",
            Change::Aside { .. } => "aside",
            Change::Reset { .. } => "reset",
        });
    assert_eq!(
        changes.collect::<Vec<_>>(),
        ["push", "aside", "splice", "aside", "push", "halt"]
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
        .add_plugins((AgentPlugin, SessionLogPlugin))
        .add_systems(AgentPass, stay_idle.in_set(Stage::Gate));
    app.finish();
    app.update();
    let world = app.world_mut();
    let Ok(agent) = world.query_filtered::<Entity, With<Agent>>().single(world) else {
        return Vec::new();
    };
    act(world, agent);
    assert!(world.resource::<SessionLog>().flush().is_ok());
    let conversation = world.get::<Conversation>(agent);
    conversation.map_or_else(Vec::new, |conversation| {
        conversation
            .messages()
            .iter()
            .map(|message| (**message).clone())
            .collect()
    })
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
    let keep = |conversation: &mut Conversation| {
        assert!(conversation.resume());
        conversation.halt(Halt::Kept);
    };
    let live = session(&store, |world, agent| {
        for (at, text) in texts.into_iter().enumerate() {
            world.trigger(Deliver::new(agent, text, DeliveryMode::Steer));
            // A pass accepts the message and starts the turn.
            world.run_schedule(AgentPass);
            world.trigger(StopTurn::new(agent));
            world.flush();
            if at == 1 {
                edit(world, agent, keep);
            }
        }
    });
    let [first, second, third] = texts;
    let said = [first, STOPPED, second, third].map(Message::user);
    assert_eq!(live, said);
    assert_eq!(session(&store, |_, _| {}), live);
}

/// A store on a full disk: it reads, but every log write fails.
#[derive(Clone, Copy)]
struct Full;

impl Files for Full {
    fn write_file(&self, _: &str, _: &[u8]) -> io::Result<()> {
        Err(io::Error::other("disk full"))
    }
    fn append_file(&self, _: &str, _: &[u8]) -> io::Result<()> {
        Err(io::Error::other("disk full"))
    }
    fn read_file(&self, _: &str) -> io::Result<Vec<u8>> {
        Err(io::ErrorKind::NotFound.into())
    }
    fn remove_file(&self, _: &str) -> io::Result<()> {
        Ok(())
    }
    fn files(&self) -> io::Result<Vec<String>> {
        Ok(Vec::new())
    }
}

impl Store for Full {
    fn logs(&self) -> io::Result<Vec<String>> {
        Ok(Vec::new())
    }
    fn read_log(&self, _: &str) -> io::Result<Vec<u8>> {
        Err(io::ErrorKind::NotFound.into())
    }
    fn append_log(&self, _: &str, _: u64, _: &[u8]) -> Result<(), LogWriteError> {
        Err(io::Error::other("disk full").into())
    }
    fn replace_log(&self, _: &str, _: u64, _: &[u8]) -> Result<(), LogWriteError> {
        Err(io::Error::other("disk full").into())
    }
    fn destroy(&self) -> io::Result<()> {
        Ok(())
    }
}

/// A failed log write stops the logging, with a notice, but the session
/// stays restored: what waits for that, such as remembering a model the
/// user picks, still runs.
#[test]
fn a_failed_log_write_leaves_the_session_restored() {
    let mut app = App::new();
    app.add_plugins((AgentPlugin, SessionLogPlugin))
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

/// Reads one file, as each call says, so a restart runs its unanswered
/// calls again.
struct Look;

impl Tool for Look {
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

    /// Each call reads one file, whatever its tool's annotations say.
    fn footprint(&self, _: &serde_json::Value) -> Touches {
        Touches::Reads(Scope::Paths(vec!["/work/a.rs".into()]))
    }

    async fn call(
        &self,
        _: &mut ToolContext,
        _: serde_json::Value,
    ) -> Result<String, ToolExecutionError> {
        Ok("seen".to_owned())
    }
}

/// May change anything, so a restart answers its unanswered calls as
/// interrupted.
struct Touch;

impl Tool for Touch {
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

    async fn call(
        &self,
        _: &mut ToolContext,
        _: serde_json::Value,
    ) -> Result<String, ToolExecutionError> {
        Ok("touched".to_owned())
    }
}

/// Each tool call a Gate system saw, by tool, and whether a restart
/// carried it on.
#[derive(Resource, Default)]
struct Gated(Vec<(String, bool)>);

fn note_gated(calls: Query<(&ToolRequest, Has<Restored>), ReadyToStart>, mut gated: ResMut<Gated>) {
    for (request, restored) in &calls {
        let name = request.call.function.name.as_str().to_owned();
        gated.0.push((name, restored));
    }
}

/// Holds every `touch` call, as an approval nobody gave.
fn hold_touches(calls: Query<(Entity, &ToolRequest), ReadyToStart>, mut commands: Commands) {
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
    store: &(impl Store + Clone),
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
    let mut app = App::new();
    app.add_plugins((TaskPoolPlugin::default(), AgentPlugin, SessionLogPlugin))
        .insert_resource(SessionStore::new(store.clone()))
        .init_resource::<Gated>()
        .init_resource::<Ended>()
        .add_tool(Look)
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
fn a_restart_gates_every_call_again_and_asks_about_a_held_change_again() {
    let store = held_at_exit();
    let model = MockCompletionModel::from_stream_turns([vec![
        MockStreamEvent::text("done"),
        MockStreamEvent::final_response_with_default_usage(),
    ]]);
    let mut app = tools_app(&store, &model, |_| {});
    let ended = |world: &World| !world.resource::<Ended>().0.is_empty();
    assert!(run_until(&mut app, ended));
    // The held change never started, so it is restored work every gate
    // sees again, as is the read-only call.
    let gated = app.world().resource::<Gated>().0.clone();
    assert_eq!(
        gated,
        [("touch".to_owned(), true), ("look".to_owned(), true)]
    );
    let said = said(&mut app);
    assert!(
        !said.contains(&interrupted()) && said.contains("touched") && said.contains("seen"),
        "{said}"
    );
    assert_eq!(model.request_count(), 1);
    let answered = matches!(
        app.world().resource::<Ended>().0.as_slice(),
        [TurnOutcome::Answered(_)]
    );
    assert!(answered, "{:?}", app.world().resource::<Ended>().0);
}

/// Once the session can no longer be saved, a call that may change
/// something is answered without running, since a restart could not tell
/// that it ran; a call that only reads still runs, and the user is told
/// once.
#[test]
fn no_changing_call_starts_once_the_session_cannot_be_saved() {
    let call = |id: &str, tool: &str| MockStreamEvent::tool_call(id, tool, serde_json::json!({}));
    let model = MockCompletionModel::from_stream_turns([
        vec![
            call("t1", "touch"),
            call("l1", "look"),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        vec![
            MockStreamEvent::text("done"),
            MockStreamEvent::final_response_with_default_usage(),
        ],
    ]);
    let mut app = tools_app(&Full, &model, |app| {
        // The log reports its failure at the end of the frame.
        app.init_resource::<Errors>()
            .add_systems(Last, note_errors.after(crate::WriteLogs));
    });
    if let Some(agent) = first_agent(&mut app) {
        app.world_mut()
            .trigger(Deliver::new(agent, "go", DeliveryMode::Steer));
    }
    let ended = |world: &World| !world.resource::<Ended>().0.is_empty();
    assert!(run_until(&mut app, ended));
    let said = said(&mut app);
    let refused = "Not run: the session can no longer be saved (";
    assert!(
        said.contains(refused) && !said.contains("touched") && said.contains("seen"),
        "{said}"
    );
    let errors = &app.world().resource::<Errors>().0;
    assert!(
        matches!(errors.as_slice(), [error] if error.starts_with("The session is no longer saved")),
        "{errors:?}"
    );
}

/// The text of every error notice.
#[derive(Resource, Default)]
struct Errors(Vec<String>);

fn note_errors(mut notices: MessageReader<Notice>, mut errors: ResMut<Errors>) {
    let failed = notices
        .read()
        .filter(|notice| notice.level == NoticeLevel::Error);
    errors.0.extend(failed.map(|notice| notice.text.clone()));
}

/// May change anything, and never answers.
struct Wait;

impl Tool for Wait {
    const NAME: &'static str = "wait";
    type Args = serde_json::Value;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        "Waits.".to_owned()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({ "type": "object" })
    }

    async fn call(
        &self,
        _: &mut ToolContext,
        _: serde_json::Value,
    ) -> Result<String, ToolExecutionError> {
        std::future::pending::<()>().await;
        Ok(String::new())
    }
}

/// A session whose agent's model asked to `first` (`touch` or `look`),
/// then `wait`, that crashed while `wait` ran, `first` settled.
fn crashed_mid_reply(first: &str) -> MemoryStore {
    let store = MemoryStore::default();
    let call = |id: &str, tool: &str| MockStreamEvent::tool_call(id, tool, serde_json::json!({}));
    let model = MockCompletionModel::from_stream_turns([vec![
        call("t1", first),
        call("w1", "wait"),
        MockStreamEvent::final_response_with_default_usage(),
    ]]);
    let mut app = tools_app(&store, &model, |app| {
        app.add_tool(Wait);
    });
    if let Some(agent) = first_agent(&mut app) {
        app.world_mut()
            .trigger(Deliver::new(agent, "go", DeliveryMode::Steer));
    }
    let waiting = |world: &World| {
        let started = world
            .iter_entities()
            .filter(|entity| entity.contains::<Started>());
        started
            .filter_map(|entity| entity.get::<ToolRequest>())
            .any(|request| request.call.function.name.as_str() == "wait")
    };
    assert!(run_until(&mut app, waiting));
    // A crash: nothing more is written.
    drop(app);
    store
}

/// Who decided each tool call that settled, by tool.
#[derive(Resource, Default)]
struct Decided(Vec<(String, Option<Decider>)>);

/// [`tools_app`] after a crash mid-reply, with `Wait`, noting who decided
/// each call, with what `configure` adds, run until its turn ended; the
/// conversation as JSON.
fn restart_after_crash(configure: impl FnOnce(&mut App)) -> (App, String, MemoryStore) {
    restart_after(crashed_mid_reply("touch"), configure)
}

/// As [`restart_after_crash`], from `store`.
fn restart_after(
    store: MemoryStore,
    configure: impl FnOnce(&mut App),
) -> (App, String, MemoryStore) {
    let model = MockCompletionModel::from_stream_turns([vec![
        MockStreamEvent::text("done"),
        MockStreamEvent::final_response_with_default_usage(),
    ]]);
    let mut app = tools_app(&store, &model, |app| {
        app.add_tool(Wait).init_resource::<Decided>().add_observer(
            |settled: On<Add<Settled>>,
             calls: Query<(&ToolRequest, &Decision)>,
             mut decided: ResMut<Decided>| {
                if let Ok((request, decision)) = calls.get(settled.entity) {
                    let name = request.call.function.name.as_str().to_owned();
                    decided.0.push((name, decision.by().cloned()));
                }
            },
        );
        configure(app);
    });
    let ended = |world: &World| !world.resource::<Ended>().0.is_empty();
    assert!(run_until(&mut app, ended));
    let said = said(&mut app);
    // Every mark goes with the results.
    let marks = aside_kinds(&mut app);
    assert!(!marks.iter().any(|kind| kind == "rig.call"), "{said}");
    (app, said, store)
}

#[test]
fn a_restart_answers_what_a_crash_interrupted_after_every_gate() {
    let (mut app, said, store) = restart_after_crash(|_| {});
    // `touch` settled before the crash: its result stands, and it did not
    // run again; `wait` was running: it is answered as interrupted. No gate
    // saw either.
    assert!(
        said.contains("touched") && said.contains(&interrupted()),
        "{said}"
    );
    assert!(app.world().resource::<Gated>().0.is_empty());
    let rig = Some(Decider::plugin("rig-ecs"));
    let decided = &app.world().resource::<Decided>().0;
    assert_eq!(
        decided,
        &[("touch".to_owned(), rig.clone()), ("wait".to_owned(), rig)]
    );
    // The saved session has no mark left either.
    app.world_mut().write_message(AppExit::Success);
    app.update();
    let saved = rig_memory::Session::read(&store).map(|session| session.agents);
    let marks = saved.map(|agents| {
        let asides = agents.iter().flat_map(|it| it.conversation.asides());
        asides.filter(|(_, aside)| aside.kind == "rig.call").count()
    });
    assert_eq!(marks.ok(), Some(0usize));
}

/// Answers a restored `wait` that a crash interrupted.
fn answer_waits(mut calls: Query<(&ToolRequest, &Interrupted, &mut Decision), Without<Outcome>>) {
    for (request, interrupted, mut decision) in &mut calls {
        if request.call.function.name.as_str() == "wait"
            && matches!(interrupted, Interrupted::Started)
        {
            let answer = rig_core::tool::ToolResult::skipped("not waited for again");
            decision.decide(Decider::plugin("crash-resume"), answer);
        }
    }
}

#[test]
fn a_plugin_answers_a_call_a_crash_interrupted_first() {
    let (_, said, _) = restart_after_crash(|app| {
        app.add_systems(AgentPass, answer_waits.in_set(Stage::Gate));
    });
    assert!(
        said.contains("not waited for again") && !said.contains(&interrupted()),
        "{said}"
    );
}

/// Lets a restored `touch` run again.
fn rerun_touches(calls: Query<(Entity, &ToolRequest), With<Interrupted>>, mut commands: Commands) {
    for (call, request) in &calls {
        if request.call.function.name.as_str() == "touch" {
            commands.entity(call).remove::<Interrupted>();
        }
    }
}

#[test]
fn a_call_a_plugin_lets_run_again_is_gated_first() {
    let (app, said, _) = restart_after_crash(|app| {
        app.add_systems(AgentPass, rerun_touches.in_set(PrepareSystems::Edit));
    });
    let gated = &app.world().resource::<Gated>().0;
    assert_eq!(gated, &[("touch".to_owned(), true)]);
    let decided = &app.world().resource::<Decided>().0;
    let touch = decided.iter().find(|(tool, _)| tool == "touch");
    assert_eq!(
        touch,
        Some(&("touch".to_owned(), None)),
        "it ran: {decided:?}"
    );
    assert!(
        said.contains("touched") && said.contains(&interrupted()),
        "{said}"
    );
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
        said.contains("Not run: restart.") && !said.contains("seen"),
        "{said}"
    );
    assert_eq!(model.request_count(), 0);
}

/// Where each restored turn began and its index, as Gate saw it.
#[derive(Resource, Default)]
struct Began(Vec<(u64, MessageId)>);

#[test]
fn a_turn_a_crash_cut_short_carries_on_where_it_began() {
    let (mut app, _, _) = restart_after_crash(|app| {
        let note = |turns: Query<&Turn, (With<Restored>, With<TurnOf>)>,
                    mut began: ResMut<Began>| {
            began
                .0
                .extend(turns.iter().map(|turn| (turn.index, turn.start)));
        };
        app.init_resource::<Began>()
            .add_systems(AgentPass, note.in_set(Stage::Gate));
    });
    let agent = first_agent(&mut app);
    let conversation = agent.and_then(|agent| app.world().get::<Conversation>(agent));
    let go = conversation.and_then(|conversation| {
        let mut entries = conversation.entries();
        entries
            .find(|entry| entry.message == &Message::user("go"))
            .map(|entry| entry.id)
    });
    assert!(go.is_some());
    // The same, first turn, carried on from the message that started it,
    // its one record left open by the crash and closed by the restart.
    let began = &app.world().resource::<Began>().0;
    assert_eq!(began.first().copied(), go.map(|go| (0, go)), "{began:?}");
    let records = conversation
        .into_iter()
        .flat_map(|conversation| conversation.turns());
    let records: Vec<_> = records
        .map(|(_, record)| (record.index, Some(record.start), record.end.is_some()))
        .collect();
    assert_eq!(records, [(0, go, true)]);
}

/// The holds a gate put on restored calls while it asked someone.
#[derive(Resource, Default)]
struct Asking(Vec<Entity>);

/// On a call [`ask`] asked about.
#[derive(Component)]
struct Asked;

/// Holds each restored call a crash interrupted while it asks, once.
fn ask(
    calls: Query<Entity, (With<Interrupted>, Without<Outcome>, Without<Asked>)>,
    mut asking: ResMut<Asking>,
    mut commands: Commands,
) {
    for call in &calls {
        commands.entity(call).insert(Asked);
        asking.0.push(commands.spawn(HoldOf(call)).id());
    }
}

#[test]
fn a_call_a_crash_interrupted_waits_while_a_plugin_holds_it() {
    let store = crashed_mid_reply("touch");
    let model = MockCompletionModel::from_stream_turns([vec![
        MockStreamEvent::text("done"),
        MockStreamEvent::final_response_with_default_usage(),
    ]]);
    let mut app = tools_app(&store, &model, |app| {
        app.add_tool(Wait)
            .init_resource::<Asking>()
            .add_systems(AgentPass, ask.in_set(Stage::Gate));
    });
    for _ in 0..5 {
        app.update();
    }
    let world = app.world_mut();
    let mut answered = world.query_filtered::<(), (With<Interrupted>, With<Outcome>)>();
    assert_eq!(
        answered.iter(world).count(),
        0,
        "nothing is answered while held"
    );
    assert!(world.resource::<Ended>().0.is_empty());
    let holds = std::mem::take(&mut world.resource_mut::<Asking>().0);
    assert_eq!(holds.len(), 2);
    for hold in holds {
        world.despawn(hold);
    }
    let ended = |world: &World| !world.resource::<Ended>().0.is_empty();
    assert!(run_until(&mut app, ended));
    let said = said(&mut app);
    assert!(
        said.contains("touched") && said.contains(&interrupted()),
        "{said}"
    );
}

#[test]
fn a_read_that_finished_before_a_crash_keeps_its_result() {
    let (app, said, _) = restart_after(crashed_mid_reply("look"), |_| {});
    // `look` reads only, and settled while `wait` ran: answered with its
    // result, so no gate saw it and it did not run again.
    assert!(
        said.contains("seen") && said.contains(&interrupted()),
        "{said}"
    );
    assert!(app.world().resource::<Gated>().0.is_empty());
    let rig = Some(Decider::plugin("rig-ecs"));
    let decided = &app.world().resource::<Decided>().0;
    assert_eq!(
        decided,
        &[("look".to_owned(), rig.clone()), ("wait".to_owned(), rig)]
    );
}

/// Keeps every message a turn did not send in the inbox.
fn keep_unsent(unsent: Query<Entity, Added<Unsent>>, mut commands: Commands) {
    for message in &unsent {
        commands.entity(message).remove::<Unsent>();
    }
}

/// A session that exited with "also the docs" in its agent's inbox: while
/// a gate held the agent's `touch` call, or, `stopped`, after the user
/// stopped that turn and a plugin kept the message.
fn pending_at_exit(stopped: bool) -> MemoryStore {
    let store = MemoryStore::default();
    let call = |id: &str, tool: &str| MockStreamEvent::tool_call(id, tool, serde_json::json!({}));
    let model = MockCompletionModel::from_stream_turns([vec![
        call("t1", "touch"),
        MockStreamEvent::final_response_with_default_usage(),
    ]]);
    let mut app = tools_app(&store, &model, |app| {
        app.add_systems(AgentPass, (hold_touches, keep_unsent).in_set(Stage::Gate));
    });
    let Some(agent) = first_agent(&mut app) else {
        return store;
    };
    let world = app.world_mut();
    world.trigger(Deliver::new(agent, "go", DeliveryMode::Steer));
    let held = |world: &World| !world.resource::<Gated>().0.is_empty();
    assert!(run_until(&mut app, held));
    let steer = Incoming::new("also the docs", DeliveryMode::Steer);
    let message = app.world_mut().spawn((steer, ForAgent(agent))).id();
    assert!(run_until(&mut app, |world| {
        world.get::<Started>(message).is_some()
    }));
    if stopped {
        app.world_mut().trigger(StopTurn::new(agent));
        app.update();
    }
    app.world_mut().write_message(AppExit::Success);
    app.update();
    store
}

/// The kinds of the asides of the first agent's conversation.
fn aside_kinds(app: &mut App) -> Vec<String> {
    let agent = first_agent(app);
    let conversation = agent.and_then(|agent| app.world().get::<Conversation>(agent));
    let asides = conversation.into_iter().flat_map(|it| it.asides());
    asides.map(|(_, aside)| aside.kind.clone()).collect()
}

#[test]
fn a_message_in_the_inbox_at_exit_goes_with_the_restored_turns_next_call() {
    let store = pending_at_exit(false);
    let model = MockCompletionModel::from_stream_turns([vec![
        MockStreamEvent::text("done"),
        MockStreamEvent::final_response_with_default_usage(),
    ]]);
    let mut app = tools_app(&store, &model, |_| {});
    let ended = |world: &World| !world.resource::<Ended>().0.is_empty();
    assert!(run_until(&mut app, ended));
    let sent = serde_json::to_string(&model.requests()).unwrap_or_default();
    assert!(sent.contains("also the docs"), "{sent}");
    // Given once, its mark gone with it.
    assert_eq!(said(&mut app).matches("also the docs").count(), 1);
    assert!(!aside_kinds(&mut app).iter().any(|kind| kind == "rig.inbox"));
}

/// The text each [`Recalled`] handed back.
#[derive(Resource, Default)]
struct RecalledTexts(Vec<String>);

#[test]
fn a_message_kept_for_an_idle_agent_comes_back_to_the_user_after_a_restart() {
    let store = pending_at_exit(true);
    let model = MockCompletionModel::from_stream_turns(Vec::<Vec<MockStreamEvent>>::new());
    let mut app = tools_app(&store, &model, |app| {
        app.init_resource::<RecalledTexts>().add_systems(
            Last,
            |mut read: MessageReader<Recalled>, mut texts: ResMut<RecalledTexts>| {
                texts.0.extend(read.read().map(|it| it.text.clone()));
            },
        );
    });
    let recalled = |world: &World| !world.resource::<RecalledTexts>().0.is_empty();
    assert!(run_until(&mut app, recalled));
    assert_eq!(app.world().resource::<RecalledTexts>().0, ["also the docs"]);
    assert!(!said(&mut app).contains("also the docs"));
    assert_eq!(model.request_count(), 0);
}

/// A plugin's label on a message.
#[derive(
    Component, Reflect, Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize,
)]
#[reflect(opaque, Component, Saved, Default, Clone, Serialize, Deserialize)]
struct Labelled {
    id: u32,
    label: String,
}

/// A session whose agent `main` has said one thing and holds `saved` under
/// the key `key`.
fn session_holding(key: &str, saved: Value) -> MemoryStore {
    let store = MemoryStore::default();
    let opened = rig_memory::Writer::open(std::sync::Arc::new(store.clone()));
    let Ok((writer, _)) = opened else {
        return store;
    };
    writer.start();
    let mut conversation = rig_core::conversation::Conversation::new();
    conversation.push(Message::user("hello"));
    writer.save("main", &mut conversation);
    writer.save_value("main", key, Some(saved), true);
    assert!(writer.flush().is_ok());
    store
}

/// A saved value that no longer reads as its type is skipped with a
/// notice, and its next save replaces it.
#[test]
fn a_saved_value_that_no_longer_reads_is_skipped_and_replaced_at_the_next_save() {
    let path = Labelled::type_path();
    let store = session_holding(path, serde_json::json!("seven"));
    let mut app = app(&store);
    let agent = first_agent(&mut app);
    assert!(agent.is_some_and(|agent| app.world().get::<Labelled>(agent).is_none()));
    let notices = app.world().resource::<Messages<Notice>>();
    let mut cursor = notices.get_cursor();
    let texts: Vec<_> = cursor.read(notices).map(|notice| &notice.text).collect();
    let skipped = texts.iter().filter(|text| {
        text.starts_with(&format!("Skipped saved component `{path}`"))
            && text.ends_with("its next save replaces it.")
    });
    assert_eq!(skipped.count(), 1, "{texts:?}");

    let changed = Labelled {
        id: 1,
        label: "new".to_owned(),
    };
    if let Some(agent) = agent {
        app.world_mut().entity_mut(agent).insert(changed.clone());
    }
    app.update();
    let session = rig_memory::Session::read(&store).unwrap_or_default();
    let now = session
        .agent("main")
        .and_then(|main| main.saved.get(path).cloned());
    assert_eq!(now, serde_json::to_value(&changed).ok());
}

/// The goal of a plugin, saved under its stable key.
#[derive(Resource, Reflect, Clone, Debug, Default, PartialEq)]
#[reflect(Resource, Saved, Default)]
struct Goal(String);

/// The same goal after its type was renamed.
#[derive(Resource, Reflect, Clone, Debug, Default, PartialEq)]
#[reflect(Resource, Saved, Default)]
struct RenamedGoal(String);

/// A value saved under a stable key is logged under it, read by a library
/// program by it, and restored into a type renamed since.
#[test]
fn a_value_saved_under_its_key_survives_a_rename_of_its_type() {
    let store = MemoryStore::default();
    let keyed = |app: &mut App, path: &str| {
        let mut keys = app.world_mut().resource_mut::<crate::restore::SavedKeys>();
        assert!(keys.add(path, "goals.goal").is_ok());
    };
    let mut first = App::new();
    first
        .set_error_handler(bevy_ecs::error::warn)
        .add_plugins((AgentPlugin, SessionLogPlugin))
        .insert_resource(SessionStore::new(store.clone()));
    keyed(&mut first, Goal::type_path());
    first.finish();
    first.update();
    first
        .world_mut()
        .insert_resource(Goal("ship it".to_owned()));
    if let Some(agent) = first_agent(&mut first) {
        say(&mut first, agent, Message::user("hello"));
    }
    first.update();
    drop(first);
    let session = rig_memory::Session::read(&store).unwrap_or_default();
    let goal = session.value::<String>("goals.goal");
    assert_eq!(goal.and_then(Result::ok).as_deref(), Some("ship it"));

    let mut second = App::new();
    second
        .set_error_handler(bevy_ecs::error::warn)
        .add_plugins((AgentPlugin, SessionLogPlugin))
        .insert_resource(SessionStore::new(store.clone()));
    keyed(&mut second, RenamedGoal::type_path());
    second.finish();
    second.update();
    let goal = second.world().get_resource::<RenamedGoal>().cloned();
    assert_eq!(goal, Some(RenamedGoal("ship it".to_owned())));
}

/// What the model reads of a call that may change something and was
/// running when the session stopped.
fn interrupted() -> String {
    format!("Stopped: {RESTART}. It may have partly run; check before running it again.")
}
