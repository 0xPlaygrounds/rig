//! What the rig agent's plugin tests build: an app on a session store or
//! in a session directory, scripted models as the catalog's DeepSeek model
//! or under references of their own, their replies, collectors of what the
//! app wrote, a front that answers questions from a script
//! ([`AnswerAsks`]) or one at a time ([`answer_open`]), the terminal view
//! without a terminal ([`tui_view`], driven by [`press`]), and the clocks
//! moved on.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fs;
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, channel};
use std::time::{Duration, Instant};

use bevy_time::TimeUpdateStrategy;
use rig_core::catalog::{Catalog, Connector, ModelSpec, Transport};
use rig_core::effect::EffectId;
use rig_core::http_client::{
    self, Bytes, DynHttpClient, HttpClientExt, LazyBody, Method, MultipartForm, Request, Response,
    StreamingResponse, Uri,
};
use rig_core::message::{Message as ChatMessage, ToolFunction, ToolName};
use rig_core::operation::Completion;
use rig_core::providers::registry::{ConnectError, ProviderId};
use rig_core::serve::ErasedHandler;
use rig_core::serve::adapters::ModelAdapter;
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_core::wasm_compat::WasmCompatSend;
use rig_ecs::effects::Handler;
use rig_ecs::prelude::*;
use rig_ecs::session_log::SessionStore;
use rig_harness::ask::{Answer, Answerer, Ask, AskSystems};
use rig_harness::clock::WakeAt;
use rig_harness::harness_protocol::{Home, SessionId};
use rig_harness::host::session::{SessionPaths, SessionPlugin};
use rig_harness::services::RigHome;
use rig_harness::settings::Settings;
/// The store a test keeps its session in ([`app_on`]).
pub use rig_memory::MemoryStore;

/// How long a test waits for its agents.
const DEADLINE: Duration = Duration::from_secs(30);

/// An app with a task pool on the session in `store`, no user settings,
/// and its loop's wakes.
pub fn app_on(store: &MemoryStore) -> (App, Receiver<()>) {
    let (mut app, wakes) = app();
    app.insert_resource(SessionStore::new(store.clone()));
    (app, wakes)
}

/// An app with a task pool, no user settings, and its loop's wakes.
fn app() -> (App, Receiver<()>) {
    let (sender, wakes) = channel();
    let mut app = App::new();
    // The agent's pools, not Bevy's defaults: on a small machine those give
    // the IO pool a single thread, which one slow call can hold.
    app.add_plugins(TaskPoolPlugin {
        task_pool_options: rig_harness::task_pools(),
    })
    .insert_resource(Wake::new(move || {
        sender.send(()).ok();
    }))
    .insert_resource(Settings::default());
    (app, wakes)
}

/// A `RIG_HOME` of the test's own named `name`, beside its binary, emptied.
pub fn scratch_home(name: &str) -> Home {
    let exe = std::env::current_exe().unwrap_or_default();
    let root: PathBuf = exe.with_file_name(name);
    fs::remove_dir_all(&root).ok();
    Home::at(root)
}

/// An app with a task pool on the session `id` of `home`, kept in its
/// directory by the harness's session service, and its loop's wakes. The
/// service opens the session's store as the app finishes building
/// (`app.finish()`), from the app's `StoreOpener`, as it does in the
/// agent. It sends panics to the session's log; a test shows them, so its
/// failed assertions print.
pub fn app_in(home: &Home, id: &SessionId) -> (App, Receiver<()>) {
    let dir = home.session(id);
    fs::create_dir_all(dir.path()).ok();
    let (mut app, wakes) = app();
    app.insert_resource(SessionPaths(dir))
        .insert_resource(RigHome(home.clone()))
        .add_plugins(SessionPlugin);
    drop(std::panic::take_hook());
    (app, wakes)
}

/// The app's first agent.
pub fn first_agent(world: &mut World) -> Option<Entity> {
    let mut agents = world.query_filtered::<Entity, With<Agent>>();
    agents.iter(world).next()
}

/// Adds `message` to the conversation of `agent`, such as an earlier
/// exchange a test starts from.
///
/// # Errors
///
/// When `agent` has no conversation.
pub fn commit_message(
    world: &mut World,
    agent: Entity,
    message: ChatMessage,
) -> Result<(), String> {
    let mut conversation = world
        .get_mut::<Conversation>(agent)
        .ok_or_else(|| format!("{agent} has no conversation"))?;
    conversation.push(message);
    Ok(())
}

/// How each turn ended, in the order they ended, kept by [`log_ended`].
#[derive(Resource, Default)]
pub struct Ended(pub Vec<TurnOutcome>);

/// Keeps how each turn ended in [`Ended`]: `app.init_resource::<Ended>()`
/// and `.add_observer(log_ended)`.
pub fn log_ended(settled: On<Add<Settled>>, turns: Query<&TurnOutcome>, mut log: ResMut<Ended>) {
    log.0.extend(turns.get(settled.entity).cloned());
}

/// `model` as the catalog's DeepSeek model.
pub fn connect(model: &MockCompletionModel) -> Option<Connection> {
    let spec = Catalog::builtin()
        .resolve("deepseek/deepseek-flash")
        .ok()?
        .shared();
    let handler = ModelAdapter::<Completion>::new(spec.reference(), model.clone());
    Some(Connection {
        spec,
        handler: Handler(ErasedHandler::new(handler)),
    })
}

/// A catalog of exactly `served`, each the model its reference names,
/// `vendor/name`, which calls tools, answered by its scripted model. A
/// vendor rig does not know is declared. No other model is listed, so a run
/// that names none picks a scripted one whatever keys the environment holds,
/// and every HTTP request through the connector is refused ([`Offline`]),
/// so a model a test adds itself reaches no network either. A test that
/// fakes HTTP wraps the connector's client again with its own.
pub fn scripted_models<'a>(
    served: impl IntoIterator<Item = (&'a str, &'a MockCompletionModel)>,
) -> Models {
    let mut connector = Connector::new(Catalog::default());
    connector.wrap_http(|_| DynHttpClient::new(Offline));
    let mut vendors: HashMap<&str, (ProviderId, Vec<(String, MockCompletionModel)>)> =
        HashMap::new();
    for (reference, model) in served {
        let Some((vendor, name)) = reference.split_once('/') else {
            continue;
        };
        let Ok(id) = ProviderId::declare(vendor) else {
            continue;
        };
        connector
            .catalog_mut()
            .insert(ModelSpec::new(id, name).with_tools(true));
        let entry = vendors.entry(vendor).or_insert((id, Vec::new()));
        entry.1.push((name.to_owned(), model.clone()));
    }
    for (id, models) in vendors.into_values() {
        connector.provide(id, move |spec: &ModelSpec, _: &Catalog, _: &Transport| {
            let model = models.iter().find(|(name, _)| *name == spec.id);
            let Some((_, model)) = model else {
                return Err(ConnectError::Provider(format!(
                    "{} is not scripted",
                    spec.id
                )));
            };
            let model = ModelAdapter::<Completion>::new(spec.reference(), model.clone());
            Ok(ErasedHandler::new(model))
        });
    }
    Models(connector)
}

/// The HTTP client of [`scripted_models`]: it refuses every request,
/// naming it, as a client that could not be built does.
#[derive(Clone, Copy, Debug)]
pub struct Offline;

impl Offline {
    /// The refusal of a request to `uri` with `method`.
    fn refuse<T>(method: &Method, uri: &Uri) -> http_client::Result<T> {
        let why = format!("a scripted test app sends no HTTP: {method} {uri}");
        Err(http_client::Error::unbuilt(std::io::Error::other(why)))
    }
}

impl HttpClientExt for Offline {
    fn send<T, U>(
        &self,
        req: Request<T>,
    ) -> impl Future<Output = http_client::Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        T: Into<Bytes> + WasmCompatSend,
        U: From<Bytes> + WasmCompatSend + 'static,
    {
        std::future::ready(Self::refuse(req.method(), req.uri()))
    }

    fn send_multipart<U>(
        &self,
        req: Request<MultipartForm>,
    ) -> impl Future<Output = http_client::Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        U: From<Bytes> + WasmCompatSend + 'static,
    {
        std::future::ready(Self::refuse(req.method(), req.uri()))
    }

    fn send_streaming<T>(
        &self,
        req: Request<T>,
    ) -> impl Future<Output = http_client::Result<StreamingResponse>> + WasmCompatSend
    where
        T: Into<Bytes> + WasmCompatSend,
    {
        std::future::ready(Self::refuse(req.method(), req.uri()))
    }
}

/// Every message `M` the app wrote, in order, kept by [`collect`].
#[derive(Resource)]
pub struct Collected<M>(pub Vec<M>);

/// Keeps every message `M` the app writes, such as [`Notice`] or
/// [`EffectRecorded`], in [`Collected<M>`], read in `Last`.
pub fn collect<M: Message + Clone>(app: &mut App) -> &mut App {
    let keep = |mut read: MessageReader<M>, mut kept: ResMut<Collected<M>>| {
        kept.0.extend(read.read().cloned());
    };
    app.insert_resource(Collected::<M>(Vec::new()))
        .add_systems(Last, keep)
}

/// The output of each settled tool call by its call id, as the model reads
/// it, kept by [`log_tool_outputs`].
#[derive(Resource, Default)]
pub struct ToolOutputs(pub HashMap<String, String>);

/// Keeps each tool call's output in [`ToolOutputs`]:
/// `app.init_resource::<ToolOutputs>()` and
/// `.add_observer(log_tool_outputs)`.
pub fn log_tool_outputs(
    settled: On<Add<Settled>>,
    calls: Query<(&ToolRequest, &Outcome)>,
    mut outputs: ResMut<ToolOutputs>,
) {
    if let Ok((request, outcome)) = calls.get(settled.entity) {
        let output = outcome.tool_result().output().render();
        outputs.0.insert(request.call.id.to_string(), output);
    }
}

/// Answers the open question titled `title` with `answer`, as a front
/// would; `false` when no such question is open.
pub fn answer_open(world: &mut World, title: &str, answer: &Answer) -> bool {
    let mut open =
        world.query_filtered::<(&Ask, &mut Decision), (Without<Outcome>, Without<Settled>)>();
    open.iter_mut(world)
        .find(|(ask, decision)| ask.title == title && !decision.is_decided())
        .is_some_and(|(_, mut decision)| decision.decide(Decider::Person, answer.outcome()))
}

/// Runs one frame of `app` `by` later on Bevy's clock, which timers and
/// delayed commands run on; frames after it take no time.
pub fn advance(app: &mut App, by: Duration) {
    let mut time = app.world_mut().resource_mut::<Time<Virtual>>();
    let max = time.max_delta().max(by);
    time.set_max_delta(max);
    app.insert_resource(TimeUpdateStrategy::ManualDuration(by));
    app.update();
    app.insert_resource(TimeUpdateStrategy::ManualDuration(Duration::ZERO));
}

/// Moves every [`WakeAt`] `by` earlier, as if that much wall-clock time
/// passed: one that passes is `Due` from the next frame.
pub fn pass_wall_clock(world: &mut World, by: Duration) {
    let mut waits = world.query::<(Entity, &WakeAt)>();
    let earlier: Vec<(Entity, WakeAt)> = waits
        .iter(world)
        .map(|(entity, wake)| (entity, WakeAt(wake.0.checked_sub(by).unwrap_or(wake.0))))
        .collect();
    for (entity, wake) in earlier {
        world.entity_mut(entity).insert(wake);
    }
}

/// Runs frames of `app`, sleeping until one of its `wakes` between them,
/// until `done`.
///
/// # Panics
///
/// When `done` still does not hold after the deadline, naming the caller
/// and every named entity still there (agents, effects, holds, tools): a
/// test that waits in vain fails where it waited and says on what.
#[track_caller]
pub fn run_until(app: &mut App, wakes: &Receiver<()>, done: impl Fn(&World) -> bool) {
    let started = Instant::now();
    while !done(app.world()) && started.elapsed() < DEADLINE {
        app.update();
        wakes.recv_timeout(Duration::from_millis(100)).ok();
    }
    if done(app.world()) {
        return;
    }
    let world = app.world_mut();
    let named: Vec<String> = world
        .query::<&Name>()
        .iter(world)
        .map(Name::to_string)
        .collect();
    assert!(
        done(app.world()),
        "waited {DEADLINE:?} and the condition never held; named entities: {named:?}"
    );
}

/// A model reply of `text`, then the end of the stream.
pub fn reply(text: &str) -> Vec<MockStreamEvent> {
    vec![
        MockStreamEvent::text(text),
        MockStreamEvent::final_response_with_default_usage(),
    ]
}

/// A model reply that calls `tool` with `args` in call `id`.
pub fn calls(id: &str, tool: &str, args: serde_json::Value) -> Vec<MockStreamEvent> {
    vec![
        MockStreamEvent::tool_call(id, tool, args),
        MockStreamEvent::final_response_with_default_usage(),
    ]
}

/// Spawns `agent`'s call `id` of `tool` with `args`, as its model call 1
/// asked, and returns the call's entity, which gets the call's `Outcome`.
/// The next frame starts it.
pub fn start_call(
    world: &mut World,
    agent: Entity,
    (tool, id): (&str, &str),
    args: serde_json::Value,
) -> Option<Entity> {
    let name = ToolName::new(tool).ok()?;
    let call = ToolCall::from_wire(id, ToolFunction::new(name, args));
    let parent = Effect::with_parent_id(Some(EffectId::from_raw(1)));
    Some(
        world
            .spawn((ToolRequest { call }, parent, ForAgent(agent)))
            .id(),
    )
}

/// The answers [`AnswerAsks`] gives, in order, and the title of every
/// question it saw, in the order they were asked.
#[derive(Resource, Default, Debug)]
pub struct Answers {
    /// What the next questions are answered; a question asked when none
    /// is left stays open.
    pub script: VecDeque<Answer>,
    /// Each question's title.
    pub asked: Vec<String>,
}

impl Answers {
    /// Answers the next questions with `script`, in order.
    pub fn new(script: impl IntoIterator<Item = Answer>) -> Self {
        Self {
            script: script.into_iter().collect(),
            asked: Vec::new(),
        }
    }
}

/// A front someone sits at who answers every question at once from the
/// [`Answers`] script: an [`Answerer`] for every agent, answering in Gate.
pub struct AnswerAsks;

impl Plugin for AnswerAsks {
    fn build(&self, app: &mut App) {
        app.world_mut().spawn(Answerer::default());
        app.init_resource::<Answers>()
            .add_systems(AgentPass, answer_asks.in_set(Stage::Gate).after(AskSystems));
    }
}

fn answer_asks(
    mut asks: Query<
        (Entity, &Ask, &mut Decision),
        (PluginEffects, Without<Outcome>, Without<Settled>),
    >,
    mut seen: Local<HashSet<Entity>>,
    mut answers: ResMut<Answers>,
) {
    for (entity, ask, mut decision) in &mut asks {
        if !seen.insert(entity) {
            continue;
        }
        answers.asked.push(ask.title.clone());
        if decision.is_decided() {
            continue;
        }
        if let Some(answer) = answers.script.pop_front() {
            decision.decide(Decider::Person, answer.outcome());
        }
    }
}

/// The terminal view on `app`, as in a run at the terminal but with no
/// terminal: its input, key bindings, completion, pickers and questions,
/// and the transcript it lays out, driven by the keys [`press`] puts in.
/// Adds the [`TuiPlugin`](rig_tui::TuiPlugin) when the app has none.
pub fn tui_view(app: &mut App) -> &mut App {
    if !app.is_plugin_added::<rig_tui::TuiPlugin>() {
        app.add_plugins(rig_tui::TuiPlugin);
    }
    rig_tui::add_view(app);
    app
}

/// Presses `keys`, such as `esc 0 d w` or `ctrl+x ctrl+e`, in one frame of
/// the terminal view ([`tui_view`]), then runs that frame.
pub fn press(app: &mut App, keys: &str) {
    use rig_tui::crossterm::event::{Event, KeyEvent};
    use rig_tui::{KeyInput, Keys, PendingInput};

    let parsed: Result<Keys, String> = keys.parse();
    assert!(parsed.is_ok(), "{keys}: {parsed:?}");
    let mut events = app.world_mut().resource_mut::<PendingInput>();
    for press in parsed.iter().flat_map(Keys::presses) {
        let key = KeyEvent::new(press.code, press.modifiers);
        events.0.push(KeyInput::Event(Event::Key(key)));
    }
    app.update();
}
