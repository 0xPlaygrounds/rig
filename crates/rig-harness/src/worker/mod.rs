//! The worker: today's agent hosting exactly one session for `rigd`. It
//! applies the commands `rigd` forwards exactly once and exports the
//! session's events.
//!
//! - **Commands** ([`WorkerCommand`]) are applied in `PreUpdate`, in
//!   `commandSeq` order, each once: a command at or below the main agent's
//!   [`Applied`] is skipped. A message's `Applied` is set when the kernel
//!   accepts it, so the value and the message are saved in the same append
//!   of the main agent's log, and a crash between the two loses neither.
//! - **Durable events** come from what the session's logs stored: in `Last`,
//!   after the frame's log write, each stored record becomes its events
//!   (`turn.*`, `message`, `conversation.change`), keyed by its place in its
//!   log ([`SourceKey`]). On start, `rigd`'s `knownThrough` says what it
//!   journaled, and every record after it is exported again.
//! - **Live events** (`message.delta`, and a plugin's through [`Publish`])
//!   go out in the same batch, one `worker/events` per frame.
//!
//! The plugin does nothing unless the worker was started by `rigd`: its
//! [`WorkerConfig`] comes from the launch environment
//! (`RIG_WORKER_TOKEN`, `RIG_SESSION`, `RIG_HOME`) or a test's resource.

mod asks;
pub mod context;
pub(crate) mod export;
mod link;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use bevy_app::{App, Plugin, PostStartup, PreUpdate};
use bevy_ecs::error::{BevyError, ErrorContext};
use bevy_ecs::prelude::*;
use bevy_ecs::system::SystemParam;
use bevy_reflect::Reflect;
use crossbeam_channel::Receiver;
use rig_ecs::WriteLogs;
use rig_ecs::agent::{Agent, AgentId, Partial, StopTurn};
use rig_ecs::inbox::{DeliveredIn, DeliveryMode, Incoming};
use rig_ecs::session_log::{ReflectSaved, SessionLog, SessionStore};
use rig_ecs::stage::{ForAgent, Outcome, Started};
use rig_ecs::tasks::Wake;
use rig_protocol::event::MessageDelta;
use rig_protocol::methods::{
    self, ClientInfo, Command, Hello, LogLen, SendMode, StopScope, VersionRange, WorkerApplied,
    WorkerCommand, WorkerEvent, WorkerEvents, WorkerHello, WorkerQuiescent, WorkerReply,
};
use rig_protocol::{AgentId as WireAgent, BuildId, CommandSeq, EventRecord, Endpoint, Method};
use rig_protocol::{Grant, PROTOCOL, RpcError, SessionId};

use crate::ask::Answerer;
use crate::channel::{Caller, ChannelEvent};
use crate::server::Principal;
use crate::host::exit::{Exit, ExitReason};
use crate::main_agent::{MainAgentQuery, main_agent};
use export::Exporter;
use link::{Inbound, Link};

pub use rig_protocol::SourceKey;

/// The launch environment's flag, `1` when nobody is attached to answer
/// during the run, such as `rig -p`'s session.
pub const UNATTENDED: &str = "RIG_UNATTENDED";

/// Whether nobody attends this session while it runs, such as `rig -p`'s:
/// the model is told nobody replies, and a plugin skips a question it
/// could do without. Required approvals still ask (they wait for an
/// answer, then their deadline denies them). `false` unless `rigd` said so.
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Unattended(pub bool);

/// While any entity has it, the worker never reports itself quiescent, so
/// `rigd` keeps it loaded: a plugin that sends the agent work over time,
/// such as one watching files, spawns one, and despawns it to let the
/// worker unload.
#[derive(Component, Clone, Copy, Debug, Default)]
pub struct KeepLoaded;

/// Whether the session's tree spent its budget: while it has, no model
/// call starts (each is refused in Gate), so the overshoot is at most the
/// calls already in flight. `rigd` sets it through [`SetBudget`].
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BudgetGate {
    pub exhausted: bool,
}

/// `x/rig/budget`: `rigd` tells the worker its tree's budget is spent, or
/// no longer is.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SetBudget {
    pub exhausted: bool,
}

impl Method for SetBudget {
    const NAME: &'static str = "x/rig/budget";
    type Params = SetBudget;
    type Result = rig_protocol::methods::Empty;
}

fn set_budget(
    call: In<crate::channel::Call<SetBudget>>,
    mut gate: ResMut<BudgetGate>,
) -> Result<rig_protocol::methods::Empty, RpcError> {
    gate.exhausted = call.params.exhausted;
    Ok(rig_protocol::methods::Empty {})
}

/// Refuses every model call about to start while the budget is spent.
fn gate_budget(
    gate: Res<BudgetGate>,
    mut calls: Query<
        &mut rig_ecs::stage::Decision,
        (rig_ecs::stage::ReadyToStart, rig_ecs::stage::ModelCalls),
    >,
) {
    if !gate.exhausted {
        return;
    }
    for mut decision in &mut calls {
        if !decision.is_decided() {
            decision.decide(
                rig_core::effect::Decider::plugin("rig-worker"),
                rig_ecs::stage::Outcome::refused(
                    "The budget of this session's agent tree is spent, so no model call starts.",
                ),
            );
        }
    }
}

/// The worker's plugin, added by [`worker_app`](crate::worker_app).
pub struct WorkerPlugin;

impl Plugin for WorkerPlugin {
    fn build(&self, app: &mut App) {
        let unattended = std::env::var(UNATTENDED).is_ok_and(|flag| flag == "1");
        if !app.world().contains_resource::<Unattended>() {
            app.insert_resource(Unattended(unattended));
        }
        use crate::channel::ChannelAppExt;
        app.init_resource::<BudgetGate>()
            .add_method::<SetBudget, _, _>(set_budget)
            .add_systems(
                rig_ecs::stage::AgentPass,
                gate_budget.in_set(rig_ecs::stage::Stage::Gate),
            );
        app.register_type::<Applied>()
            .register_type::<Through>()
            .register_type::<context::ContextSent>()
            .add_message::<ChannelEventRead>();
        let config = app
            .world_mut()
            .remove_resource::<WorkerConfig>()
            .or_else(WorkerConfig::from_env);
        let Some(config) = config else {
            return;
        };
        if let Some(grant) = &config.grant {
            let home = crate::services::RigHome::of(app.world());
            let applied = match config.session.0.parse() {
                Ok(id) => confine(app, grant, home.session(&id).path()),
                Err(why) => Err(why.to_string()),
            };
            if let Err(why) = applied {
                // Never unconfined: a grant this worker can't enforce ends it.
                eprintln!("rig: the session's sandbox cannot be applied: {why}");
                std::process::exit(i32::from(LOST));
            }
        }
        app.world_mut().spawn((Name::new("channel answerer"), Answerer::default()));
        app.insert_resource(config)
            .init_resource::<WorkerOutbox>()
            .init_resource::<asks::AskIds>()
            .add_systems(PostStartup, connect)
            .add_systems(
                bevy_app::PostUpdate,
                (asks::open_asks, asks::open_pickers).run_if(resource_exists::<WorkerChannel>),
            )
            .add_systems(
                bevy_app::Last,
                asks::close_asks
                    .before(export)
                    .run_if(resource_exists::<WorkerChannel>),
            )
            .add_observer(asks::expire)
            .add_observer(context::export_context)
            .add_systems(
                PreUpdate,
                apply_commands.run_if(resource_exists::<WorkerChannel>),
            )
            .add_systems(
                bevy_app::Last,
                export.after(WriteLogs).run_if(resource_exists::<WorkerChannel>),
            )
            .add_observer(applied_on_accept)
            .add_observer(applied_on_answer)
            .add_observer(through_on_delivery);
    }
}

/// Confines this worker to `grant` before any agent plugin runs: the
/// process (`sandbox::apply`), and, when the grant restricts anything, every
/// command it starts (through its confine step and network prefix, without
/// the variables that reach out of the sandbox, with its own `TMPDIR`) and
/// the tools its
/// agents may call.
fn confine(app: &mut App, grant: &Grant, session_dir: &std::path::Path) -> Result<(), String> {
    use crate::sandbox;
    sandbox::apply(grant, session_dir).map_err(|why| why.to_string())?;
    if let Some(tools) = grant.tools.clone() {
        app.insert_resource(GrantedTools(tools))
            .add_observer(grant_tools);
    }
    if !sandbox::restricts(grant) {
        return Ok(());
    }
    let processes = app
        .world_mut()
        .get_resource_or_init::<crate::services::Processes>()
        .clone();
    let mut launcher = processes.launcher();
    let exe = std::env::current_exe().map_err(|why| why.to_string())?;
    let (mut prefix, env) =
        sandbox::shell_launch(grant, session_dir, &exe).map_err(|why| why.to_string())?;
    prefix.append(&mut launcher.prefix);
    launcher = launcher.wrapped(prefix).without_env(sandbox::scrubbed_env());
    for (name, value) in env {
        launcher = launcher.with_env(name, value);
    }
    processes.set(launcher);
    Ok(())
}

/// The tools a restricted session's agents may call, from its grant.
#[derive(Resource, Clone, Debug)]
struct GrantedTools(Vec<String>);

/// Each agent gets the grant's tools, or fewer when its own access is
/// narrower already, whatever its saved access says.
fn grant_tools(
    added: On<Insert<rig_ecs::agent::ToolAccess>>,
    granted: Res<GrantedTools>,
    access: Query<&rig_ecs::agent::ToolAccess>,
    mut commands: Commands,
) {
    use rig_ecs::agent::ToolAccess;
    let narrowed = match access.get(added.entity) {
        Ok(ToolAccess::Only(names)) => names
            .iter()
            .filter(|name| granted.0.contains(name))
            .cloned()
            .collect(),
        _ => granted.0.clone(),
    };
    let same = |now: &ToolAccess| matches!(now, ToolAccess::Only(names) if *names == narrowed);
    if access.get(added.entity).is_ok_and(same) {
        return;
    }
    commands.entity(added.entity).insert(ToolAccess::Only(narrowed));
}

/// Where `rigd` is and who this worker is to it.
#[derive(Resource, Clone, Debug)]
pub struct WorkerConfig {
    pub endpoint: Endpoint,
    pub token: String,
    pub session: SessionId,
    pub build: BuildId,
    /// The session's grant, whose up-front rules answer approvals.
    pub grant: Option<Grant>,
    /// The folder the rules' paths are read from.
    pub cwd: PathBuf,
    /// How long an ask nobody answers stays open.
    pub ask_timeout: Duration,
}

/// The launch environment's grant of a restricted session, as JSON.
pub const GRANT: &str = "RIG_GRANT";

impl WorkerConfig {
    /// From the launch environment, when `rigd` started this process.
    pub fn from_env() -> Option<WorkerConfig> {
        use crate::harness_protocol::{Home, env};
        let token = std::env::var(env::WORKER_TOKEN).ok().filter(|t| !t.is_empty())?;
        let session = std::env::var(env::SESSION).ok().filter(|s| !s.is_empty())?;
        // `bin/builds/<BuildId>/rig-agent`.
        let build = std::env::current_exe()
            .ok()
            .and_then(|exe| Some(exe.parent()?.file_name()?.to_string_lossy().into_owned()))
            .unwrap_or_default();
        Some(WorkerConfig {
            endpoint: Endpoint::for_home(Home::from_env().root()),
            token,
            session: SessionId(session),
            build: BuildId(build),
            grant: std::env::var(GRANT)
                .ok()
                .and_then(|grant| serde_json::from_str(&grant).ok()),
            cwd: std::env::current_dir().unwrap_or_default(),
            ask_timeout: asks::ASK_TIMEOUT,
        })
    }
}

/// The worker's connection to `rigd`, as its session's principal. Clones
/// share it; [`call`](Self::call) may block, so a tool calls it from its
/// own thread ([`blocking`](crate::prelude::blocking)).
#[derive(Resource, Clone)]
pub struct WorkerChannel {
    link: Arc<Link>,
    inbound: Receiver<Inbound>,
    session: SessionId,
}

impl WorkerChannel {
    /// Calls `M` on `rigd` and waits for its answer, such as
    /// `rig/session/start` for a child.
    pub fn call<M: Method>(&self, params: &M::Params) -> Result<M::Result, RpcError> {
        self.link.call::<M>(params)
    }

    /// Asks `rigd` to restart this session on the installed build `build`
    /// (`rig/session/reload`), which it does once the worker is quiescent.
    pub fn reload(&self, build: &str) -> Result<CommandSeq, RpcError> {
        let reload = methods::Reload {
            session_id: self.session.clone(),
            build: BuildId(build.to_owned()),
            key: rig_protocol::Key(format!("reload-{}-{build}", self.session.0)),
        };
        self.call::<methods::Reload>(&reload)
            .map(|reloading| reloading.command_seq)
    }

    /// This worker's session.
    pub fn session(&self) -> &SessionId {
        &self.session
    }
}

/// An event of a session this worker subscribed to (`rig/events/subscribe`),
/// such as a child's, read in the frame after it came.
#[derive(Message, Clone, Debug)]
pub struct ChannelEventRead(pub EventRecord);

/// On the session's main agent: the last command applied, saved in the same
/// agent-log append as what it produced.
#[derive(Component, Reflect, Default, Clone, Copy, Debug, PartialEq, Eq)]
#[reflect(Component, Saved)]
pub struct Applied(pub u64);

/// On an agent: the last command its running turn answers, saved with the
/// conversation, for `turn.ended`'s `through`.
#[derive(Component, Reflect, Default, Clone, Copy, Debug, PartialEq, Eq)]
#[reflect(Component, Saved)]
pub struct Through(pub u64);

/// On a message a command sent: that command's `commandSeq`.
#[derive(Component, Clone, Copy, Debug)]
pub struct CommandOf(pub u64);

/// Events published this frame, before `rigd` journals them.
#[derive(Resource, Default)]
pub struct WorkerOutbox {
    pub events: Vec<WorkerEvent>,
}

/// Publishes an event to this session's journal.
#[derive(SystemParam)]
pub struct Publish<'w> {
    outbox: ResMut<'w, WorkerOutbox>,
}

impl Publish<'_> {
    pub fn event<E: ChannelEvent>(&mut self, e: &E) {
        let data = serde_json::to_value(e).unwrap_or_default();
        self.raw(E::KIND, data, None);
    }

    /// An event of any kind; `durable` is its saved record's key.
    pub fn raw(&mut self, kind: &str, data: serde_json::Value, durable: Option<SourceKey>) {
        self.outbox.events.push(WorkerEvent {
            kind: kind.to_owned(),
            data,
            source: durable,
        });
    }
}

/// The worker's error handler: a failing system leaves partial mutations,
/// so the worker logs it and exits with 70 for `rigd` to restart it.
pub fn fail(error: BevyError, ctx: ErrorContext) {
    eprintln!("rig: worker failed in {}: {error}", ctx.name());
    std::process::exit(70);
}

/// The exit code of a worker that lost `rigd` or could not reach it.
const LOST: u8 = 70;

/// What the worker keeps between frames.
#[derive(Resource, Default)]
struct WorkerState {
    exporter: Exporter,
    /// Commands not applied yet, in order.
    queued: Vec<WorkerCommand>,
    /// The `Applied` last reported to `rigd`.
    reported: u64,
    quiescent: Option<bool>,
    /// What each streaming model call sent of its text and reasoning.
    streamed: HashMap<Entity, (usize, usize)>,
    /// Whether the session log failed, and the worker exits.
    failed: bool,
}

/// Connects to `rigd` once the session is restored: says hello, tells it
/// what the session holds, takes the commands to replay and exports again
/// what it has not journaled.
fn connect(world: &mut World) {
    let Some(config) = world.get_resource::<WorkerConfig>().cloned() else {
        return;
    };
    if let Err(why) = hello(world, &config) {
        bevy_log::error!("the worker cannot reach rigd: {why}");
        world.trigger(Exit(ExitReason::Failed(LOST)));
    }
}

fn hello(world: &mut World, config: &WorkerConfig) -> Result<(), String> {
    let conn = rig_protocol::Conn::connect(&config.endpoint).map_err(|e| e.to_string())?;
    let (reader, writer) = conn.split().map_err(|e| e.to_string())?;
    let wake = world.get_resource::<Wake>().cloned().unwrap_or_default();
    let (link, inbound) = Link::start(reader, writer, wake);
    link.call::<Hello>(&Hello {
        token: config.token.clone(),
        client: ClientInfo {
            name: "worker".into(),
            version: crate::harness_protocol::VERSION.into(),
        },
        rig: VersionRange {
            min: PROTOCOL,
            max: PROTOCOL,
        },
    })
    .map_err(|e| e.to_string())?;
    let store = world.get_resource::<SessionStore>().map(|s| s.0.clone());
    let mut histories = Vec::new();
    if let Some(store) = &store {
        for log in store.logs().map_err(|e| e.to_string())? {
            let records = rig_memory::history(&**store, &log).map_err(|e| e.to_string())?;
            histories.push((log, records));
        }
    }
    let applied = main_applied(world);
    let logs = histories
        .iter()
        .map(|(log, records)| LogLen {
            log: log.clone(),
            len: records.last().map_or(0, |(stamp, _)| stamp.seq + 1),
        })
        .collect();
    let welcome = link
        .call::<WorkerHello>(&WorkerHello {
            rig: PROTOCOL,
            session_id: config.session.clone(),
            build: config.build.clone(),
            applied: CommandSeq(applied),
            logs,
        })
        .map_err(|e| e.to_string())?;
    let mut state = WorkerState {
        reported: applied,
        ..WorkerState::default()
    };
    let mut events = Vec::new();
    for (log, records) in &histories {
        let known = welcome.known_through.get(log).copied();
        for (stamp, record) in records {
            let emit = known.is_none_or(|known| stamp.seq > known);
            state
                .exporter
                .record(log, stamp.seq, record, emit, &mut events);
        }
    }
    state.queued = welcome.replay;
    if !events.is_empty() {
        link.send::<WorkerEvents>(&WorkerEvents { events });
    }
    if let Some(log) = world.get_resource::<SessionLog>() {
        log.tap();
    }
    world.insert_resource(state);
    world.insert_resource(WorkerChannel {
        link,
        inbound,
        session: config.session.clone(),
    });
    Ok(())
}

/// The main agent's [`Applied`], 0 without one.
fn main_applied(world: &mut World) -> u64 {
    let mut agents = world.query_filtered::<(Entity, &AgentId, Has<rig_ecs::agent::SpawnedBy>), With<Agent>>();
    let main = agents
        .iter(world)
        .min_by(|a, b| {
            rig_memory::main_order(a.2, &a.1.0).cmp(&rig_memory::main_order(b.2, &b.1.0))
        })
        .map(|(entity, ..)| entity);
    main.and_then(|agent| world.get::<Applied>(agent))
        .map_or(0, |applied| applied.0)
}

/// Applies the commands that came, each once and in order.
fn apply_commands(
    channel: Res<WorkerChannel>,
    mut state: ResMut<WorkerState>,
    agents: MainAgentQuery,
    applied: Query<&Applied>,
    mut events: MessageWriter<ChannelEventRead>,
    mut commands: Commands,
) {
    for inbound in channel.inbound.try_iter() {
        match inbound {
            Inbound::Command(command) => state.queued.push(command),
            Inbound::Event(record) => {
                events.write(ChannelEventRead(record));
            }
            Inbound::Closed => {
                bevy_log::error!("rigd closed the worker's channel");
                commands.trigger(Exit(ExitReason::Failed(LOST)));
            }
        }
    }
    if state.queued.is_empty() {
        return;
    }
    let Some(agent) = main_agent(&agents) else {
        return;
    };
    let mut through = applied.get(agent).map_or(0, |a| a.0);
    let queued = std::mem::take(&mut state.queued);
    for WorkerCommand {
        command_seq: CommandSeq(seq),
        command,
    } in queued
    {
        // A plugin call is no command of the session's: it has no seq.
        if let Command::Plugin {
            method,
            params,
            reply,
            owner,
        } = command
        {
            plugin_call(&channel, (method, params, reply, owner), &mut commands);
            continue;
        }
        if seq <= through {
            continue;
        }
        through = seq;
        match command {
            Command::Send(send) => {
                let text = send_text(&send);
                let mode = match send.mode {
                    Some(SendMode::Queue) => DeliveryMode::Queue,
                    Some(SendMode::Steer | SendMode::Interrupt) | None => DeliveryMode::Steer,
                };
                if send.mode == Some(SendMode::Interrupt) {
                    commands.trigger(StopTurn {
                        entity: agent,
                        reason: None,
                    });
                }
                // Applied once the kernel takes it ([`applied_on_accept`]).
                commands.spawn((Incoming::new(text, mode), ForAgent(agent), CommandOf(seq)));
                continue;
            }
            Command::Stop(stop) => {
                // What the running turn ends with answers this command too.
                commands.queue(move |world: &mut World| answering(world, agent, seq));
                commands.trigger(StopTurn {
                    entity: agent,
                    reason: None,
                });
                if stop.scope == StopScope::Session {
                    commands.trigger(Exit(ExitReason::Quit));
                }
            }
            Command::Answer(answer) => {
                // The turn the ask held goes on with this answer.
                commands.queue(move |world: &mut World| answering(world, agent, seq));
                commands.queue(move |world: &mut World| asks::answer(world, &answer));
            }
            Command::Plugin { .. } => {}
            Command::Reload(_) => {
                bevy_log::warn!("the worker does not apply this command yet");
            }
        }
        commands.entity(agent).insert(Applied(seq));
    }
}

/// Runs the plugin method `method` for `rigd` and answers it.
/// The method is called as the owner when `rigd` says the owner called it
/// (`owner`), else as this session.
fn plugin_call(
    channel: &WorkerChannel,
    (method, params, reply, owner): (String, serde_json::Value, u64, bool),
    commands: &mut Commands,
) {
    let session = channel.session.clone();
    let link = channel.link.clone();
    commands.queue(move |world: &mut World| {
        let principal = match owner {
            true => Principal::Owner,
            false => Principal::Session(session),
        };
        let caller = Caller {
            client: Entity::PLACEHOLDER,
            principal,
        };
        let outcome = match crate::channel::call(world, &method, caller, params) {
            Some(Ok(result)) => rig_protocol::Outcome::Result(result),
            Some(Err(e)) => rig_protocol::Outcome::Error(e),
            None => rig_protocol::Outcome::Error(RpcError::new(
                rig_protocol::codes::METHOD_NOT_FOUND,
                format!("this session's build has no method {method}"),
            )),
        };
        link.send::<WorkerReply>(&WorkerReply { reply, outcome });
    });
}

/// A message's text: its `text`, else its content's text blocks.
fn send_text(send: &methods::Send) -> String {
    if let Some(text) = &send.text {
        return text.clone();
    }
    let blocks = send.content.iter().flatten();
    let texts: Vec<&str> = blocks
        .filter(|block| block.get("type").and_then(|t| t.as_str()) == Some("text"))
        .filter_map(|block| block.get("text")?.as_str())
        .collect();
    texts.join("\n\n")
}

/// A command's message the kernel accepted is applied, in the frame its
/// conversation saves it.
fn applied_on_accept(
    accepted: On<Add<Started>>,
    sent: Query<&CommandOf>,
    agents: MainAgentQuery,
    applied: Query<&Applied>,
    commands: Commands,
) {
    if let Ok(&CommandOf(seq)) = sent.get(accepted.entity) {
        mark_applied(seq, &agents, &applied, commands);
    }
}

/// A command's message a plugin answered instead, such as a slash command,
/// is applied.
fn applied_on_answer(
    answered: On<Insert<Outcome>>,
    sent: Query<&CommandOf>,
    agents: MainAgentQuery,
    applied: Query<&Applied>,
    commands: Commands,
) {
    if let Ok(&CommandOf(seq)) = sent.get(answered.entity) {
        mark_applied(seq, &agents, &applied, commands);
    }
}

fn mark_applied(
    seq: u64,
    agents: &MainAgentQuery,
    applied: &Query<&Applied>,
    mut commands: Commands,
) {
    let Some(agent) = main_agent(agents) else {
        return;
    };
    if applied.get(agent).is_ok_and(|a| a.0 >= seq) {
        return;
    }
    commands.entity(agent).insert(Applied(seq));
}

/// A command's message delivered into a turn is what the turn answers.
///
/// Its record goes in the log at once, before the turn's own records: a
/// turn may start and end within the frame, and its end is read against
/// the last `Through` before it.
fn through_on_delivery(
    delivered: On<Insert<DeliveredIn>>,
    sent: Query<(&CommandOf, &ForAgent)>,
    mut commands: Commands,
) {
    if let Ok((&CommandOf(seq), &ForAgent(agent))) = sent.get(delivered.entity) {
        commands.queue(move |world: &mut World| answering(world, agent, seq));
    }
}

/// `agent`'s running or next turn answers the command `seq` too: its
/// [`Through`] rises to it, its record queued at once, ahead of the turn's
/// own records, as the component's save in `Last` would write it, so that
/// one finds it unchanged.
fn answering(world: &mut World, agent: Entity, seq: u64) {
    if world.get::<Through>(agent).is_some_and(|t| t.0 >= seq) {
        return;
    }
    let Some(id) = world.get::<AgentId>(agent).cloned() else {
        return;
    };
    if let (Some(log), Some(registry)) = (
        world.get_resource::<SessionLog>(),
        world.get_resource::<bevy_ecs::reflect::AppTypeRegistry>(),
    ) {
        let registry = registry.read();
        let saved = Through(seq);
        let value = bevy_reflect::serde::TypedReflectSerializer::new(&saved, &registry);
        if let Ok(value) = serde_json::to_value(value) {
            log.save_value_now(&id.0, <Through as bevy_reflect::TypePath>::type_path(), value);
        }
    }
    if let Ok(mut agent) = world.get_entity_mut(agent) {
        agent.insert(Through(seq));
    }
}

/// Sends the frame's events: the records its log write stored, the
/// streamed text, and what plugins published; then what was applied and
/// whether the worker is quiescent.
#[allow(clippy::too_many_arguments)]
fn export(
    channel: Res<WorkerChannel>,
    mut state: ResMut<WorkerState>,
    log: Option<Res<SessionLog>>,
    mut outbox: ResMut<WorkerOutbox>,
    partials: Query<(Entity, &Partial, &ForAgent)>,
    ids: Query<&AgentId>,
    agents: MainAgentQuery,
    applied: Query<&Applied>,
    busy: Query<(), Or<(With<rig_ecs::agent::TurnOf>, With<KeepLoaded>)>>,
    progress: Query<
        (&crate::progress::ToolProgress, &rig_ecs::tools::ToolRequest),
        Changed<crate::progress::ToolProgress>,
    >,
    mut commands: Commands,
) {
    let state = &mut *state;
    // What is not saved is never exported, so a worker that can no longer
    // save would answer nobody: it exits, and rigd restarts it from what
    // it saved (or faults the session after repeated crashes).
    if let Some(failure) = log.as_ref().and_then(|log| log.failure())
        && !state.failed
    {
        state.failed = true;
        bevy_log::error!("the worker stops: its session is no longer saved: {failure}");
        commands.trigger(Exit(ExitReason::Failed(LOST)));
    }
    let mut events = Vec::new();
    for (name, stamp, record) in log.iter().flat_map(|log| log.take_written()) {
        state
            .exporter
            .record(&name, stamp.seq, &record, true, &mut events);
    }
    for (call, partial, &ForAgent(agent)) in &partials {
        let sent = state.streamed.entry(call).or_default();
        let text = partial.text.get(sent.0..).filter(|t| !t.is_empty());
        let reasoning = partial.reasoning.get(sent.1..).filter(|t| !t.is_empty());
        if text.is_none() && reasoning.is_none() {
            continue;
        }
        let delta = MessageDelta {
            agent_id: WireAgent(ids.get(agent).map(|id| id.0.clone()).unwrap_or_default()),
            message_id: format!("stream-{}", call.index()),
            text: text.map(str::to_owned),
            reasoning: reasoning.map(str::to_owned),
        };
        *sent = (partial.text.len(), partial.reasoning.len());
        let event = rig_protocol::Event::from(delta);
        events.push(WorkerEvent {
            kind: event.kind().to_owned(),
            data: event.encode(),
            source: None,
        });
    }
    state.streamed.retain(|call, _| partials.contains(*call));
    for (crate::progress::ToolProgress(line), request) in &progress {
        if line.trim().is_empty() {
            continue;
        }
        let event = rig_protocol::Event::from(rig_protocol::event::CallProgress {
            call_id: serde_json::to_string(&request.call.id).unwrap_or_default(),
            text: line.clone(),
        });
        events.push(WorkerEvent {
            kind: event.kind().to_owned(),
            data: event.encode(),
            source: None,
        });
    }
    events.append(&mut outbox.events);
    if !events.is_empty() {
        channel.link.send::<WorkerEvents>(&WorkerEvents { events });
    }
    let now = main_agent(&agents)
        .and_then(|agent| applied.get(agent).ok())
        .map_or(0, |a| a.0);
    if now > state.reported {
        state.reported = now;
        channel.link.send::<WorkerApplied>(&WorkerApplied {
            through: CommandSeq(now),
        });
    }
    let quiescent = busy.is_empty() && state.queued.is_empty();
    if state.quiescent != Some(quiescent) {
        state.quiescent = Some(quiescent);
        channel
            .link
            .send::<WorkerQuiescent>(&WorkerQuiescent { quiescent });
    }
}

#[cfg(test)]
mod tests;
