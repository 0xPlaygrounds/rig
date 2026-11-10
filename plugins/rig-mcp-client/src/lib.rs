//! MCP servers' tools for the rig agent. The servers of `RIG_HOME/mcp.json`
//! and `<project>/.rig/mcp.json` (the common `mcpServers` format, stdio or
//! streamable HTTP), the project's entry replacing the user's of the same
//! name, and those a client of rig's names for the run ([`AddMcpServer`]),
//! connect in the background on rig's tokio runtime, each server's tools
//! registered as `<server>__<tool>` as the server lists them, with its
//! annotations, and removed when it disconnects; a lost server is connected
//! again with backoff.
//!
//! A server is an entity with an [`McpStatus`]; its tools are entities
//! [`McpToolOf`] it. A policy plugin keeps a server from running with an
//! [`McpHold`], as the opt-in [`approval::McpApprovalPlugin`] does until
//! the user approves a project's server. Each server that starts holds the
//! run's start ([`Starting`]) until its first attempt connected or failed,
//! which `connect_timeout_secs` bounds, so `--tools <server>__<tool>` can
//! name its tools in every front. `/mcp` shows the servers; `/mcp add` and
//! `/mcp remove` change the files, and `/mcp <verb>` runs a plugin's
//! `/mcp-<verb>`. The `[mcp]` table of `settings.toml` is [`McpSettings`].
//! [`McpPlugin`] reads the files; a front that hands rig its client's
//! servers adds [`McpCorePlugin`], which reads none.
//!
//! ```no_run
//! use rig_harness::prelude::*;
//!
//! App::new().add_plugins(rig_mcp_client::McpPlugin);
//! ```

pub mod approval;
mod config;

use std::collections::{BTreeMap, HashMap, HashSet};
use std::io;
use std::num::NonZeroU64;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use async_channel::{Receiver, Sender};
use bevy_ecs::system::SystemParam;
use bevy_tasks::futures_lite::future;
use bevy_tasks::{IoTaskPool, TaskPool};
use rig_core::completion::ToolDefinition;
use rig_core::effect::{Answered as Answer, EffectId, EffectKind};
use rig_core::error::ErrorReport;
use rig_core::message::ToolName;
use rig_core::serve::{self, BeforeServe, ErasedHandler, Intercept};
use rig_core::tool::{DynamicTool, Touches};
use rig_harness::host::cli::Cli;
use rig_harness::host::pipes::{Limited, TooLarge};
use rig_harness::prelude::*;
use rig_harness::rig_tools::process::{self, Process};
use rig_rmcp::{McpConnection, McpTool};
use rmcp::model::{
    ClientCapabilities, ClientConfig, Implementation, ProtocolVersion, Tool as McpDefinition,
};
use rmcp::service::{ClientInitializeError, NotificationContext, RoleClient};
use rmcp::transport::StreamableHttpClientTransport;
use rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig;
use rmcp::{ClientHandler, ServiceExt};
use serde::Deserialize;
use serde_json::{Value, json};

use config::McpFile;
pub use config::{Scope, ServerSpec};

#[cfg(test)]
mod tests;

/// How long the first lists of a server wait for a change the server said
/// before it answered, and how many there are at most.
const CHANGE_GRACE: Duration = Duration::from_millis(50);
const GRACE_LISTS: usize = 3;

/// The longest wait between two attempts to reach a lost server.
const MAX_RETRY: Duration = Duration::from_secs(60);

/// Connects the MCP servers of the `mcp.json` files and those code adds
/// ([`AddMcpServer`]), and offers their tools.
#[derive(Default)]
pub struct McpPlugin;

impl Plugin for McpPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<McpCorePlugin>() {
            app.add_plugins(McpCorePlugin);
        }
        app.insert_resource(Files);
    }
}

/// Connects the MCP servers code adds ([`AddMcpServer`]) and offers their
/// tools, with `/mcp`, and reads no `mcp.json`: what a front adds for the
/// servers its client names, so the user's own join only with [`McpPlugin`]
/// on. Add it unless `app.is_plugin_added::<McpCorePlugin>()`.
#[derive(Default)]
pub struct McpCorePlugin;

impl Plugin for McpCorePlugin {
    fn finish(&self, app: &mut App) {
        app.init_resource::<Project>();
    }

    fn build(&self, app: &mut App) {
        app.add_setting::<McpSettings>("mcp")
            .init_resource::<RigHome>()
            .init_resource::<Processes>()
            .init_resource::<Groups>()
            .register_type::<McpStatus>()
            // A run that keeps no session, such as `rig acp`'s, runs no
            // agent to use them.
            .add_systems(
                Startup,
                connect_configured
                    .run_if(resource_exists::<Files>)
                    .run_if(|cli: Option<Res<Cli>>| !cli.is_some_and(|cli| cli.sessionless())),
            )
            .add_observer(add_server)
            .add_observer(hold)
            .add_observer(release)
            .add_observer(apply_event)
            .add_observer(close_link)
            .add_systems(
                Last,
                kill_on_exit
                    .in_set(OnAppExitSystems)
                    .run_if(on_message::<AppExit>),
            )
            // A server's task that ended without a word holds no start.
            .add_observer(
                |gone: On<Remove<Running>>,
                 servers: Query<(), With<McpServer>>,
                 mut commands: Commands| {
                    if servers.contains(gone.entity)
                        && let Ok(mut server) = commands.get_entity(gone.entity)
                    {
                        server.try_remove::<Starting>();
                    }
                },
            )
            .add_command("mcp", USAGE, mcp_command);
    }
}

/// That [`McpPlugin`] is on: the `mcp.json` files are read, and `/mcp`
/// changes them.
#[derive(Resource)]
struct Files;

/// Connects the server `name` for this run, as a client of rig's names it:
/// in no file, its values taken as they are (no
/// `${VAR}`), shown by `/mcp` as `(client)`. It takes the name over a
/// server of a file, which then does not run (a notice says so), even
/// when this one fails or is refused. A second server of a name a client
/// gave is refused with a notice. Trigger it before `Startup`, as in
/// `PreStartup`: the files' servers start then. Its entity holds
/// [`Starting`] until its first attempt connected or failed.
#[derive(Event, Clone, Debug)]
pub struct AddMcpServer {
    /// The name its tools are prefixed with, as [`McpServer::name`].
    pub name: String,
    /// How it is reached, or why the client's declaration names no way rig
    /// can: the server is then listed invalid, holding its name.
    pub spec: Result<ServerSpec, String>,
}

/// `/mcp`'s line of help.
const USAGE: &str = "MCP servers; `/mcp add [--user|--project] <name> <command…|url>`, `/mcp \
                     remove [--user|--project] <name>`";

/// What `/mcp` takes, for a line it does not.
const COMMANDS: &str = "/mcp, /mcp add [--user|--project] <name> <command…|url> or /mcp remove \
                        [--user|--project] <name>";

/// How long an attempt waits for a server by default: 30 seconds.
const CONNECT_TIMEOUT: NonZeroU64 = match NonZeroU64::new(30) {
    Some(secs) => secs,
    None => NonZeroU64::MIN,
};

/// The `[mcp]` table of `settings.toml`, read from the user's file only.
#[derive(Resource, Deserialize, Clone, Debug)]
#[serde(default, deny_unknown_fields)]
pub struct McpSettings {
    /// Seconds before a lost server is connected again; the wait doubles
    /// up to a minute and starts over once the server answers. 0 does not
    /// read, so the settings refuse the run.
    pub first_retry_secs: NonZeroU64,
    /// Seconds an attempt waits for a server to start, answer and list its
    /// tools before it counts as failed; what a headless run waits at most
    /// for a server. 0 does not read.
    pub connect_timeout_secs: NonZeroU64,
    /// The servers, of any file or client, trusted to say which of their
    /// tools only read. A tool of one whose `readOnlyHint` says so only
    /// reads: its calls run beside others, and again after a restart in a
    /// restored turn a plugin carries on. Another server's word is a hint:
    /// a call it says only reads still runs beside the reply's other
    /// readers, but a restart never runs it again.
    pub trust_read_only: Vec<String>,
}

impl Default for McpSettings {
    fn default() -> Self {
        Self {
            first_retry_secs: NonZeroU64::MIN,
            connect_timeout_secs: CONNECT_TIMEOUT,
            trust_read_only: Vec::new(),
        }
    }
}

/// A configured server, on its entity, named `mcp:<name>`.
#[derive(Component, Clone, Debug)]
pub struct McpServer {
    /// The name its tools are prefixed with.
    pub name: String,
    /// The file it is in.
    pub scope: Scope,
    /// Its object as the file writes it.
    pub written: Value,
}

/// Where a server is.
#[derive(Reflect, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum McpState {
    #[default]
    Connecting,
    Connected,
    Retrying,
    /// An [`McpHold`] keeps it from running.
    Held,
    /// `"disabled": true` in its file.
    Disabled,
    /// Its object does not describe a server; `last_error` says why.
    Invalid,
}

/// While a server's entity carries it, the server does not run: one
/// spawned with it, or given it before it starts, waits, and one given it
/// while it runs stops. Removing it starts the server. `/mcp` shows the
/// reason it holds. Only a policy plugin adds one, typically in an
/// observer of `On<Add<McpServer>>`, which runs before the server starts.
#[derive(Component, Clone, Debug)]
pub struct McpHold(pub String);

/// A server's state, for `/mcp` and views.
#[derive(Component, Reflect, Clone, Debug, Default)]
#[reflect(Component, Clone, Debug, Default)]
pub struct McpStatus {
    pub state: McpState,
    /// How many tools it offers.
    pub tools: usize,
    /// Why it was last lost, failed to connect, or is invalid.
    pub last_error: Option<String>,
    /// The process of a stdio server, while it runs.
    pub pid: Option<u32>,
}

/// A tool of the server it names. Its `ToolDef` carries the server's
/// annotations (`readOnlyHint`, `destructiveHint`, …) for approval plugins.
#[derive(Component, Debug)]
#[relationship(relationship_target = McpTools)]
pub struct McpToolOf(pub Entity);

/// A server's tools, despawned with it.
#[derive(Component, Debug)]
#[relationship_target(relationship = McpToolOf, linked_spawn)]
pub struct McpTools(Vec<Entity>);

/// What a server's connection task tells the app.
enum ServerEvent {
    Connecting,
    Started(u32),
    Connected(McpConnection, Vec<McpDefinition>),
    Listed(Vec<McpDefinition>),
    /// A list after the first failed, saying why.
    Unlisted(String),
    Lost(String),
}

/// A connected server's connection, open while its entity holds it and
/// its tools.
#[derive(Component)]
struct McpLink(McpConnection);

/// A server task's way to its entity, where each event arrives as a
/// `Done<ServerEvent>` (a [`Running::stream`] of the channel), which holds
/// at most [`EVENTS`]: the task waits for room.
#[derive(Clone)]
struct Outbox(Sender<ServerEvent>);

/// The most events of a server that wait for its entity.
const EVENTS: usize = 16;

impl Outbox {
    async fn send(&self, event: ServerEvent) {
        self.0.send(event).await.ok();
    }
}

/// The most bytes of one message a stdio server writes: far above any
/// real message, below what would exhaust memory.
pub const LINE_LIMIT: usize = 64 * 1024 * 1024;

/// A server as the files name it.
struct Configured {
    scope: Scope,
    /// Its object as written.
    written: Value,
    /// What it describes; `None` when disabled.
    spec: Result<Option<ServerSpec>, String>,
}

impl Configured {
    /// How it is, before any attempt to reach it; `None` when it starts.
    fn idle(&self) -> Option<McpStatus> {
        let (state, why) = match &self.spec {
            Err(why) => (McpState::Invalid, Some(why.clone())),
            Ok(None) => (McpState::Disabled, None),
            Ok(Some(_)) => return None,
        };
        Some(McpStatus {
            state,
            last_error: why,
            ..McpStatus::default()
        })
    }
}

/// What the files name: the server each name stands for, and what was
/// wrong with a file or a server in one.
struct Read {
    servers: BTreeMap<String, Configured>,
    problems: Vec<String>,
}

impl Read {
    /// Takes out the servers of the names a client gave, which do not
    /// start: each name and the file it is in.
    fn give_way(&mut self, clients: &HashSet<String>) -> Vec<(String, Scope)> {
        let mut gone = Vec::new();
        self.servers.retain(|name, configured| {
            let kept = !clients.contains(name);
            if !kept {
                gone.push((name.clone(), configured.scope));
            }
            kept
        });
        gone
    }
}

/// What starting a server reads.
#[derive(SystemParam)]
struct Connect<'w> {
    runtime: Option<Res<'w, TokioRuntime>>,
    wake: Res<'w, Wake>,
    settings: Res<'w, McpSettings>,
    processes: Res<'w, Processes>,
    groups: Res<'w, Groups>,
    project: Res<'w, Project>,
    home: Res<'w, RigHome>,
    workspace: Option<Res<'w, ProjectWorkspace>>,
}

impl Connect<'_> {
    /// The user's file and the project's, but for a project on another
    /// host, whose `.rig` is not on this machine, or one without a `.rig`
    /// of its own ([`Project::config`]).
    fn files(&self) -> Vec<McpFile> {
        let home = self.home.0.root();
        let config = self.project.config.as_deref();
        let project = config.filter(|_| self.no_project_file().is_none());
        let project = project.map(|config| config::project_file(home, config));
        [config::user_file(home)]
            .into_iter()
            .chain(project)
            .collect()
    }

    /// Why the project has no file of its own, if it has none.
    fn no_project_file(&self) -> Option<&'static str> {
        if self.workspace.as_ref().and_then(|it| it.0.host()).is_some() {
            Some("This project is on another host: rig reads only RIG_HOME/mcp.json for it.")
        } else if self.project.config.is_none() {
            Some(
                "This project's .rig is RIG_HOME, or it is your home folder: its servers are \
                 those of RIG_HOME/mcp.json.",
            )
        } else {
            None
        }
    }

    /// The file of `scope`.
    ///
    /// # Errors
    ///
    /// For the project's, when it has none ([`Self::no_project_file`]).
    fn file(&self, scope: Scope) -> Result<McpFile, String> {
        let file = self.files().into_iter().find(|file| file.scope == scope);
        file.ok_or_else(|| self.no_project_file().unwrap_or_default().to_owned())
    }

    /// Every server the files name, by name, the project's in place of the
    /// user's of the same name; and what was wrong with a file, or with a
    /// server in one.
    fn configured(&self) -> Read {
        let mut servers = BTreeMap::new();
        let mut problems = Vec::new();
        for file in self.files() {
            let named = match file.servers() {
                Ok(named) => named,
                Err(why) => {
                    problems.push(why);
                    continue;
                }
            };
            for (name, written) in named {
                let spec = config::check_name(&name).and_then(|()| ServerSpec::parse(&written));
                if let Err(why) = &spec {
                    let path = file.path();
                    problems.push(format!("{}: server `{name}`: {why}", path.display()));
                }
                let configured = Configured {
                    scope: file.scope,
                    written,
                    spec,
                };
                servers.insert(name, configured);
            }
        }
        Read { servers, problems }
    }

    /// Spawns the server `name`, connecting it unless it is disabled or
    /// invalid.
    fn spawn(
        &self,
        commands: &mut Commands,
        name: String,
        configured: &Configured,
    ) -> Result<(), String> {
        let server = McpServer {
            name: name.clone(),
            scope: configured.scope,
            written: configured.written.clone(),
        };
        let label = Name::new(format!("mcp:{name}"));
        match &configured.spec {
            Ok(Some(spec)) => self.connect(commands, server, label, spec.clone()),
            _ => {
                commands.spawn((label, server, configured.idle().unwrap_or_default()));
                Ok(())
            }
        }
    }

    /// Spawns `server`, which starts unless a policy holds it ([`McpHold`]).
    fn connect(
        &self,
        commands: &mut Commands,
        server: McpServer,
        label: Name,
        spec: ServerSpec,
    ) -> Result<(), String> {
        if self.runtime.is_none() {
            return Err("MCP is off: rig has no tokio runtime.".to_owned());
        }
        let entity = commands
            .spawn((label, McpStatus::default(), server, Reach(spec)))
            .id();
        // After the observers of its spawn, where a policy holds it.
        commands.queue(move |world: &mut World| start(world, entity));
        Ok(())
    }

    /// Starts connecting the server of `entity`, reached by `spec`: it holds
    /// a headless start ([`Starting`]) until its first attempt ends.
    fn run(&self, commands: &mut Commands, entity: Entity, name: String, spec: ServerSpec) {
        let Some(runtime) = &self.runtime else {
            return;
        };
        let (sender, events) = async_channel::bounded(EVENTS);
        let start = Start {
            spec,
            processes: self.processes.clone(),
            groups: self.groups.clone(),
            root: self.project.root.clone(),
            limit: Duration::from_secs(self.settings.connect_timeout_secs.get()),
            line_limit: LINE_LIMIT,
        };
        let first = Duration::from_secs(self.settings.first_retry_secs.get());
        // The connection lasts until nothing reads its events: until the
        // server's `Running` is gone.
        let closed = sender.clone();
        let connection = supervise(name, start, first, Outbox(sender));
        runtime
            .0
            .spawn(future::or(connection, async move { closed.closed().await }));
        let pool = IoTaskPool::get_or_init(TaskPool::default);
        commands.entity(entity).insert((
            McpStatus::default(),
            Starting,
            Running::stream(pool, &self.wake, events),
        ));
    }
}

/// How a server that can connect is reached, on its entity.
#[derive(Component)]
struct Reach(ServerSpec);

/// Starts the server of `entity` unless it runs, is held or is gone.
fn start(world: &mut World, entity: Entity) {
    world.run_system_cached_with(start_server, entity).ok();
}

fn start_server(
    In(entity): In<Entity>,
    connect: Connect,
    servers: Query<(&McpServer, &Reach), (Without<McpHold>, Without<Running>)>,
    mut commands: Commands,
) {
    if let Ok((server, Reach(spec))) = servers.get(entity) {
        connect.run(&mut commands, entity, server.name.clone(), spec.clone());
    }
}

/// Stops a server as a hold comes, and shows it held.
fn hold(
    added: On<Add<McpHold>>,
    mut servers: Query<&mut McpStatus, With<Reach>>,
    mut commands: Commands,
) {
    let Ok(mut status) = servers.get_mut(added.entity) else {
        return;
    };
    *status = McpStatus {
        state: McpState::Held,
        ..McpStatus::default()
    };
    commands
        .entity(added.entity)
        .remove::<(Running, McpLink, Starting)>()
        .despawn_related::<McpTools>();
}

/// Starts a server as its hold goes.
fn release(removed: On<Remove<McpHold>>, mut commands: Commands) {
    let entity = removed.entity;
    commands.queue(move |world: &mut World| start(world, entity));
}

/// Spawns every server the files name, and says what was wrong with the
/// files.
fn connect_configured(
    connect: Connect,
    servers: Query<&McpServer>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let mut read = connect.configured();
    for why in &read.problems {
        notices.write(Notice::error(None, why.clone()));
    }
    for (name, scope) in read.give_way(&clients(servers.iter())) {
        notices.write(Notice::warn(None, gives_way(&name, scope)));
    }
    for (name, configured) in read.servers {
        if let Err(why) = connect.spawn(&mut commands, name, &configured) {
            notices.write(Notice::error(None, why));
        }
    }
}

/// The names of the servers a client gave.
fn clients<'a>(servers: impl Iterator<Item = &'a McpServer>) -> HashSet<String> {
    let clients = servers.filter(|server| server.scope == Scope::Client);
    clients.map(|server| server.name.clone()).collect()
}

/// Connects the server a client gave for this run ([`AddMcpServer`]),
/// stopping a server of a file of that name, or says why it does not.
fn add_server(
    add: On<AddMcpServer>,
    connect: Connect,
    servers: Query<(Entity, &McpServer)>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let AddMcpServer { name, spec } = add.event();
    let taken = || servers.iter().filter(|(_, server)| &server.name == name);
    if taken().any(|(_, server)| server.scope == Scope::Client) {
        let text = format!("MCP server `{name}`: the client named two; the first stays.");
        notices.write(Notice::warn(None, text));
        return;
    }
    let server = McpServer {
        name: name.clone(),
        scope: Scope::Client,
        written: Value::Null,
    };
    let label = Name::new(format!("mcp:{name}"));
    for (entity, file) in taken() {
        commands.entity(entity).despawn();
        notices.write(Notice::warn(None, gives_way(name, file.scope)));
    }
    let connected = config::check_name(name)
        .and(spec.clone())
        .and_then(|spec| connect.connect(&mut commands, server.clone(), label.clone(), spec));
    if let Err(why) = connected {
        let text = format!("MCP server `{name}` could not connect: {why}.");
        notices.write(Notice::warn(None, text));
        let status = McpStatus {
            state: McpState::Invalid,
            last_error: Some(why),
            ..McpStatus::default()
        };
        commands.spawn((label, server, status));
    }
}

/// Why the server `name` of the `scope` file does not run.
fn gives_way(name: &str, scope: Scope) -> String {
    format!(
        "MCP server `{name}` of the {scope} mcp.json does not run: the client named one of that \
         name for this session."
    )
}

/// The message a stdio server's stdout refused, which ended its
/// connection, if one did.
type Refused = Arc<Mutex<Option<TooLarge>>>;

/// How one server's process or endpoint is reached.
struct Start {
    spec: ServerSpec,
    processes: Processes,
    groups: Groups,
    root: PathBuf,
    /// How long an attempt may take to list the tools.
    limit: Duration,
    /// The most bytes of a stdio server's message ([`LINE_LIMIT`]).
    line_limit: usize,
}

/// Connects `name` again and again: lists its tools, waits for it to go,
/// then waits before the next attempt.
async fn supervise(name: String, start: Start, first: Duration, outbox: Outbox) {
    let mut wait = first;
    loop {
        outbox.send(ServerEvent::Connecting).await;
        let lost = match start.run(&outbox).await {
            Ok(()) => {
                wait = first;
                format!("server `{name}` disconnected")
            }
            Err(error) => error,
        };
        outbox.send(ServerEvent::Lost(lost)).await;
        tokio::time::sleep(wait).await;
        wait = (wait * 2).min(MAX_RETRY.max(first));
    }
}

/// The process groups of the stdio servers that run, each from the moment
/// its process started: as the app exits all are killed at once, and none
/// starts after. Starting and closing each hold its lock, so no process is
/// ever outside it. Each server's [`Process`] kills its group as it is
/// dropped, so nothing the server started outlives it.
#[derive(Resource, Clone, Default)]
struct Groups(Arc<Mutex<Live>>);

#[derive(Default)]
struct Live {
    closed: bool,
    groups: Vec<process::Group>,
}

impl Groups {
    fn live(&self) -> MutexGuard<'_, Live> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Starts `command`, its process in the registry; none once closed.
    fn spawn(&self, command: std::process::Command) -> io::Result<Process> {
        let mut live = self.live();
        if live.closed {
            return Err(io::Error::other("rig is exiting"));
        }
        let process = Process::start_with_stdio(command)?;
        live.groups.retain(process::Group::running);
        live.groups.push(process.group());
        Ok(process)
    }

    /// Kills every group, and starts none from now on.
    fn close(&self) {
        let mut live = self.live();
        live.closed = true;
        live.groups.drain(..).for_each(|group| group.kill());
    }
}

/// A connection open, and the process it is to, which lives as long as
/// this.
type Reached = (McpConnection, Option<Process>);

/// [`Reached`], with the server's tools.
type Opened = (McpConnection, Vec<McpDefinition>, Option<Process>);

impl Start {
    /// One connection to the server: `Ok` once it closed after answering;
    /// a message past the limit that ended it, at any point, is why it
    /// failed.
    async fn run(&self, outbox: &Outbox) -> Result<(), String> {
        let refused = Refused::default();
        let ran = self.connection(outbox, &refused).await;
        match *refused.lock().unwrap_or_else(PoisonError::into_inner) {
            Some(too_large) => Err(format!("the server sent a message of {too_large}")),
            None => ran,
        }
    }

    /// [`Start::run`]'s connection. Every list of its tools is asked for
    /// here, one at a time, so no older list replaces a newer one.
    async fn connection(&self, outbox: &Outbox, refused: &Refused) -> Result<(), String> {
        let limit = self.limit.as_secs();
        let (changed, changes) = async_channel::bounded(1);
        let opened = async {
            let (connection, child) = self.open(outbox, Listener(changed), refused).await?;
            // rmcp hands each notification to a task of its own, so a change
            // the server said before a list's answer may reach the listener
            // just after it: the tools count as in once a list stays quiet
            // for a moment, or after a few lists, the next change left to
            // the lists that follow.
            let mut tools = Vec::new();
            for _ in 0..GRACE_LISTS {
                tools = list(&connection, &changes).await?;
                tokio::time::sleep(CHANGE_GRACE).await;
                if changes.is_empty() {
                    break;
                }
            }
            Ok::<Opened, String>((connection, tools, child))
        };
        let (connection, tools, child) = tokio::time::timeout(self.limit, opened)
            .await
            .map_err(|_| format!("no answer within {limit} s (`[mcp] connect_timeout_secs`)"))??;
        outbox
            .send(ServerEvent::Connected(connection.clone(), tools))
            .await;
        loop {
            let closed = async {
                connection.closed().await;
                false
            };
            let changed = async { changes.recv().await.is_ok() };
            if !future::or(closed, changed).await {
                break;
            }
            // Each list is bounded too, so one left unanswered never holds
            // back the next change.
            let listed = tokio::time::timeout(self.limit, list(&connection, &changes)).await;
            match listed
                .map_err(|_| format!("no answer within {limit} s (`[mcp] connect_timeout_secs`)"))
            {
                Ok(Ok(tools)) => outbox.send(ServerEvent::Listed(tools)).await,
                Ok(Err(error)) | Err(error) => outbox.send(ServerEvent::Unlisted(error)).await,
            }
        }
        // A stdio server's process group is killed as it is dropped.
        drop(child);
        Ok(())
    }

    /// Starts or reaches the server, with `client` as its MCP client.
    async fn open(
        &self,
        outbox: &Outbox,
        client: Listener,
        refused: &Refused,
    ) -> Result<Reached, String> {
        let (service, child) = match &self.spec {
            ServerSpec::Stdio {
                command,
                args,
                env,
                cwd,
            } => {
                let mut launched = self.processes.launcher().command(command);
                launched
                    .args(args)
                    .envs(env)
                    .stdin(Stdio::piped())
                    .stdout(Stdio::piped())
                    .stderr(Stdio::null());
                if let Some(cwd) = cwd {
                    launched.current_dir(self.root.join(cwd));
                }
                let mut child = self
                    .groups
                    .spawn(launched)
                    .map_err(|error| format!("could not start `{command}`: {error}"))?;
                outbox.send(ServerEvent::Started(child.id())).await;
                let pipes = (child.take_stdout(), child.take_stdin());
                let (Some(stdout), Some(stdin)) = pipes else {
                    return Err(format!("`{command}` has no pipes"));
                };
                let pipes = tokio::process::ChildStdout::from_std(stdout)
                    .and_then(|stdout| Ok((stdout, tokio::process::ChildStdin::from_std(stdin)?)));
                let (stdout, stdin) =
                    pipes.map_err(|error| format!("`{command}`'s pipes: {error}"))?;
                // rmcp frames the lines; only a line past the limit ends them.
                let stdout = Limited::new(stdout, self.line_limit, refused.clone());
                let service = client.serve((stdout, stdin)).await.map_err(unreached)?;
                (service, Some(child))
            }
            ServerSpec::Http { url, headers } => {
                let mut sent = HashMap::new();
                for (name, value) in headers {
                    let name = http::HeaderName::try_from(name.as_str())
                        .map_err(|error| format!("header `{name}`: {error}"))?;
                    let value = http::HeaderValue::try_from(value.as_str())
                        .map_err(|error| format!("header `{name}`: {error}"))?;
                    sent.insert(name, value);
                }
                let config = StreamableHttpClientTransportConfig::with_uri(url.as_str())
                    .custom_headers(sent);
                let transport = StreamableHttpClientTransport::from_config(config);
                let service = client.serve(transport).await.map_err(unreached)?;
                (service, None)
            }
        };
        Ok((McpConnection::new(service), child))
    }
}

/// The server's tools, none for a server that offers none (it says so as
/// it starts); a change said while they are listed waits in `changes`.
async fn list(
    connection: &McpConnection,
    changes: &Receiver<()>,
) -> Result<Vec<McpDefinition>, String> {
    changes.try_recv().ok();
    let offers = connection
        .peer()
        .peer_info()
        .is_some_and(|info| info.capabilities.tools.is_some());
    if !offers {
        return Ok(Vec::new());
    }
    connection
        .peer()
        .list_all_tools()
        .await
        .map_err(|error| error.to_string())
}

/// Why the MCP handshake with a server failed, in words without rmcp's
/// transport type names. A server that wants an OAuth sign-in is told as such.
fn unreached(error: ClientInitializeError) -> String {
    if error.is_authorization_required() {
        return "it asks for a sign-in (OAuth), which rig does not support. A server that \
                takes a token in a header works: see \"Servers: `mcp.json`\" in the \
                rig-mcp-client README"
            .to_owned();
    }
    match error {
        ClientInitializeError::TransportError { error, context } => {
            // An HTTP transport's error already names what it was doing.
            let why = error.error.to_string();
            if why.contains(context.as_ref()) {
                why
            } else {
                format!("{why}, when {context}")
            }
        }
        error => error.to_string(),
    }
}

/// The MCP client of one connection: tells its task when the server says
/// its tools changed, holding at most one such word.
struct Listener(Sender<()>);

impl ClientHandler for Listener {
    /// Offers the newest released protocol version, not rmcp's draft
    /// `LATEST`: some servers accept a draft offer and then refuse the session.
    fn get_info(&self) -> ClientConfig {
        let rig = Implementation::new("rig", env!("CARGO_PKG_VERSION"));
        ClientConfig::new(ClientCapabilities::default(), rig)
            .with_protocol_version(ProtocolVersion::LATEST_WITH_INITIALIZE)
    }

    async fn on_tool_list_changed(&self, _: NotificationContext<RoleClient>) {
        self.0.try_send(()).ok();
    }
}

/// Applies what a server's task sent. The first attempt's end, and the
/// loss of a connected server, are told in a notice.
fn apply_event(
    event: On<Insert<Done<ServerEvent>>>,
    mut servers: Query<(
        &McpServer,
        &mut McpStatus,
        &Done<ServerEvent>,
        Option<&McpLink>,
        Has<Starting>,
    )>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let server = event.entity;
    let Ok((spec, mut status, Done(event), link, starting)) = servers.get_mut(server) else {
        return;
    };
    let name = spec.name.clone();
    match event {
        ServerEvent::Connecting => status.state = McpState::Connecting,
        ServerEvent::Started(pid) => status.pid = Some(*pid),
        ServerEvent::Connected(connection, tools) => {
            commands.entity(server).insert(McpLink(connection.clone()));
            status.state = McpState::Connected;
            let (connection, tools) = (connection.clone(), tools.clone());
            commands.queue(move |world: &mut World| {
                register(world, server, &connection, tools, true);
            });
            // After its tools are in.
            commands.entity(server).remove::<Starting>();
        }
        ServerEvent::Lost(why) => {
            let text = match (status.state, starting) {
                (McpState::Connected, _) => Some(format!(
                    "MCP server `{name}` was lost: {why}; reconnecting."
                )),
                (_, true) => Some(format!("MCP server `{name}` could not connect: {why}.")),
                _ => None,
            };
            if let Some(text) = text {
                notices.write(Notice::warn(None, text));
            }
            *status = McpStatus {
                state: McpState::Retrying,
                last_error: Some(why.clone()),
                ..McpStatus::default()
            };
            commands
                .entity(server)
                .remove::<(McpLink, Starting)>()
                .despawn_related::<McpTools>();
        }
        ServerEvent::Listed(tools) => {
            // A connection's list after its first; one that outlived its
            // connection finds no link.
            let Some(McpLink(connection)) = link else {
                return;
            };
            let (connection, tools) = (connection.clone(), tools.clone());
            commands.queue(move |world: &mut World| {
                register(world, server, &connection, tools, false);
            });
        }
        ServerEvent::Unlisted(why) => {
            let text = format!(
                "MCP server `{name}`: its tools could not be listed again: {why}; the last \
                 list stays."
            );
            notices.write(Notice::warn(None, text));
        }
    }
}

/// Kills the process group of every stdio server as the app exits, at
/// once, and of every one that starts after: a server's own task would as
/// it is dropped, but may not run again before the process ends.
fn kill_on_exit(groups: Res<Groups>) {
    groups.close();
}

/// Closes a connection as its server's link goes, as it is removed,
/// held or replaced, for every call and tool still sharing it.
fn close_link(
    gone: On<Remove<McpLink>>,
    links: Query<&McpLink>,
    runtime: Option<Res<TokioRuntime>>,
) {
    if let (Ok(McpLink(connection)), Some(runtime)) = (links.get(gone.entity), runtime) {
        let connection = connection.clone();
        runtime.0.spawn(async move { connection.close().await });
    }
}

/// Replaces the tools of `server` with `tools`, saying how many it has when
/// it just `connected`, and which were left out: those whose names a
/// provider would refuse, or another tool has.
fn register(
    world: &mut World,
    server: Entity,
    connection: &McpConnection,
    tools: Vec<McpDefinition>,
    connected: bool,
) {
    let Some(McpServer { name, .. }) = world.get::<McpServer>(server).cloned() else {
        return;
    };
    let trusted = world
        .get_resource::<McpSettings>()
        .is_some_and(|settings| settings.trust_read_only.contains(&name));
    let Ok(mut entity) = world.get_entity_mut(server) else {
        return;
    };
    entity.despawn_related::<McpTools>();
    let (mut count, mut left_out) = (0, Vec::new());
    for definition in tools {
        let full = format!("{name}__{}", definition.name);
        if !config::fits_providers(&full) {
            left_out.push(format!(
                "`{full}` (model providers take only 1 to 64 letters, digits, `_` and `-`)"
            ));
            continue;
        }
        let Some((definition, handler)) = tool(&name, definition, connection) else {
            continue;
        };
        // The server's annotations are hints unless the user trusts it.
        let footprint = trusted.then(|| Touches::from_annotations(&definition.annotations));
        let options = ToolOptions { footprint };
        let Some(tool) = world.spawn_tool(definition, handler, options) else {
            left_out.push(format!("`{full}` (another tool has that name)"));
            continue;
        };
        // Attributed as the tools a plugin spawns as it builds are, so a
        // plugin filter such as tool search finds them.
        let plugin = world.get_resource::<PluginEntity<McpPlugin>>();
        let by = plugin.map(|plugin| ProvidedBy(plugin.entity()));
        if let Ok(mut tool) = world.get_entity_mut(tool) {
            tool.insert(McpToolOf(server));
            if let Some(by) = by {
                tool.insert(by);
            }
        }
        // A policy that saw the tool may have held the server: it then
        // offers none.
        if world.get::<McpHold>(server).is_some() {
            if let Ok(mut server) = world.get_entity_mut(server) {
                server.despawn_related::<McpTools>();
            }
            return;
        }
        count += 1;
    }
    if let Some(mut status) = world.get_mut::<McpStatus>(server) {
        status.tools = count;
    }
    if connected {
        let tools = if count == 1 { "tool" } else { "tools" };
        let text = format!("MCP server `{name}` connected: {count} {tools}.");
        world.write_message(Notice::info(None, text));
    }
    if !left_out.is_empty() {
        let text = format!("MCP server `{name}`: left out {}.", left_out.join(", "));
        world.write_message(Notice::warn(None, text));
    }
}

/// The definition `<server>__<tool>` of an MCP tool and its handler, as
/// rig-rmcp makes them (schema, description, annotations; the raw result
/// published with the call), the handler layered so that a call the
/// server's loss cut short says so.
fn tool(
    server: &str,
    definition: McpDefinition,
    connection: &McpConnection,
) -> Option<(ToolDefinition, ErasedHandler)> {
    let name = ToolName::new(format!("{server}__{}", definition.name)).ok()?;
    let mcp = McpTool::new(definition, connection).with_name(name);
    let (definition, handler, _) = DynamicTool::try_from(mcp).ok()?.into_parts();
    let disconnected = Disconnected {
        server: server.to_owned(),
        connection: connection.clone(),
    };
    Some((definition, handler.layered(disconnected)))
}

/// Answers a call that failed because its server went with "server `x`
/// disconnected".
struct Disconnected {
    server: String,
    connection: McpConnection,
}

impl Intercept for Disconnected {
    fn name(&self) -> String {
        "mcp-disconnected".to_owned()
    }

    async fn before(&self, _: EffectId, _: &EffectKind) -> BeforeServe {
        BeforeServe::Proceed
    }

    async fn after(
        &self,
        _: EffectId,
        _: &EffectKind,
        outcome: &Result<Answer, ErrorReport>,
    ) -> serve::AfterServe {
        let failed = match outcome {
            Ok(Answer::ToolResult { result }) => result.is_error(),
            Ok(_) => false,
            Err(_) => true,
        };
        if !failed || !self.connection.is_closed() {
            return serve::AfterServe::Keep;
        }
        let why = format!("server `{}` disconnected", self.server);
        let result = ToolResult::failed(ToolExecutionError::other(why));
        serve::AfterServe::Replace(Ok(Answer::ToolResult { result }))
    }
}

/// What `/mcp` reads and changes.
#[derive(SystemParam)]
struct Servers<'w, 's> {
    servers: Query<
        'w,
        's,
        (
            Entity,
            &'static McpServer,
            &'static McpStatus,
            Option<&'static McpHold>,
        ),
    >,
    connect: Connect<'w>,
    files: Option<Res<'w, Files>>,
    commands: Commands<'w, 's>,
}

impl Servers<'_, '_> {
    /// The names of the servers a client gave.
    fn clients(&self) -> HashSet<String> {
        clients(self.servers.iter().map(|(_, server, ..)| server))
    }

    /// Spawns `name` again as the files name it now, if they do; a server
    /// that is as it was keeps running. Whether it was spawned again.
    fn settle(&mut self, name: &str) -> Result<bool, String> {
        // The client's server of that name runs, whatever the files say.
        if self.clients().contains(name) {
            return Ok(false);
        }
        let configured = self.connect.configured().servers.remove(name);
        let mut kept = false;
        for (entity, server, ..) in &self.servers {
            if server.name != name {
                continue;
            }
            let same = configured.as_ref().is_some_and(|configured| {
                configured.scope == server.scope && configured.written == server.written
            });
            if same && !kept {
                kept = true;
            } else {
                self.commands.entity(entity).despawn();
            }
        }
        match configured {
            Some(configured) if !kept => {
                self.connect
                    .spawn(&mut self.commands, name.to_owned(), &configured)?;
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    /// Every server, a line each.
    fn list(&self) -> String {
        let mut lines: Vec<String> = self
            .servers
            .iter()
            .map(|(_, server, status, held)| {
                let name = &server.name;
                let error = status.last_error.as_deref().unwrap_or("none");
                let state = match status.state {
                    McpState::Connecting => "connecting".to_owned(),
                    McpState::Connected => {
                        let tools = if status.tools == 1 { "tool" } else { "tools" };
                        format!("connected, {} {tools}", status.tools)
                    }
                    McpState::Retrying => format!("retrying; last error: {error}"),
                    McpState::Held => held.map_or("held", |McpHold(why)| why).to_owned(),
                    McpState::Disabled => "disabled in its mcp.json".to_owned(),
                    McpState::Invalid => format!("invalid: {error}"),
                };
                format!("{name} ({}): {state}", server.scope)
            })
            .collect();
        if self.files.is_none() {
            lines.sort();
            return match lines.is_empty() {
                true => "No MCP servers: rig reads no mcp.json with rig_mcp_client::McpPlugin off."
                    .to_owned(),
                false => lines.join("\n"),
            };
        }
        let given = self.connect.configured().give_way(&self.clients());
        lines.extend(given.iter().map(|(name, scope)| {
            format!("{name} ({scope}): does not run; the client's of that name does")
        }));
        lines.sort();
        if lines.is_empty() {
            let files = self.connect.files();
            let paths: Vec<String> = files
                .iter()
                .map(|file| file.path().display().to_string())
                .collect();
            return format!(
                "No MCP servers. Add them to {}, or with /mcp add.",
                paths.join(" or ")
            );
        }
        lines.join("\n")
    }

    /// `/mcp add [--project] <name> <command…|url>`: saves the server in the
    /// user's file, or the project's, and starts it.
    fn add<'a>(&mut self, mut words: impl Iterator<Item = &'a str>) -> Result<String, String> {
        let usage = "/mcp add [--user|--project] <name> <command…|url>";
        let (scope, name) = match words.next().ok_or_else(|| format!("Use {usage}."))? {
            "--user" => (Scope::User, words.next()),
            "--project" => (Scope::Project, words.next()),
            name => (Scope::User, Some(name)),
        };
        let name = name.ok_or_else(|| format!("Use {usage}."))?;
        config::check_name(name)?;
        let target = words.next().ok_or_else(|| format!("Use {usage}."))?;
        let args: Vec<&str> = words.collect();
        let web = target.starts_with("http://") || target.starts_with("https://");
        let written = match (web, args.is_empty()) {
            (true, true) => json!({ "type": "http", "url": target }),
            (true, false) => return Err(format!("A url takes no arguments. Use {usage}.")),
            (false, true) => json!({ "command": target }),
            (false, false) => json!({ "command": target, "args": args }),
        };
        ServerSpec::parse(&written)?;
        let file = self.connect.file(scope)?;
        file.add(name, written)?;
        self.settle(name)?;
        Ok(format!(
            "Saved MCP server `{name}` in {}.",
            file.path().display()
        ))
    }

    /// `/mcp remove [--user|--project] <name>`: takes it out of that
    /// file, else the one of the server `/mcp` shows, and starts the other
    /// file's server of that name, if any.
    fn remove<'a>(&mut self, mut words: impl Iterator<Item = &'a str>) -> Result<String, String> {
        let usage = "/mcp remove [--user|--project] <name>";
        let (scope, name) = match words.next().ok_or_else(|| format!("Use {usage}."))? {
            "--user" => (
                Some(Scope::User),
                words.next().ok_or_else(|| format!("Use {usage}."))?,
            ),
            "--project" => (
                Some(Scope::Project),
                words.next().ok_or_else(|| format!("Use {usage}."))?,
            ),
            name => (None, name),
        };
        if words.next().is_some() {
            return Err(format!("Use {usage}."));
        }
        let shown = self
            .servers
            .iter()
            .find(|(_, server, ..)| server.name == name)
            .map(|(_, server, ..)| server.scope);
        let scope = match scope.or(shown) {
            None => return Err(format!("No MCP server `{name}`.")),
            Some(Scope::Client) => {
                return Err(format!(
                    "MCP server `{name}` is the client's, for this session, in no mcp.json; /mcp \
                     remove --user {name} or --project {name} removes a file's."
                ));
            }
            Some(scope) => scope,
        };
        let file = self.connect.file(scope)?;
        let path = file.path();
        if !file.remove(name)? {
            return Err(format!("{} names no server `{name}`.", path.display()));
        }
        let text = format!("Removed MCP server `{name}` from {}.", path.display());
        if !self.settle(name)? {
            return Ok(text);
        }
        let other = self.connect.file(match scope {
            Scope::User => Scope::Project,
            Scope::Project | Scope::Client => Scope::User,
        })?;
        Ok(format!(
            "{text} The one in {} takes its place; /mcp shows it.",
            other.path().display()
        ))
    }
}

/// `/mcp`, `/mcp add`, `/mcp remove`, and `/mcp <verb>` for a plugin's
/// `/mcp-<verb>`.
fn mcp_command(
    In(args): In<CommandArgs>,
    mut servers: Servers,
    registered: Query<&Name, With<SlashCommand>>,
    mut notices: MessageWriter<Notice>,
) {
    let agent = args.agent;
    let mut words = args.args.split_whitespace();
    let done = match words.next() {
        None => Ok(servers.list()),
        Some(verb @ ("add" | "remove")) if servers.files.is_none() => Err(format!(
            "/mcp {verb} changes an mcp.json, which rig reads only with rig_mcp_client::McpPlugin \
             on: rig plugin enable rig_mcp_client::McpPlugin"
        )),
        Some("add") => servers.add(words),
        Some("remove") => servers.remove(words),
        Some(verb)
            if registered
                .iter()
                .any(|name| name.as_str() == format!("/mcp-{verb}")) =>
        {
            let rest: Vec<&str> = words.collect();
            let line = format!("mcp-{verb} {}", rest.join(" "));
            servers.commands.trigger(RunCommand {
                mode: args.mode,
                ..RunCommand::new(agent, line)
            });
            return;
        }
        _ => Err(format!("Use {COMMANDS}.")),
    };
    match done {
        Ok(text) => notices.write(Notice::info(agent, text)),
        Err(why) => notices.write(Notice::error(agent, why)),
    };
}
