//! MCP servers' tools for the rig agent. The servers of `RIG_HOME/mcp.json`
//! and the approved ones of `<project>/.rig/mcp.json` (the common
//! `mcpServers` format, stdio or streamable HTTP), and those a client of
//! rig's names for the run ([`AddMcpServer`]), connect in the background on
//! rig's tokio runtime, each server's tools registered as `<server>__<tool>`
//! as the server lists them, with its annotations, and removed when it
//! disconnects; a lost server is connected again with backoff. A server of
//! the project starts only once the user approved it as it is written, in
//! every front.
//!
//! A server is an entity with an [`McpStatus`]; its tools are entities
//! [`McpToolOf`] it. Each new server holds a headless run's start
//! ([`Starting`]) until its first attempt connected or failed, which
//! `connect_timeout_secs` bounds, so `rig -p --tools <server>__<tool>`
//! can name its tools. `/mcp` shows the servers; `/mcp add`, `/mcp
//! remove`, `/mcp approve` and `/mcp deny` change the files and save it.
//! The `[mcp]` table of `settings.toml` is [`McpSettings`]. [`McpPlugin`]
//! reads the files; a front that hands rig its client's servers adds
//! [`McpCorePlugin`], which reads none.
//!
//! ```no_run
//! use rig_harness::prelude::*;
//!
//! App::new().add_plugins(rig_mcp_client::McpPlugin);
//! ```

mod approval;
mod config;

use std::collections::{BTreeMap, HashMap, HashSet};
use std::io;
use std::path::{Path, PathBuf};
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
use rig_harness::plugin_data::PluginData;
use rig_harness::prelude::*;
use rig_harness::rig_tools::process;
use rig_rmcp::{McpConnection, McpTool};
use rmcp::model::{ClientCapabilities, ClientConfig, Implementation, Tool as McpDefinition};
use rmcp::service::{NotificationContext, RoleClient};
use rmcp::transport::StreamableHttpClientTransport;
use rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig;
use rmcp::{ClientHandler, ServiceExt};
use serde::Deserialize;
use serde_json::{Value, json};

use approval::Approvals;
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
    fn build(&self, app: &mut App) {
        app.add_setting::<McpSettings>("mcp")
            .init_resource::<RigHome>()
            .init_resource::<ProjectRoot>()
            .init_resource::<Processes>()
            .init_resource::<Approvals>()
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
/// in no file, with no approval, its values taken as they are (no
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
                     remove [--user|--project] <name>`, `/mcp approve <name>`, `/mcp deny <name>`";

/// What `/mcp` takes, for a line it does not.
const COMMANDS: &str = "/mcp, /mcp add [--user|--project] <name> <command…|url>, /mcp remove \
                        [--user|--project] <name>, /mcp approve <name> or /mcp deny <name>";

/// The `[mcp]` table of `settings.toml`, read from the user's file only.
#[derive(Resource, Deserialize, Clone, Debug)]
#[serde(default, deny_unknown_fields)]
pub struct McpSettings {
    /// Seconds before a lost server is connected again, at least 1; the
    /// wait doubles up to a minute and starts over once the server answers.
    pub first_retry_secs: u64,
    /// Seconds an attempt waits for a server to start, answer and list its
    /// tools before it counts as failed, at least 1; what a headless run
    /// waits at most for a server.
    pub connect_timeout_secs: u64,
    /// The servers of `RIG_HOME/mcp.json` trusted to say which of their
    /// tools only read; a project's server never is. A tool
    /// of one whose `readOnlyHint` says so only reads: its calls run beside
    /// others, and again after a restart in a restored turn a plugin carries
    /// on. Another server's word is a hint: a call it says only reads still
    /// runs beside the reply's other readers, but a restart never runs it
    /// again.
    pub trust_read_only: Vec<String>,
}

impl Default for McpSettings {
    fn default() -> Self {
        Self {
            first_retry_secs: 1,
            connect_timeout_secs: 30,
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
    /// A project's server the user did not answer about yet.
    NotApproved,
    /// A project's server the user denied.
    Denied,
    /// `"disabled": true` in its file.
    Disabled,
    /// Its object does not describe a server; `last_error` says why.
    Invalid,
}

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
/// `Done<ServerEvent>` (a [`Running::stream`] of the channel).
#[derive(Clone)]
struct Outbox(Sender<ServerEvent>);

impl Outbox {
    fn send(&self, event: ServerEvent) {
        self.0.try_send(event).ok();
    }
}

/// A server as the files name it.
struct Configured {
    scope: Scope,
    /// Its object as written.
    written: Value,
    /// What it describes; `None` when disabled.
    spec: Result<Option<ServerSpec>, String>,
    /// For a project's server, whether the user approved it as written.
    answer: Option<bool>,
}

impl Configured {
    /// How it is, before any attempt to reach it; `None` when it starts.
    fn idle(&self) -> Option<McpStatus> {
        let (state, why) = match &self.spec {
            Err(why) => (McpState::Invalid, Some(why.clone())),
            Ok(None) => (McpState::Disabled, None),
            Ok(Some(_)) => match (self.scope, self.answer) {
                (Scope::User | Scope::Client, _) | (Scope::Project, Some(true)) => return None,
                (Scope::Project, Some(false)) => (McpState::Denied, None),
                (Scope::Project, None) => (McpState::NotApproved, None),
            },
        };
        Some(McpStatus {
            state,
            last_error: why,
            ..McpStatus::default()
        })
    }
}

/// What the files name: the server each name stands for, the project's
/// servers not approved that a server of the user's of the same name stands
/// in for meanwhile, and what was wrong with a file or a server in one.
struct Read {
    servers: BTreeMap<String, Configured>,
    shadowed: BTreeMap<String, Configured>,
    problems: Vec<String>,
}

impl Read {
    /// Takes out the servers of the names a client gave, which do not
    /// start: each name and the file it is in.
    fn give_way(&mut self, clients: &HashSet<String>) -> Vec<(String, Scope)> {
        let mut gone = Vec::new();
        for map in [&mut self.servers, &mut self.shadowed] {
            map.retain(|name, configured| {
                let kept = !clients.contains(name);
                if !kept {
                    gone.push((name.clone(), configured.scope));
                }
                kept
            });
        }
        gone.sort();
        gone
    }

    /// Each project server that waits for an answer, as a line: its name
    /// and its object as written.
    fn waiting(&self) -> Vec<String> {
        let waits = |configured: &&Configured| {
            configured
                .idle()
                .is_some_and(|status| status.state == McpState::NotApproved)
        };
        let servers = self.servers.iter().filter(|(_, it)| waits(it));
        let mut lines: Vec<String> = servers
            .map(|(name, it)| format!("  {name}: {}", config::describe(&it.written)))
            .collect();
        let shadowed = self.shadowed.iter().filter(|(_, it)| waits(it));
        lines.extend(shadowed.map(|(name, it)| {
            let written = config::describe(&it.written);
            format!("  {name} (yours of that name comes first meanwhile): {written}")
        }));
        lines
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
    root: Res<'w, ProjectRoot>,
    home: Res<'w, RigHome>,
    workspace: Option<Res<'w, ProjectWorkspace>>,
}

impl Connect<'_> {
    /// The project approvals are about: its canonical path.
    fn project(&self) -> String {
        let root = &self.root.0;
        let root = std::fs::canonicalize(root).unwrap_or_else(|_| root.clone());
        root.display().to_string()
    }

    /// The user's file and the project's, but for a project on another
    /// host, whose `.rig` is not on this machine, or one whose `.rig` is
    /// `RIG_HOME`, as the home directory's is.
    fn files(&self) -> Vec<McpFile> {
        let [user, project] = config::files(self.home.0.root(), &self.root.0);
        match self.no_project_file() {
            Some(_) => vec![user],
            None => vec![user, project],
        }
    }

    /// Why the project has no file of its own, if it has none.
    fn no_project_file(&self) -> Option<&'static str> {
        let home = self.home.0.root();
        let same = |a: &Path, b: &Path| matches!((std::fs::canonicalize(a), std::fs::canonicalize(b)), (Ok(a), Ok(b)) if a == b);
        if self.workspace.as_ref().and_then(|it| it.0.host()).is_some() {
            Some("This project is on another host: rig reads only RIG_HOME/mcp.json for it.")
        } else if self.root.0.to_str().is_none() {
            // Its approvals could not tell it from another such path.
            Some("This project's path is not UTF-8: rig reads only RIG_HOME/mcp.json for it.")
        } else if same(&self.root.0.join(".rig"), home) {
            Some(
                "This project's .rig is RIG_HOME itself: its servers are those of RIG_HOME/mcp.json.",
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

    /// Every server the files name, by name: the project's in place of the
    /// user's of the same name once approved; and what was wrong with a
    /// file, or with a server in one.
    fn configured(&self, approvals: &Approvals) -> Read {
        let mut servers = BTreeMap::new();
        let mut shadowed = BTreeMap::new();
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
                let answer = match file.scope {
                    Scope::User | Scope::Client => None,
                    Scope::Project => approvals.answer(&name, &written),
                };
                let configured = Configured {
                    scope: file.scope,
                    written,
                    spec,
                    answer,
                };
                if servers.contains_key(&name) && answer != Some(true) {
                    shadowed.insert(name, configured);
                } else {
                    servers.insert(name, configured);
                }
            }
        }
        Read {
            servers,
            shadowed,
            problems,
        }
    }

    /// Spawns the server `name`, connecting it unless it waits, is
    /// disabled or is invalid. A new server holds a headless start
    /// ([`Starting`]) until its first attempt ends.
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
        match (configured.idle(), &configured.spec) {
            (None, Ok(Some(spec))) => self.connect(commands, server, label, spec.clone()),
            (status, _) => {
                commands.spawn((label, server, status.unwrap_or_default()));
                Ok(())
            }
        }
    }

    /// Spawns `server` and starts connecting it.
    fn connect(
        &self,
        commands: &mut Commands,
        server: McpServer,
        label: Name,
        spec: ServerSpec,
    ) -> Result<(), String> {
        let Some(runtime) = &self.runtime else {
            return Err("MCP is off: rig has no tokio runtime.".to_owned());
        };
        let (sender, events) = async_channel::unbounded();
        let start = Start {
            spec,
            processes: self.processes.clone(),
            groups: self.groups.clone(),
            root: self.root.0.clone(),
            limit: Duration::from_secs(self.settings.connect_timeout_secs.max(1)),
        };
        let first = Duration::from_secs(self.settings.first_retry_secs.max(1));
        // The connection lasts until nothing reads its events: until the
        // server's entity, and its `Running` with them, is gone.
        let closed = sender.clone();
        let connection = supervise(server.name.clone(), start, first, Outbox(sender));
        runtime
            .0
            .spawn(future::or(connection, async move { closed.closed().await }));
        let pool = IoTaskPool::get_or_init(TaskPool::default);
        commands.spawn((
            label,
            McpStatus::default(),
            server,
            Starting,
            Running::stream(pool, &self.wake, events),
        ));
        Ok(())
    }
}

/// Spawns every server the files name, and says which of the project's
/// wait for the user's approval, and what was wrong with the files.
fn connect_configured(
    connect: Connect,
    data: Option<Res<PluginData>>,
    servers: Query<&McpServer>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let lock = config::lock_path(connect.home.0.root());
    let approvals = Approvals::read(data.as_deref(), connect.project(), lock);
    let mut read = connect.configured(&approvals);
    commands.insert_resource(approvals);
    for why in &read.problems {
        notices.write(Notice::error(None, why.clone()));
    }
    for (name, scope) in read.give_way(&clients(servers.iter())) {
        notices.write(Notice::warn(None, gives_way(&name, scope)));
    }
    let waiting = read.waiting();
    for (name, configured) in read.servers {
        if let Err(why) = connect.spawn(&mut commands, name, &configured) {
            notices.write(Notice::error(None, why));
        }
    }
    if !waiting.is_empty() {
        let path = connect.root.0.join(".rig/mcp.json");
        let text = format!(
            "{} names MCP servers you have not approved, so they did not start:\n{}\n/mcp \
             approve <name> starts one and remembers it; /mcp deny <name> stops asking.",
            path.display(),
            waiting.join("\n")
        );
        notices.write(Notice::warn(None, text));
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

/// How one server's process or endpoint is reached.
struct Start {
    spec: ServerSpec,
    processes: Processes,
    groups: Groups,
    root: PathBuf,
    /// How long an attempt may take to list the tools.
    limit: Duration,
}

/// Connects `name` again and again: lists its tools, waits for it to go,
/// then waits before the next attempt.
async fn supervise(name: String, start: Start, first: Duration, outbox: Outbox) {
    let mut wait = first;
    loop {
        outbox.send(ServerEvent::Connecting);
        let lost = match start.run(&outbox).await {
            Ok(()) => {
                wait = first;
                format!("server `{name}` disconnected")
            }
            Err(error) => error,
        };
        outbox.send(ServerEvent::Lost(lost));
        tokio::time::sleep(wait).await;
        wait = (wait * 2).min(MAX_RETRY.max(first));
    }
}

/// The process groups of the stdio servers that run, each from the moment
/// its process started: as the app exits all are killed at once, and none
/// starts after. Starting, killing and closing each hold its lock, so no
/// process is ever outside it.
#[derive(Resource, Clone, Default)]
struct Groups(Arc<Mutex<Live>>);

#[derive(Default)]
struct Live {
    closed: bool,
    leaders: HashSet<u32>,
}

impl Groups {
    fn live(&self) -> MutexGuard<'_, Live> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Starts `command`, its process in the registry; none once closed.
    fn spawn(&self, command: &mut tokio::process::Command) -> io::Result<Group> {
        let mut live = self.live();
        if live.closed {
            return Err(io::Error::other("rig is exiting"));
        }
        let child = command.spawn()?;
        if let Some(leader) = child.id() {
            live.leaders.insert(leader);
            process::list_group(leader);
        }
        Ok(Group {
            child,
            groups: self.clone(),
        })
    }

    /// Kills every group, and starts none from now on.
    fn close(&self) {
        let mut live = self.live();
        live.closed = true;
        live.leaders.drain().for_each(|leader| {
            kill_group(leader);
            process::unlist_group(leader);
        });
    }
}

/// Kills the process group `leader` leads; elsewhere than unix, nothing,
/// as a process is killed as it is dropped there.
fn kill_group(leader: u32) {
    #[cfg(unix)]
    process::kill_group_of(leader);
    #[cfg(not(unix))]
    let _ = leader;
}

/// A stdio server's process, which leads a process group of its own: the
/// group is killed as this is dropped, so nothing the server started
/// outlives it.
struct Group {
    child: tokio::process::Child,
    groups: Groups,
}

impl Drop for Group {
    fn drop(&mut self) {
        if let Some(leader) = self.child.id() {
            let mut live = self.groups.live();
            kill_group(leader);
            if live.leaders.remove(&leader) {
                process::unlist_group(leader);
            }
        }
    }
}

/// A connection open, and the process it is to, which lives as long as
/// this.
type Reached = (McpConnection, Option<Group>);

/// [`Reached`], with the server's tools.
type Opened = (McpConnection, Vec<McpDefinition>, Option<Group>);

impl Start {
    /// One connection to the server: `Ok` once it closed after answering.
    /// Every list of its tools is asked for here, one at a time, so no
    /// older list replaces a newer one.
    async fn run(&self, outbox: &Outbox) -> Result<(), String> {
        let limit = self.limit.as_secs();
        let (changed, changes) = async_channel::bounded(1);
        let opened = async {
            let (connection, child) = self.open(outbox, Listener(changed)).await?;
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
            .map_err(|_| format!("no answer within {limit}s"))??;
        outbox.send(ServerEvent::Connected(connection.clone(), tools));
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
            match listed.map_err(|_| format!("no answer within {limit}s")) {
                Ok(Ok(tools)) => outbox.send(ServerEvent::Listed(tools)),
                Ok(Err(error)) | Err(error) => outbox.send(ServerEvent::Unlisted(error)),
            }
        }
        // A stdio server's process group is killed as it is dropped.
        drop(child);
        Ok(())
    }

    /// Starts or reaches the server, with `client` as its MCP client.
    async fn open(&self, outbox: &Outbox, client: Listener) -> Result<Reached, String> {
        let (service, child) = match &self.spec {
            ServerSpec::Stdio {
                command,
                args,
                env,
                cwd,
            } => {
                let launched = self.processes.launcher().command(command);
                let mut process = tokio::process::Command::from(launched);
                process
                    .args(args)
                    .envs(env)
                    .stdin(Stdio::piped())
                    .stdout(Stdio::piped())
                    .stderr(Stdio::null())
                    .kill_on_drop(true);
                if let Some(cwd) = cwd {
                    process.current_dir(self.root.join(cwd));
                }
                let mut child = self
                    .groups
                    .spawn(&mut process)
                    .map_err(|error| format!("could not start `{command}`: {error}"))?;
                if let Some(pid) = child.child.id() {
                    outbox.send(ServerEvent::Started(pid));
                }
                let pipes = (child.child.stdout.take(), child.child.stdin.take());
                let (Some(stdout), Some(stdin)) = pipes else {
                    return Err(format!("`{command}` has no pipes"));
                };
                let service = client
                    .serve((stdout, stdin))
                    .await
                    .map_err(|error| error.to_string())?;
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
                let service = client
                    .serve(transport)
                    .await
                    .map_err(|error| error.to_string())?;
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

/// The MCP client of one connection: tells its task when the server says
/// its tools changed, holding at most one such word.
struct Listener(Sender<()>);

impl ClientHandler for Listener {
    fn get_info(&self) -> ClientConfig {
        let rig = Implementation::new("rig", env!("CARGO_PKG_VERSION"));
        ClientConfig::new(ClientCapabilities::default(), rig)
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
/// denied or replaced, for every call and tool still sharing it.
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
    let Some(McpServer { name, scope, .. }) = world.get::<McpServer>(server).cloned() else {
        return;
    };
    let trusted = scope == Scope::User
        && world
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
        match world.spawn_tool(definition, handler, options) {
            Some(tool) => {
                world.entity_mut(tool).insert(McpToolOf(server));
                // Attributed as the tools a plugin spawns as it builds are,
                // so a plugin filter such as tool search finds them.
                if let Some(plugin) = world.get_resource::<PluginEntity<McpPlugin>>() {
                    let plugin = plugin.entity();
                    world.entity_mut(tool).insert(ProvidedBy(plugin));
                }
                count += 1;
            }
            None => left_out.push(format!("`{full}` (another tool has that name)")),
        }
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
    servers: Query<'w, 's, (Entity, &'static McpServer, &'static McpStatus)>,
    connect: Connect<'w>,
    approvals: ResMut<'w, Approvals>,
    files: Option<Res<'w, Files>>,
    commands: Commands<'w, 's>,
}

impl Servers<'_, '_> {
    /// The names of the servers a client gave.
    fn clients(&self) -> HashSet<String> {
        clients(self.servers.iter().map(|(_, server, _)| server))
    }

    /// Spawns `name` again as the files name it now, if they do; a server
    /// that is as it was keeps running. What became of it: its state, and
    /// whether it was spawned again.
    fn settle(&mut self, name: &str) -> Result<Option<(McpState, bool)>, String> {
        // The client's server of that name runs, whatever the files say.
        if self.clients().contains(name) {
            return Ok(None);
        }
        let configured = self
            .connect
            .configured(&self.approvals)
            .servers
            .remove(name);
        let idle = configured
            .as_ref()
            .map(|configured| configured.idle().map(|status| status.state));
        let waiting = |state: McpState| {
            let running = [
                McpState::Connecting,
                McpState::Connected,
                McpState::Retrying,
            ];
            (!running.contains(&state)).then_some(state)
        };
        let mut kept = None;
        for (entity, server, status) in &self.servers {
            if server.name != name {
                continue;
            }
            let same = configured.as_ref().is_some_and(|configured| {
                configured.scope == server.scope && configured.written == server.written
            });
            if same && idle == Some(waiting(status.state)) && kept.is_none() {
                kept = Some(status.state);
            } else {
                self.commands.entity(entity).despawn();
            }
        }
        let Some(configured) = configured else {
            return Ok(None);
        };
        if let Some(state) = kept {
            return Ok(Some((state, false)));
        }
        let state = configured
            .idle()
            .map_or(McpState::Connecting, |status| status.state);
        self.connect
            .spawn(&mut self.commands, name.to_owned(), &configured)?;
        Ok(Some((state, true)))
    }

    /// Every server, a line each.
    fn list(&self) -> String {
        let mut lines: Vec<String> = self
            .servers
            .iter()
            .map(|(_, server, status)| {
                let name = &server.name;
                let error = status.last_error.as_deref().unwrap_or("none");
                let state = match status.state {
                    McpState::Connecting => "connecting".to_owned(),
                    McpState::Connected => {
                        let tools = if status.tools == 1 { "tool" } else { "tools" };
                        format!("connected, {} {tools}", status.tools)
                    }
                    McpState::Retrying => format!("retrying; last error: {error}"),
                    McpState::NotApproved => format!(
                        "not approved; /mcp approve {name} starts it as written: {}",
                        config::describe(&server.written)
                    ),
                    McpState::Denied => format!(
                        "denied; /mcp approve {name} starts it as written: {}",
                        config::describe(&server.written)
                    ),
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
        let mut read = self.connect.configured(&self.approvals);
        let given = read.give_way(&self.clients());
        lines.extend(given.iter().map(|(name, scope)| {
            format!("{name} ({scope}): does not run; the client's of that name does")
        }));
        lines.extend(read.shadowed.iter().map(|(name, configured)| {
            let written = config::describe(&configured.written);
            let idle = configured.idle();
            let error = idle.as_ref().and_then(|it| it.last_error.clone());
            match idle.map(|it| it.state) {
                Some(McpState::Invalid) => format!(
                    "{name} (project): invalid: {}; yours comes first",
                    error.unwrap_or_default()
                ),
                Some(McpState::Disabled) => {
                    format!("{name} (project): disabled in its mcp.json; yours comes first")
                }
                Some(McpState::Denied) => format!(
                    "{name} (project): denied, yours comes first meanwhile; /mcp approve {name} starts \
                     it in place of yours, as written: {written}"
                ),
                _ => format!(
                    "{name} (project): not approved, yours comes first meanwhile; /mcp approve {name} \
                     starts it in place of yours, as written: {written}"
                ),
            }
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
    /// user's file, or the project's, approved, and starts it.
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
        file.add(name, written.clone())?;
        if scope == Scope::Project {
            self.approvals.set(name, written, true)?;
        }
        let path = file.path();
        Ok(match self.settle(name)? {
            Some((McpState::Connecting, true)) => {
                format!(
                    "Saved MCP server `{name}` in {}; connecting.",
                    path.display()
                )
            }
            _ => format!("Saved MCP server `{name}` in {}.", path.display()),
        })
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
            .find(|(_, server, _)| server.name == name)
            .map(|(_, server, _)| server.scope);
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
        Ok(match self.settle(name)? {
            Some((state, true)) => {
                let other = self.connect.file(match scope {
                    Scope::User => Scope::Project,
                    Scope::Project | Scope::Client => Scope::User,
                })?;
                let other = other.path();
                let other = other.display();
                match state {
                    McpState::Connecting => format!("{text} The one in {other} takes its place."),
                    McpState::NotApproved => format!(
                        "{text} The one in {other} is not approved: /mcp approve {name} starts \
                         it."
                    ),
                    McpState::Denied => format!(
                        "{text} The one in {other} is denied: /mcp approve {name} starts it."
                    ),
                    McpState::Disabled => format!("{text} The one in {other} is disabled."),
                    _ => format!("{text} The one in {other} is invalid; /mcp says why."),
                }
            }
            _ => text,
        })
    }

    /// `/mcp approve <name>` and `/mcp deny <name>`: keeps the answer about
    /// the project's server as it is written now, and starts or stops it.
    fn answer(&mut self, name: &str, approved: bool) -> Result<String, String> {
        let file = self.connect.file(Scope::Project)?;
        let written = file.servers()?.into_iter().find(|(it, _)| it == name);
        let Some((_, written)) = written else {
            return Err(format!(
                "{} names no server `{name}`.",
                file.path().display()
            ));
        };
        let shown = config::describe(&written);
        self.approvals.set(name, written, approved)?;
        let settled = self.settle(name)?;
        let state = settled.map(|(state, _)| state);
        Ok(match (approved, state) {
            (true, Some(McpState::Connecting)) => {
                format!("Approved `{name}` as written: {shown}; connecting.")
            }
            (true, Some(McpState::Disabled)) => {
                format!("Approved `{name}` as written: {shown}; it is disabled in its mcp.json.")
            }
            (true, Some(McpState::Invalid)) => {
                format!("Approved `{name}` as written: {shown}; it is invalid, /mcp says why.")
            }
            (true, _) => format!("Approved `{name}` as written: {shown}."),
            // Denied, the project's server can only give way to the user's.
            (false, _) if settled == Some((McpState::Connecting, true)) => format!(
                "Denied `{name}`: it does not start until /mcp approve {name}. Yours takes its \
                 place; connecting."
            ),
            (false, _) => format!("Denied `{name}`: it does not start until /mcp approve {name}."),
        })
    }
}

/// `/mcp`, `/mcp add`, `/mcp remove`, `/mcp approve`, `/mcp deny`.
fn mcp_command(
    In(args): In<CommandArgs>,
    mut servers: Servers,
    mut notices: MessageWriter<Notice>,
) {
    let agent = args.agent;
    let mut words = args.args.split_whitespace();
    let done = match (words.next(), words.clone().next()) {
        (None, _) => Ok(servers.list()),
        (Some(verb), _) if servers.files.is_none() => Err(format!(
            "/mcp {verb} changes an mcp.json, which rig reads only with rig_mcp_client::McpPlugin \
             on: rig plugin enable rig_mcp_client::McpPlugin"
        )),
        (Some("add"), _) => servers.add(words),
        (Some("remove"), _) => servers.remove(words),
        (Some("approve"), Some(name)) if words.clone().count() == 1 => servers.answer(name, true),
        (Some("deny"), Some(name)) if words.clone().count() == 1 => servers.answer(name, false),
        (Some(verb @ ("approve" | "deny")), _) => Err(format!("Use /mcp {verb} <name>.")),
        _ => Err(format!("Use {COMMANDS}.")),
    };
    match done {
        Ok(text) => notices.write(Notice::info(agent, text)),
        Err(why) => notices.write(Notice::error(agent, why)),
    };
}
