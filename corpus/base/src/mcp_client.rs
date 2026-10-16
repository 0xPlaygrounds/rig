//! Plugin corpus: 19. MCP client (spec `19-mcp-client.md`).
//!
//! MCP servers named in `RIG_HOME/mcp.json` and `<project>/.rig/mcp.json`
//! (the common `mcpServers` format, stdio or streamable HTTP) connect in the
//! background on rig's tokio runtime; each server's tools are registered as
//! `<server>__<tool>` as the server lists them, with its annotations, and
//! removed when it disconnects. A server is an entity with an
//! [`McpStatus`]; its tools are entities [`McpToolOf`] it. `/mcp` shows the
//! servers, `/mcp add <name> <command…>` and `/mcp remove <name>` change
//! them for the session. The `[mcp]` table of `settings.toml` sets how long
//! a lost server's first retry waits, and which servers are trusted to say
//! which of their tools only read ([`McpSettings`]).

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::process::Stdio;

use async_channel::Sender;
use bevy_ecs::system::SystemParam;
use bevy_log::warn;
use bevy_tasks::futures_lite::future;
use bevy_tasks::{IoTaskPool, TaskPool};
use rig_core::completion::ToolDefinition;
use rig_core::effect::{Answered as Answer, EffectId, EffectKind};
use rig_core::error::ErrorReport;
use rig_core::message::ToolName;
use rig_core::serve::{self, BeforeServe, ErasedHandler, Intercept};
use rig_core::tool::{DynamicTool, Touches};
use rig_harness::prelude::*;
use rig_rmcp::{McpConnection, McpTool};
use rmcp::model::{ClientConfig, Tool as McpDefinition};
use rmcp::service::{NotificationContext, RoleClient};
use rmcp::transport::StreamableHttpClientTransport;
use rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig;
use rmcp::{ClientHandler, ServiceExt};
use serde::Deserialize;

#[cfg(test)]
mod tests;

/// The longest wait between two attempts to reach a lost server.
const MAX_RETRY: Duration = Duration::from_secs(60);

/// Connects the configured MCP servers and offers their tools.
#[derive(Default)]
pub struct McpClientPlugin;

impl Plugin for McpClientPlugin {
    fn build(&self, app: &mut App) {
        app.add_setting::<McpSettings>("mcp")
            .init_resource::<RigHome>()
            .init_resource::<ProjectRoot>()
            .init_resource::<Processes>()
            .register_type::<McpStatus>()
            .add_systems(Startup, connect_configured)
            .add_observer(apply_event)
            .add_command(
                "mcp",
                "MCP servers; `/mcp add <name> <command…>`, `/mcp remove <name>`",
                mcp_command,
            );
    }
}

/// The `[mcp]` table of `settings.toml`.
#[derive(Resource, Deserialize, Clone, Debug)]
#[serde(default, deny_unknown_fields)]
pub struct McpSettings {
    /// Seconds before a lost server is connected again; the wait doubles
    /// up to a minute and starts over once the server answers.
    pub first_retry_secs: u64,
    /// The servers trusted to say which of their tools only read. A tool
    /// of one whose `readOnlyHint` says so only reads: its calls run beside
    /// others and again after a restart. Another server's word is a hint: a
    /// call it says only reads still runs beside the reply's other readers,
    /// but a restart never runs it again.
    pub trust_read_only: Vec<String>,
}

impl Default for McpSettings {
    fn default() -> Self {
        Self {
            first_retry_secs: 1,
            trust_read_only: Vec::new(),
        }
    }
}

/// One `mcp.json`: `{"mcpServers": {"<name>": {…}}}`.
#[derive(Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct ConfigFile {
    #[serde(default)]
    mcp_servers: BTreeMap<String, ServerSpec>,
}

/// How a server is reached.
#[derive(Deserialize, Clone, Debug)]
#[serde(untagged)]
pub enum ServerSpec {
    /// A child process speaking MCP on its stdin and stdout, started as
    /// [`Processes`] says.
    Stdio {
        /// The program.
        command: String,
        /// Its arguments.
        #[serde(default)]
        args: Vec<String>,
        /// Variables added to its environment.
        #[serde(default)]
        env: HashMap<String, String>,
        /// Its working directory, relative to the project.
        #[serde(default)]
        cwd: Option<PathBuf>,
    },
    /// A streamable HTTP endpoint.
    Http {
        /// The endpoint.
        url: String,
        /// Headers sent with every request.
        #[serde(default)]
        headers: HashMap<String, String>,
    },
}

/// A configured server, on its entity, named `mcp:<name>`.
#[derive(Component)]
pub struct McpServer {
    /// The name its tools are prefixed with.
    pub name: String,
    /// How it is reached.
    pub spec: ServerSpec,
}

/// Where a server is.
#[derive(Reflect, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum McpState {
    #[default]
    Connecting,
    Connected,
    Retrying,
}

/// A server's state, for `/mcp` and views.
#[derive(Component, Reflect, Clone, Debug, Default)]
#[reflect(Component, Clone, Debug, Default)]
pub struct McpStatus {
    pub state: McpState,
    /// How many tools it offers.
    pub tools: usize,
    /// Why it was last lost, or failed to connect.
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

/// What starting a server reads.
#[derive(SystemParam)]
struct Connect<'w> {
    runtime: Option<Res<'w, TokioRuntime>>,
    wake: Res<'w, Wake>,
    settings: Res<'w, McpSettings>,
    processes: Res<'w, Processes>,
    root: Res<'w, ProjectRoot>,
}

impl Connect<'_> {
    /// Spawns the server `name` and starts connecting it.
    fn spawn(&self, commands: &mut Commands, name: String, spec: ServerSpec) -> Result<(), String> {
        let Some(runtime) = &self.runtime else {
            return Err("MCP is off: rig has no tokio runtime.".to_owned());
        };
        let (sender, events) = async_channel::unbounded();
        let start = Start {
            spec: spec.clone(),
            processes: self.processes.clone(),
            root: self.root.0.clone(),
        };
        let first = Duration::from_secs(self.settings.first_retry_secs);
        // The connection lasts until nothing reads its events: until the
        // server's entity, and its `Running` with them, is gone.
        let closed = sender.clone();
        let connection = supervise(name.clone(), start, first, Outbox(sender));
        runtime
            .0
            .spawn(future::or(connection, async move { closed.closed().await }));
        let pool = IoTaskPool::get_or_init(TaskPool::default);
        commands.spawn((
            Name::new(format!("mcp:{name}")),
            McpStatus::default(),
            McpServer { name, spec },
            Running::stream(pool, &self.wake, events),
        ));
        Ok(())
    }
}

/// Connects every server of `RIG_HOME/mcp.json` and `<project>/.rig/mcp.json`;
/// the project's replaces a server of the same name.
fn connect_configured(
    connect: Connect,
    home: Res<RigHome>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let files = [
        home.0.root().join("mcp.json"),
        connect.root.0.join(".rig").join("mcp.json"),
    ];
    let mut servers = BTreeMap::new();
    for file in files {
        let Ok(text) = std::fs::read_to_string(&file) else {
            continue;
        };
        match serde_json::from_str::<ConfigFile>(&text) {
            Ok(parsed) => servers.extend(parsed.mcp_servers),
            Err(error) => {
                let why = format!("{} is not an MCP config: {error}", file.display());
                notices.write(Notice::error(None, why));
            }
        }
    }
    for (name, spec) in servers {
        if let Err(why) = connect.spawn(&mut commands, name, spec) {
            notices.write(Notice::error(None, why));
            return;
        }
    }
}

/// How one server's process or endpoint is reached.
struct Start {
    spec: ServerSpec,
    processes: Processes,
    root: PathBuf,
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

impl Start {
    /// One connection to the server: `Ok` once it closed after answering.
    async fn run(&self, outbox: &Outbox) -> Result<(), String> {
        let client = Listener(outbox.clone());
        match &self.spec {
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
                let mut child = process
                    .spawn()
                    .map_err(|error| format!("could not start `{command}`: {error}"))?;
                if let Some(pid) = child.id() {
                    outbox.send(ServerEvent::Started(pid));
                }
                let (Some(stdout), Some(stdin)) = (child.stdout.take(), child.stdin.take()) else {
                    return Err(format!("`{command}` has no pipes"));
                };
                let service = client
                    .serve((stdout, stdin))
                    .await
                    .map_err(|error| error.to_string())?;
                serve(McpConnection::new(service), outbox).await
                // `child` is killed as it is dropped here.
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
                serve(McpConnection::new(service), outbox).await
            }
        }
    }
}

/// Lists a connected server's tools, then waits for it to close.
async fn serve(connection: McpConnection, outbox: &Outbox) -> Result<(), String> {
    let tools = connection
        .peer()
        .list_all_tools()
        .await
        .map_err(|error| error.to_string())?;
    outbox.send(ServerEvent::Connected(connection.clone(), tools));
    connection.closed().await;
    Ok(())
}

/// The MCP client of one connection: lists the tools again when the server
/// says they changed.
struct Listener(Outbox);

impl ClientHandler for Listener {
    fn get_info(&self) -> ClientConfig {
        ClientConfig::default()
    }

    async fn on_tool_list_changed(&self, context: NotificationContext<RoleClient>) {
        match context.peer.list_all_tools().await {
            Ok(tools) => self.0.send(ServerEvent::Listed(tools)),
            Err(error) => warn!("MCP tools could not be listed again: {error}"),
        }
    }
}

/// Applies what a server's task sent.
fn apply_event(
    event: On<Insert<Done<ServerEvent>>>,
    mut servers: Query<(
        &McpServer,
        &mut McpStatus,
        &Done<ServerEvent>,
        Option<&McpLink>,
    )>,
    mut commands: Commands,
) {
    let server = event.entity;
    let Ok((spec, mut status, Done(event), link)) = servers.get_mut(server) else {
        return;
    };
    match event {
        ServerEvent::Connecting => status.state = McpState::Connecting,
        ServerEvent::Started(pid) => status.pid = Some(*pid),
        ServerEvent::Connected(connection, tools) => {
            commands.entity(server).insert(McpLink(connection.clone()));
            status.state = McpState::Connected;
            let (name, connection, tools) = (spec.name.clone(), connection.clone(), tools.clone());
            commands.queue(move |world: &mut World| {
                register(world, server, &name, &connection, tools);
            });
        }
        ServerEvent::Lost(why) => {
            *status = McpStatus {
                state: McpState::Retrying,
                last_error: Some(why.clone()),
                ..McpStatus::default()
            };
            commands
                .entity(server)
                .remove::<McpLink>()
                .despawn_related::<McpTools>();
        }
        ServerEvent::Listed(tools) => {
            // A connection's list after its first; one that outlived its
            // connection finds no link.
            let Some(McpLink(connection)) = link else {
                return;
            };
            let (name, connection, tools) = (spec.name.clone(), connection.clone(), tools.clone());
            commands.queue(move |world: &mut World| {
                register(world, server, &name, &connection, tools);
            });
        }
    }
}

/// Replaces the tools of `server` with `tools`.
fn register(
    world: &mut World,
    server: Entity,
    name: &str,
    connection: &McpConnection,
    tools: Vec<McpDefinition>,
) {
    let trusted = world
        .get_resource::<McpSettings>()
        .is_some_and(|settings| settings.trust_read_only.iter().any(|it| it == name));
    let Ok(mut entity) = world.get_entity_mut(server) else {
        return;
    };
    entity.despawn_related::<McpTools>();
    let mut count = 0;
    for definition in tools {
        let Some((definition, handler)) = tool(name, definition, connection) else {
            continue;
        };
        // The server's annotations are hints unless the user trusts it.
        let footprint = trusted.then(|| Touches::from_annotations(&definition.annotations));
        let options = ToolOptions { footprint };
        if let Some(tool) = world.spawn_tool(definition, handler, options) {
            world.entity_mut(tool).insert(McpToolOf(server));
            count += 1;
        }
    }
    if let Some(mut status) = world.get_mut::<McpStatus>(server) {
        status.tools = count;
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

/// `/mcp`, `/mcp add <name> <command…>`, `/mcp remove <name>`.
fn mcp_command(
    In(args): In<CommandArgs>,
    servers: Query<(Entity, &McpServer, &McpStatus)>,
    connect: Connect,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let agent = args.agent;
    let mut words = args.args.split_whitespace();
    let find = |name: &str| {
        servers
            .iter()
            .find(|(_, server, _)| server.name == name)
            .map(|(entity, ..)| entity)
    };
    match (words.next(), words.next()) {
        (None, _) => {
            let mut lines: Vec<String> = servers
                .iter()
                .map(|(_, server, status)| {
                    let error = status.last_error.as_deref().unwrap_or("none");
                    format!(
                        "{}: {:?}, {} tools, last error: {error}",
                        server.name, status.state, status.tools
                    )
                })
                .collect();
            lines.sort();
            if lines.is_empty() {
                lines.push("No MCP servers. Add them to mcp.json or with /mcp add.".to_owned());
            }
            notices.write(Notice::info(agent, lines.join("\n")));
        }
        (Some("add"), Some(name)) => {
            let Some(command) = words.next() else {
                notices.write(Notice::error(agent, "/mcp add needs a name and a command"));
                return;
            };
            if find(name).is_some() {
                let why = format!("An MCP server `{name}` exists; /mcp remove it first.");
                notices.write(Notice::error(agent, why));
                return;
            }
            let spec = ServerSpec::Stdio {
                command: command.to_owned(),
                args: words.map(str::to_owned).collect(),
                env: HashMap::new(),
                cwd: None,
            };
            match connect.spawn(&mut commands, name.to_owned(), spec) {
                Ok(()) => notices.write(Notice::info(
                    agent,
                    format!("Connecting MCP server `{name}`."),
                )),
                Err(why) => notices.write(Notice::error(agent, why)),
            };
        }
        (Some("remove"), Some(name)) => match find(name) {
            Some(server) => {
                commands.entity(server).despawn();
                notices.write(Notice::info(agent, format!("Removed MCP server `{name}`.")));
            }
            None => {
                notices.write(Notice::error(agent, format!("No MCP server `{name}`.")));
            }
        },
        _ => {
            let usage = "Use /mcp, /mcp add <name> <command…> or /mcp remove <name>.";
            notices.write(Notice::error(agent, usage));
        }
    }
}
