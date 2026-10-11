//! Plugin corpus: 19. MCP client (spec `19-mcp-client.md`).
//!
//! MCP servers named in `RIG_HOME/mcp.json` and `<project>/.rig/mcp.json`
//! (the common `mcpServers` format, stdio or streamable HTTP) connect in the
//! background; each server's tools are registered as `<server>__<tool>` as
//! the server lists them, and removed when it disconnects. A server is an
//! entity with an [`McpStatus`]; its tools are entities [`McpToolOf`] it,
//! with the tool's [`McpToolHints`] for approval plugins. `/mcp` shows the
//! servers, `/mcp add <name> <command…>` and `/mcp remove <name>` change
//! them for the session.

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;

use async_channel::{Receiver, Sender};
use bevy_log::warn;
use bevy_tasks::{IoTaskPool, TaskPool};
use rig_core::message::ToolName;
use rig_core::tool::{DynamicTool, ToolOutput as Answer};
use rig_harness::prelude::*;
use rig_rmcp::{McpTool, mcp_result_output};
use rmcp::model::{ClientInfo, Tool as McpDefinition};
use rmcp::service::{NotificationContext, RoleClient, RunningService, ServerSink};
use rmcp::transport::StreamableHttpClientTransport;
use rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig;
use rmcp::{ClientHandler, ServiceExt};
use serde::Deserialize;
use tokio::runtime::Runtime;
use tokio::task::AbortHandle;

#[cfg(test)]
mod tests;

/// The first wait before a lost server is connected again; it doubles up
/// to [`MAX_RETRY`] and starts over once the server answers.
const FIRST_RETRY: Duration = Duration::from_secs(1);
/// The longest wait between two attempts.
const MAX_RETRY: Duration = Duration::from_secs(60);

/// Connects the configured MCP servers and offers their tools.
#[derive(Default)]
pub struct McpClientPlugin;

impl Plugin for McpClientPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<McpConfig>()
            .register_type::<McpStatus>()
            .register_type::<McpToolHints>()
            .add_systems(Startup, connect_configured)
            .add_observer(on_event)
            .add_command(
                "mcp",
                "MCP servers; `/mcp add <name> <command…>`, `/mcp remove <name>`",
                mcp_command,
            );
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .thread_name("mcp")
            .enable_all()
            .build();
        match runtime {
            Ok(runtime) => {
                app.insert_resource(McpRuntime(runtime));
            }
            Err(error) => warn!("MCP servers stay off: no tokio runtime: {error}"),
        }
    }
}

/// Where the servers are configured, and how long a lost one's first retry
/// waits. Insert it before the plugin to read other files.
#[derive(Resource, Clone, Debug)]
pub struct McpConfig {
    /// `mcp.json` files, read in order; a later file's server replaces an
    /// earlier one of the same name.
    pub files: Vec<PathBuf>,
    /// The first wait before a lost server is connected again.
    pub first_retry: Duration,
}

impl Default for McpConfig {
    fn default() -> Self {
        let mut files = vec![Home::from_env().root().join("mcp.json")];
        if let Ok(project) = std::env::current_dir() {
            files.push(project.join(".rig").join("mcp.json"));
        }
        Self {
            files,
            first_retry: FIRST_RETRY,
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
    /// A child process speaking MCP on its stdin and stdout.
    Stdio {
        /// The program.
        command: String,
        /// Its arguments.
        #[serde(default)]
        args: Vec<String>,
        /// Variables added to its environment.
        #[serde(default)]
        env: HashMap<String, String>,
        /// Its working directory.
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

/// The tokio runtime the MCP clients run on.
#[derive(Resource)]
struct McpRuntime(Runtime);

/// A configured server, on its entity, named `mcp:<name>`.
#[derive(Component)]
pub struct McpServer {
    /// The name its tools are prefixed with.
    pub name: String,
    /// How it is reached.
    pub spec: ServerSpec,
    events: Receiver<ServerEvent>,
}

/// Where a server is, for `/mcp` and views.
#[derive(Component, Reflect, Clone, Debug, Default)]
#[reflect(Component, Clone, Debug, Default)]
pub struct McpStatus {
    /// `connecting`, `connected` or `retrying`.
    pub state: String,
    /// How many tools it offers.
    pub tools: usize,
    /// Why it was last lost, or failed to connect.
    pub last_error: Option<String>,
    /// The process of a stdio server, while it runs.
    pub pid: Option<u32>,
}

/// A tool of the server it names.
#[derive(Component, Debug)]
#[relationship(relationship_target = McpTools)]
pub struct McpToolOf(pub Entity);

/// A server's tools, despawned with it.
#[derive(Component, Debug)]
#[relationship_target(relationship = McpToolOf, linked_spawn)]
pub struct McpTools(Vec<Entity>);

/// The annotations an MCP tool came with, on its tool entity, for approval
/// plugins. A read-only tool also has `Footprint::ReadOnly`.
#[derive(Component, Reflect, Clone, Copy, Debug, Default)]
#[reflect(Component, Clone, Debug, Default)]
pub struct McpToolHints {
    /// `readOnlyHint`.
    pub read_only: bool,
    /// `destructiveHint`; MCP's default is `true`.
    pub destructive: bool,
}

/// Stops a server's connection task when its entity goes.
#[derive(Component)]
struct Supervisor(AbortHandle);

impl Drop for Supervisor {
    fn drop(&mut self) {
        self.0.abort();
    }
}

/// What a server's connection task tells the app.
enum ServerEvent {
    Connecting,
    Started(u32),
    Listed(ServerSink, Vec<McpDefinition>),
    Lost(String),
}

/// The next event of a server, once it came; `None` once its task ended.
struct Arrived(Option<ServerEvent>);

/// Waits for the next event on `events`.
fn listen(events: &Receiver<ServerEvent>, wake: &Wake) -> Running {
    let events = events.clone();
    let pool = IoTaskPool::get_or_init(TaskPool::default);
    Running::spawn(pool, wake, async move { Arrived(events.recv().await.ok()) })
}

/// Spawns the server `name` and starts connecting it.
fn spawn_server(
    commands: &mut Commands,
    runtime: &McpRuntime,
    config: &McpConfig,
    wake: &Wake,
    name: String,
    spec: ServerSpec,
) {
    let (sender, events) = async_channel::unbounded();
    let task = runtime.0.spawn(supervise(
        name.clone(),
        spec.clone(),
        config.first_retry,
        sender,
    ));
    commands.spawn((
        Name::new(format!("mcp:{name}")),
        McpStatus {
            state: "connecting".to_owned(),
            ..McpStatus::default()
        },
        Supervisor(task.abort_handle()),
        listen(&events, wake),
        McpServer { name, spec, events },
    ));
}

/// Connects every server of the config files.
fn connect_configured(
    config: Res<McpConfig>,
    runtime: Option<Res<McpRuntime>>,
    wake: Res<Wake>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Some(runtime) = runtime else {
        return;
    };
    let mut servers = BTreeMap::new();
    for file in &config.files {
        let Ok(text) = std::fs::read_to_string(file) else {
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
        spawn_server(&mut commands, &runtime, &config, &wake, name, spec);
    }
}

/// Connects `name` until it is removed: lists its tools, waits for it to go,
/// then waits before the next attempt.
async fn supervise(name: String, spec: ServerSpec, first: Duration, events: Sender<ServerEvent>) {
    let mut wait = first;
    loop {
        events.send(ServerEvent::Connecting).await.ok();
        let (lost, answered) = match run_once(&spec, &events).await {
            Ok(()) => (format!("server `{name}` disconnected"), true),
            Err(error) => (error, false),
        };
        if answered {
            wait = first;
        }
        if events.send(ServerEvent::Lost(lost)).await.is_err() {
            return;
        }
        tokio::time::sleep(wait).await;
        wait = (wait * 2).min(MAX_RETRY.max(first));
    }
}

/// One connection to the server: `Ok` once it closed after answering.
async fn run_once(spec: &ServerSpec, events: &Sender<ServerEvent>) -> Result<(), String> {
    let client = Listener(events.clone());
    match spec {
        ServerSpec::Stdio {
            command,
            args,
            env,
            cwd,
        } => {
            let mut process = tokio::process::Command::new(command);
            process
                .args(args)
                .envs(env)
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::null())
                .kill_on_drop(true);
            if let Some(cwd) = cwd {
                process.current_dir(cwd);
            }
            let mut child = process
                .spawn()
                .map_err(|error| format!("could not start `{command}`: {error}"))?;
            if let Some(pid) = child.id() {
                events.send(ServerEvent::Started(pid)).await.ok();
            }
            let (Some(stdout), Some(stdin)) = (child.stdout.take(), child.stdin.take()) else {
                return Err(format!("`{command}` has no pipes"));
            };
            let service = client
                .serve((stdout, stdin))
                .await
                .map_err(|error| error.to_string())?;
            serve(service, events).await
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
            let config =
                StreamableHttpClientTransportConfig::with_uri(url.as_str()).custom_headers(sent);
            let transport = StreamableHttpClientTransport::from_config(config);
            let service = client
                .serve(transport)
                .await
                .map_err(|error| error.to_string())?;
            serve(service, events).await
        }
    }
}

/// Lists a connected server's tools, then waits for it to close.
async fn serve(
    service: RunningService<RoleClient, Listener>,
    events: &Sender<ServerEvent>,
) -> Result<(), String> {
    let tools = service
        .peer()
        .list_all_tools()
        .await
        .map_err(|error| error.to_string())?;
    let sink = service.peer().clone();
    events.send(ServerEvent::Listed(sink, tools)).await.ok();
    service.waiting().await.ok();
    Ok(())
}

/// The MCP client of one connection: lists the tools again when the server
/// says they changed.
struct Listener(Sender<ServerEvent>);

impl ClientHandler for Listener {
    fn get_info(&self) -> ClientInfo {
        ClientInfo::default()
    }

    async fn on_tool_list_changed(&self, context: NotificationContext<RoleClient>) {
        match context.peer.list_all_tools().await {
            Ok(tools) => {
                let listed = ServerEvent::Listed(context.peer.clone(), tools);
                self.0.send(listed).await.ok();
            }
            Err(error) => warn!("MCP tools could not be listed again: {error}"),
        }
    }
}

/// Applies a server's event and waits for the next one.
fn on_event(
    arrived: On<Insert<Done<Arrived>>>,
    mut servers: Query<(&McpServer, &mut McpStatus, &mut Done<Arrived>)>,
    runtime: Option<Res<McpRuntime>>,
    wake: Res<Wake>,
    mut commands: Commands,
) {
    let server = arrived.entity;
    let Ok((spec, mut status, mut done)) = servers.get_mut(server) else {
        return;
    };
    let Some(event) = done.0.0.take() else {
        commands.entity(server).remove::<Done<Arrived>>();
        return;
    };
    commands
        .entity(server)
        .remove::<Done<Arrived>>()
        .insert(listen(&spec.events, &wake));
    match event {
        ServerEvent::Connecting => status.state = "connecting".to_owned(),
        ServerEvent::Started(pid) => status.pid = Some(pid),
        ServerEvent::Lost(why) => {
            status.state = "retrying".to_owned();
            status.tools = 0;
            status.pid = None;
            status.last_error = Some(why);
            commands.entity(server).despawn_related::<McpTools>();
        }
        ServerEvent::Listed(sink, tools) => {
            let Some(runtime) = runtime else {
                return;
            };
            status.state = "connected".to_owned();
            let handle = runtime.0.handle().clone();
            let name = spec.name.clone();
            commands.queue(move |world: &mut World| {
                register(world, server, &name, &handle, &sink, tools);
            });
        }
    }
}

/// Replaces the tools of `server` with `tools`.
fn register(
    world: &mut World,
    server: Entity,
    name: &str,
    runtime: &tokio::runtime::Handle,
    sink: &ServerSink,
    tools: Vec<McpDefinition>,
) {
    let Ok(mut entity) = world.get_entity_mut(server) else {
        return;
    };
    entity.despawn_related::<McpTools>();
    let mut count = 0;
    for definition in tools {
        let hints = definition.annotations.as_ref();
        let hints = McpToolHints {
            read_only: hints.and_then(|hints| hints.read_only_hint) == Some(true),
            destructive: hints.and_then(|hints| hints.destructive_hint) != Some(false),
        };
        let Some((definition, handler)) = tool(name, definition, sink, runtime) else {
            continue;
        };
        let footprint = if hints.read_only {
            Footprint::ReadOnly
        } else {
            Footprint::Exclusive
        };
        let options = ToolOptions {
            rules: &[],
            footprint,
        };
        if let Some(tool) = world.spawn_tool(definition, handler, options) {
            world.entity_mut(tool).insert((McpToolOf(server), hints));
            count += 1;
        }
    }
    if let Some(mut status) = world.get_mut::<McpStatus>(server) {
        status.tools = count;
    }
}

/// The definition `<server>__<tool>` and the handler of an MCP tool: the
/// call runs on the MCP runtime, and one the server's loss cut short says
/// so.
fn tool(
    server: &str,
    definition: McpDefinition,
    sink: &ServerSink,
    runtime: &tokio::runtime::Handle,
) -> Option<(
    rig_core::completion::ToolDefinition,
    rig_core::serve::ErasedHandler,
)> {
    let name = ToolName::new(format!("{server}__{}", definition.name)).ok()?;
    let description = definition
        .description
        .as_deref()
        .or(definition.title.as_deref())
        .unwrap_or_default()
        .to_owned();
    let parameters = definition.schema_as_json_value();
    let mcp = Arc::new(McpTool::from_mcp_server(definition, sink.clone()));
    let (sink, runtime, server) = (sink.clone(), runtime.clone(), server.to_owned());
    let dynamic = DynamicTool::new(name, description, parameters, move |args| {
        let (mcp, sink, runtime, server) =
            (mcp.clone(), sink.clone(), runtime.clone(), server.clone());
        Box::pin(async move {
            let disconnected =
                || ToolExecutionError::other(format!("server `{server}` disconnected"));
            if sink.is_transport_closed() {
                return Err(disconnected());
            }
            let call = runtime.spawn(async move { mcp.execute_mcp(args.to_string(), None).await });
            let result = match call.await {
                Ok(Ok(result)) => result,
                Ok(Err(_)) | Err(_) if sink.is_transport_closed() => return Err(disconnected()),
                Ok(Err(error)) => return Err(error),
                Err(_) => return Err(disconnected()),
            };
            let output: Answer = mcp_result_output(&result)?;
            if result.is_error == Some(true) {
                let error = ToolExecutionError::other("the MCP tool reported an error");
                return Err(error.with_model_output(output));
            }
            Ok(output)
        })
    });
    let (definition, handler, _) = dynamic.into_parts();
    Some((definition, handler))
}

/// `/mcp`, `/mcp add <name> <command…>`, `/mcp remove <name>`.
fn mcp_command(
    In(args): In<CommandArgs>,
    servers: Query<(Entity, &McpServer, &McpStatus)>,
    runtime: Option<Res<McpRuntime>>,
    config: Res<McpConfig>,
    wake: Res<Wake>,
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
                        "{}: {}, {} tools, last error: {error}",
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
            let Some(runtime) = runtime else {
                notices.write(Notice::error(
                    agent,
                    "MCP is off: no tokio runtime started.",
                ));
                return;
            };
            let spec = ServerSpec::Stdio {
                command: command.to_owned(),
                args: words.map(str::to_owned).collect(),
                env: HashMap::new(),
                cwd: None,
            };
            spawn_server(
                &mut commands,
                &runtime,
                &config,
                &wake,
                name.to_owned(),
                spec,
            );
            notices.write(Notice::info(
                agent,
                format!("Connecting MCP server `{name}`."),
            ));
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
