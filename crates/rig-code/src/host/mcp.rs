//! MCP servers: each server of `RIG_HOME/mcp.json` is an entity, and each
//! tool it lists is a tool entity like any other, named
//! `mcp__<server>__<tool>`, whose calls go through the one recorded
//! dispatch path and the approval gate. rig-rmcp's
//! [`McpClientHandler`] connects, lists the tools and lists them again
//! when the server says they changed; it hands each list to this module
//! through a channel, and the tool entities follow it. The servers run on
//! a small tokio runtime of their own, which rmcp needs; a tool call's
//! task on Bevy's pool awaits its work there.
//!
//! The file takes the `mcpServers` shape other agents use:
//!
//! ```json
//! {
//!   "mcpServers": {
//!     "github": {
//!       "command": "github-mcp-server",
//!       "args": ["stdio"],
//!       "env": { "GITHUB_PERSONAL_ACCESS_TOKEN": "…" }
//!     },
//!     "docs": { "url": "https://example.com/mcp", "headers": { "Authorization": "Bearer …" } },
//!     "off": { "command": "some-server", "disabled": true, "timeout": 60 }
//!   }
//! }
//! ```
//!
//! A server's standard error goes to the session log, never to the
//! terminal. Only the global file is read: a project's own `.mcp.json`
//! would start commands from whatever repository the agent is opened in.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::fs;
use std::io::ErrorKind;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_log::info;
use crossbeam_channel::{Receiver, Sender};
use rig::code_protocol::Home;
use rig_core::effect::{EffectKind, FamilyDescriptor, HandlerDescriptor, family, tool_key};
use rig_core::error::{ErrorKind as ReportKind, ErrorReport};
use rig_core::serve::{Dispatch, ErasedHandler, Reply, Serve};
use rig_core::tool::{DynamicTool, ManagedToolSink, ManagedToolToken};
use rig_rmcp::{DEFAULT_MCP_TOOL_TIMEOUT, McpClientHandler};
use rmcp::model::{ClientCapabilities, ClientInfo, Implementation};
use rmcp::service::QuitReason;
use rmcp::transport::StreamableHttpClientTransport;
use rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig;
use serde::Deserialize;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::runtime::{Handle, Runtime};
use tokio::task::AbortHandle;

use crate::core::agent::Notice;
use crate::core::calls::Wake;
use crate::core::commands::{AppCommandsExt, CommandArgs};
use crate::core::effects::Effects;
use crate::core::tools::{Footprint, ToolOptions, register_tool};
use crate::host::process;

/// The longest tool name providers take.
const MAX_TOOL_NAME: usize = 64;
/// Lines of a server's standard error kept for the notice when it fails.
const STDERR_LINES: usize = 5;

/// Starts the servers of `mcp.json`, keeps their tools as tool entities,
/// and adds `/mcp`.
pub struct McpPlugin;

impl Plugin for McpPlugin {
    fn build(&self, app: &mut App) {
        app.add_command(
            "mcp",
            "List the MCP servers and their tools; /mcp restart <server> starts one again",
            mcp_command,
        );
        let servers: Vec<(String, ServerConfig)> = match read_config() {
            Ok(servers) => servers
                .into_iter()
                .filter(|(_, config)| !config.disabled)
                .collect(),
            Err(why) => {
                app.world_mut().write_message(Notice::error(None, why));
                return;
            }
        };
        if servers.is_empty() {
            return;
        }
        let runtime = match tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .thread_name("rig-code-mcp")
            .enable_all()
            .build()
        {
            Ok(runtime) => runtime,
            Err(failure) => {
                app.world_mut().write_message(Notice::error(
                    None,
                    format!("MCP servers not started: no runtime for them ({failure})."),
                ));
                return;
            }
        };
        for (name, config) in servers {
            app.world_mut().spawn((
                Name::new(format!("mcp:{name}")),
                McpServer {
                    name,
                    config,
                    generation: 0,
                },
                McpStatus::Connecting,
            ));
        }
        let (sender, events) = crossbeam_channel::unbounded();
        app.insert_resource(McpRuntime {
            runtime: Some(runtime),
            sender,
            events,
        })
        .add_systems(Update, (start_servers, receive).chain());
    }
}

/// `mcp.json`.
#[derive(Deserialize)]
struct ConfigFile {
    #[serde(rename = "mcpServers", default)]
    servers: BTreeMap<String, ServerConfig>,
}

/// One server of `mcp.json`: a command to run, speaking MCP on its
/// standard input and output, or the URL of a streamable HTTP server.
#[derive(Deserialize, Clone, Debug)]
pub struct ServerConfig {
    /// The program to run.
    command: Option<String>,
    /// Its arguments.
    #[serde(default)]
    args: Vec<String>,
    /// Variables added to its environment.
    #[serde(default)]
    env: BTreeMap<String, String>,
    /// The directory it runs in; the agent's own by default.
    cwd: Option<PathBuf>,
    /// The URL of an HTTP server.
    url: Option<String>,
    /// Headers sent with every request to it.
    #[serde(default)]
    headers: BTreeMap<String, String>,
    /// Listed but not started.
    #[serde(default)]
    disabled: bool,
    /// Seconds a tool call may take; five minutes by default.
    timeout: Option<u64>,
}

impl ServerConfig {
    /// How the server is reached, as `/mcp` shows it.
    fn label(&self) -> String {
        match (&self.command, &self.url) {
            (Some(command), _) => format!("`{command}`"),
            (None, Some(url)) => url.clone(),
            (None, None) => "nothing to start".to_owned(),
        }
    }

    fn timeout(&self) -> Duration {
        self.timeout
            .map_or(DEFAULT_MCP_TOOL_TIMEOUT, Duration::from_secs)
    }
}

/// The servers of `RIG_HOME/mcp.json`; none without the file.
fn read_config() -> Result<Vec<(String, ServerConfig)>, String> {
    let path = Home::from_env().mcp();
    let text = match fs::read_to_string(&path) {
        Ok(text) => text,
        Err(failure) if failure.kind() == ErrorKind::NotFound => return Ok(Vec::new()),
        Err(failure) => {
            return Err(format!(
                "Could not read {} ({failure}); no MCP server started.",
                path.display()
            ));
        }
    };
    serde_json::from_str::<ConfigFile>(&text)
        .map(|file| file.servers.into_iter().collect())
        .map_err(|failure| {
            format!(
                "{} does not load ({failure}); no MCP server started.",
                path.display()
            )
        })
}

/// An MCP server of `mcp.json`. Its tools are [`McpToolOf`] it, so
/// despawning it removes them; its connection is a task that despawning
/// it, or restarting it, stops.
#[derive(Component)]
pub struct McpServer {
    /// Its name in `mcp.json`.
    pub name: String,
    config: ServerConfig,
    /// Which start of the server events are from; an event of an earlier
    /// start is ignored.
    generation: u64,
}

/// Where an MCP server is.
#[derive(Component, Clone, Debug, PartialEq, Eq)]
pub enum McpStatus {
    /// Starting, or listing its tools.
    Connecting,
    /// Serving this many tools.
    Ready(usize),
    /// It did not start, for this reason.
    Failed(String),
    /// It stopped, for this reason; its tools are gone.
    Closed(String),
}

/// A tool of the MCP server it names.
#[derive(Component, Debug)]
#[relationship(relationship_target = McpTools)]
pub struct McpToolOf(pub Entity);

/// The tools of an MCP server.
#[derive(Component, Debug)]
#[relationship_target(relationship = McpToolOf, linked_spawn)]
pub struct McpTools(Vec<Entity>);

/// A running server's connection task, stopped when this goes away.
#[derive(Component)]
struct ServerTask(AbortHandle);

impl Drop for ServerTask {
    fn drop(&mut self) {
        self.0.abort();
    }
}

/// The tokio runtime the servers run on, and the channel their news comes
/// back through. Dropping it at exit stops every server.
#[derive(Resource)]
struct McpRuntime {
    runtime: Option<Runtime>,
    sender: Sender<Event>,
    events: Receiver<Event>,
}

impl McpRuntime {
    fn handle(&self) -> Option<&Handle> {
        self.runtime.as_ref().map(Runtime::handle)
    }
}

impl Drop for McpRuntime {
    fn drop(&mut self) {
        if let Some(runtime) = self.runtime.take() {
            runtime.shutdown_background();
        }
    }
}

/// News from a server's task, for the start `generation` of `server`.
struct Event {
    server: Entity,
    generation: u64,
    news: News,
}

enum News {
    /// The server's tools, first or after it said they changed.
    Tools(Vec<DynamicTool>),
    /// It did not start.
    Failed(String),
    /// It stopped.
    Closed(String),
}

/// Starts each server that is not running: at startup, and after
/// `/mcp restart`.
fn start_servers(
    mut servers: Query<(Entity, &mut McpServer), Without<ServerTask>>,
    runtime: Res<McpRuntime>,
    wake: Res<Wake>,
    mut commands: Commands,
) {
    let Some(handle) = runtime.handle() else {
        return;
    };
    for (entity, mut server) in &mut servers {
        server.generation += 1;
        let sender = Sender::clone(&runtime.sender);
        let report = Report {
            server: entity,
            generation: server.generation,
            sender,
            wake: wake.clone(),
        };
        let task = handle.spawn(run(
            server.name.clone(),
            server.config.clone(),
            handle.clone(),
            report,
        ));
        commands
            .entity(entity)
            .insert((ServerTask(task.abort_handle()), McpStatus::Connecting));
    }
}

/// Where a server's task sends its news.
#[derive(Clone)]
struct Report {
    server: Entity,
    generation: u64,
    sender: Sender<Event>,
    wake: Wake,
}

impl Report {
    fn send(&self, news: News) {
        let event = Event {
            server: self.server,
            generation: self.generation,
            news,
        };
        if self.sender.send(event).is_ok() {
            self.wake.wake();
        }
    }
}

/// Connects to the server, serves it until it closes, and says how that
/// ended.
async fn run(name: String, config: ServerConfig, runtime: Handle, report: Report) {
    let handler = McpClientHandler::new(
        client_info(),
        Sink {
            report: report.clone(),
        },
    )
    .with_timeout(config.timeout());
    let news = match connect(&name, &config, handler, runtime).await {
        Ok(why) => News::Closed(why),
        Err(why) => News::Failed(why),
    };
    report.send(news);
}

fn client_info() -> ClientInfo {
    ClientInfo::new(
        ClientCapabilities::default(),
        Implementation::new("rig-code", env!("CARGO_PKG_VERSION")),
    )
}

/// Serves the server until it closes; `Ok` says why it closed, `Err` why
/// it did not start.
async fn connect(
    name: &str,
    config: &ServerConfig,
    handler: McpClientHandler<Sink>,
    runtime: Handle,
) -> Result<String, String> {
    match (&config.command, &config.url) {
        (Some(command), _) => {
            let mut server = Process::start(name, command, config, &runtime)?;
            let (Some(stdout), Some(stdin)) =
                (server.child.stdout.take(), server.child.stdin.take())
            else {
                return Err("its standard input and output are not connected".to_owned());
            };
            let service = match handler.connect((stdout, stdin)).await {
                Ok(service) => service,
                Err(failure) => return Err(server.with_stderr(failure.to_string())),
            };
            let ended = service.waiting().await;
            Ok(server.with_stderr(quit_reason(ended)))
        }
        (None, Some(url)) => {
            let headers = config
                .headers
                .iter()
                .map(|(name, value)| {
                    Ok((
                        http::HeaderName::try_from(name.as_str())
                            .map_err(|failure| format!("header `{name}`: {failure}"))?,
                        http::HeaderValue::try_from(value.as_str())
                            .map_err(|failure| format!("header `{name}`: {failure}"))?,
                    ))
                })
                .collect::<Result<HashMap<_, _>, String>>()?;
            let transport = StreamableHttpClientTransport::from_config(
                StreamableHttpClientTransportConfig::with_uri(url.as_str()).custom_headers(headers),
            );
            let service = handler
                .connect(transport)
                .await
                .map_err(|failure| failure.to_string())?;
            Ok(quit_reason(service.waiting().await))
        }
        (None, None) => Err("it has neither a `command` nor a `url`".to_owned()),
    }
}

fn quit_reason(ended: Result<QuitReason, tokio::task::JoinError>) -> String {
    match ended {
        Ok(QuitReason::Closed) => "the connection closed".to_owned(),
        Ok(QuitReason::Cancelled) => "it was stopped".to_owned(),
        Ok(QuitReason::JoinError(failure)) | Err(failure) => format!("it crashed ({failure})"),
        Ok(_) => "it stopped".to_owned(),
    }
}

/// A server's process, in a process group of its own that goes when this
/// does, with its standard error logged.
struct Process {
    child: tokio::process::Child,
    stderr: Arc<Mutex<VecDeque<String>>>,
}

impl Process {
    fn start(
        name: &str,
        command: &str,
        config: &ServerConfig,
        runtime: &Handle,
    ) -> Result<Self, String> {
        let mut std_command = std::process::Command::new(command);
        std_command
            .args(&config.args)
            .envs(&config.env)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        if let Some(cwd) = &config.cwd {
            std_command.current_dir(cwd);
        }
        process::detach(&mut std_command);
        let mut command = tokio::process::Command::from(std_command);
        command.kill_on_drop(true);
        let mut child = command
            .spawn()
            .map_err(|failure| format!("`{command:?}` did not start: {failure}"))?;
        let stderr = Arc::new(Mutex::new(VecDeque::new()));
        if let Some(output) = child.stderr.take() {
            let kept = Arc::clone(&stderr);
            let name = name.to_owned();
            runtime.spawn(async move {
                let mut lines = BufReader::new(output).lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    info!(target: "mcp", server = %name, "{line}");
                    let mut kept = kept.lock().unwrap_or_else(PoisonError::into_inner);
                    if kept.len() == STDERR_LINES {
                        kept.pop_front();
                    }
                    kept.push_back(line);
                }
            });
        }
        Ok(Self { child, stderr })
    }

    /// `why`, with the last lines the server wrote to its standard error.
    fn with_stderr(&self, why: String) -> String {
        let kept = self.stderr.lock().unwrap_or_else(PoisonError::into_inner);
        if kept.is_empty() {
            return why;
        }
        let lines: Vec<&str> = kept.iter().map(String::as_str).collect();
        format!("{why}; it said: {}", lines.join(" / "))
    }
}

impl Drop for Process {
    fn drop(&mut self) {
        #[cfg(unix)]
        if let Some(leader) = self.child.id() {
            process::kill_group_of(leader);
        }
        self.child.start_kill().ok();
    }
}

/// Where rig-rmcp's handler puts a server's tools: every list it installs
/// goes to the app whole, which replaces the server's tool entities with
/// it. The handler is the server's only source of tools, so each list
/// simply wins.
struct Sink {
    report: Report,
}

impl ManagedToolSink for Sink {
    fn add_managed_tools(&self, tools: Vec<DynamicTool>) -> HashMap<String, ManagedToolToken> {
        self.reconcile_managed_tools(HashMap::new(), tools)
    }

    fn reconcile_managed_tools(
        &self,
        _expected: HashMap<String, ManagedToolToken>,
        tools: Vec<DynamicTool>,
    ) -> HashMap<String, ManagedToolToken> {
        let tools: Vec<DynamicTool> = tools.into_iter().filter(DynamicTool::is_live).collect();
        let tokens = tools
            .iter()
            .map(|tool| (tool.name().to_string(), ManagedToolToken::new()))
            .collect();
        self.report.send(News::Tools(tools));
        tokens
    }
}

/// Applies the servers' news: their tool lists, failures and ends.
fn receive(
    runtime: Res<McpRuntime>,
    mut servers: Query<(&McpServer, &mut McpStatus)>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Some(handle) = runtime.handle() else {
        return;
    };
    for Event {
        server,
        generation,
        news,
    } in runtime.events.try_iter()
    {
        let Ok((config, mut status)) = servers.get_mut(server) else {
            continue;
        };
        if config.generation != generation {
            continue;
        }
        let name = config.name.clone();
        match news {
            News::Tools(tools) => {
                if *status == McpStatus::Connecting {
                    notices.write(Notice::info(
                        None,
                        format!(
                            "MCP server `{name}`: {} tool{}.",
                            tools.len(),
                            if tools.len() == 1 { "" } else { "s" }
                        ),
                    ));
                } else {
                    info!(server = %name, tools = tools.len(), "MCP tools changed");
                }
                *status = McpStatus::Ready(tools.len());
                commands.entity(server).despawn_related::<McpTools>();
                let handle = handle.clone();
                commands.queue(move |world: &mut World| {
                    register_tools(world, server, &name, tools, &handle);
                });
            }
            News::Failed(why) => {
                notices.write(Notice::error(
                    None,
                    format!(
                        "MCP server `{name}` did not start: {why}. /mcp restart {name} tries again."
                    ),
                ));
                *status = McpStatus::Failed(why);
            }
            News::Closed(why) => {
                notices.write(Notice::error(
                    None,
                    format!(
                        "MCP server `{name}` stopped: {why}. /mcp restart {name} starts it again."
                    ),
                ));
                commands.entity(server).despawn_related::<McpTools>();
                *status = McpStatus::Closed(why);
            }
        }
    }
}

/// Spawns a tool entity for each of `tools`, the tools of `server`, named
/// `mcp__<server>__<tool>`, and describes them in the effect log's header.
fn register_tools(
    world: &mut World,
    server: Entity,
    name: &str,
    tools: Vec<DynamicTool>,
    runtime: &Handle,
) {
    if world.get_entity(server).is_err() {
        return;
    }
    let mut described = Vec::new();
    for tool in tools {
        let (definition, inner, _) = tool.into_parts();
        let call = McpCall {
            name: tool_name(name, definition.name.as_str()),
            description: definition.description,
            parameters: definition.parameters,
            inner,
            runtime: runtime.clone(),
        };
        let handler = ErasedHandler::new(call.clone());
        // An MCP tool may do anything: its calls run on their own, and a
        // policy that asks first asks for them.
        let options = ToolOptions {
            rules: &[],
            footprint: Footprint::Exclusive,
        };
        if let Some(tool) = register_tool(
            world,
            &call.name,
            call.description,
            call.parameters,
            handler.clone(),
            options,
        ) {
            world.entity_mut(tool).insert(McpToolOf(server));
            described.push(handler.descriptor());
        }
    }
    if let Some(effects) = world.get_resource::<Effects>() {
        effects.describe(described);
    }
}

/// `mcp__<server>__<tool>`, with any character providers refuse in a tool
/// name made `_`, cut to the length they take.
fn tool_name(server: &str, tool: &str) -> String {
    format!("mcp__{server}__{tool}")
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || character == '_' || character == '-' {
                character
            } else {
                '_'
            }
        })
        .take(MAX_TOOL_NAME)
        .collect()
}

/// An MCP tool's handler under the name the agent knows it by. Its work
/// runs on the servers' runtime, which rmcp needs; dropping the call
/// stops it there too.
#[derive(Clone)]
struct McpCall {
    name: String,
    description: String,
    parameters: serde_json::Value,
    inner: ErasedHandler,
    runtime: Handle,
}

impl Serve for McpCall {
    type Family = family::Tool;

    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor {
            key: tool_key(&self.name),
            family: FamilyDescriptor::Tool {
                name: self.name.clone(),
                description: self.description.clone(),
                parameters: self.parameters.clone(),
                embedding: None,
            },
            layers: Vec::new(),
        }
    }

    async fn serve(&self, kind: EffectKind, dispatch: Dispatch) -> Reply {
        let inner = self.inner.clone();
        let task = self
            .runtime
            .spawn(async move { inner.handle(kind, dispatch).await.into_outcome().await });
        let _stop = AbortOnDrop(task.abort_handle());
        Reply::Outcome(task.await.unwrap_or_else(|failure| {
            Err(ErrorReport::new(
                ReportKind::Internal,
                format!("the MCP call did not finish: {failure}"),
            ))
        }))
    }
}

/// Stops a tokio task when dropped.
struct AbortOnDrop(AbortHandle);

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        self.0.abort();
    }
}

/// `/mcp`: lists the servers, or restarts one.
fn mcp_command(
    In(args): In<CommandArgs>,
    servers: Query<(Entity, &McpServer, &McpStatus, Option<&McpTools>)>,
    tools: Query<&crate::core::tools::ToolDef>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let mut words = args.args.split_whitespace();
    match words.next() {
        None => {
            if servers.is_empty() {
                notices.write(Notice::info(
                    args.agent,
                    "No MCP server running: list them in RIG_HOME/mcp.json and /reload.",
                ));
                return;
            }
            let mut lines = Vec::new();
            for (_, server, status, listed) in &servers {
                let status = match status {
                    McpStatus::Connecting => "starting".to_owned(),
                    McpStatus::Ready(_) => "ready".to_owned(),
                    McpStatus::Failed(why) => format!("did not start: {why}"),
                    McpStatus::Closed(why) => format!("stopped: {why}"),
                };
                lines.push(format!(
                    "{} ({}): {status}",
                    server.name,
                    server.config.label()
                ));
                let names: Vec<&str> = listed
                    .into_iter()
                    .flat_map(|listed| listed.iter())
                    .filter_map(|tool| tools.get(tool).ok())
                    .map(|tool| tool.0.name.as_str())
                    .collect();
                if !names.is_empty() {
                    lines.push(format!("  {}", names.join(", ")));
                }
            }
            notices.write(Notice::info(args.agent, lines.join("\n")));
        }
        Some("restart") => {
            let wanted = words.next().unwrap_or_default();
            let Some((entity, server, ..)) =
                servers.iter().find(|(_, server, ..)| server.name == wanted)
            else {
                notices.write(Notice::error(
                    args.agent,
                    format!("No MCP server named `{wanted}`; /mcp lists them."),
                ));
                return;
            };
            // Without its task, the server starts again on the next frame.
            commands.entity(entity).despawn_related::<McpTools>();
            commands.entity(entity).remove::<ServerTask>();
            notices.write(Notice::info(
                args.agent,
                format!("Restarting MCP server `{}`.", server.name),
            ));
        }
        Some(_) => {
            notices.write(Notice::error(
                args.agent,
                "/mcp lists the servers; /mcp restart <server> starts one again.",
            ));
        }
    }
}
