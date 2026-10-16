//! FH-08: language server tools. The agent asks a real language server
//! for diagnostics, definitions, references, hover text and workspace
//! symbols (`lsp_diagnostics`, `lsp_definition`, `lsp_references`,
//! `lsp_hover`, `lsp_symbols`).
//!
//! `RIG_HOME/lsp.toml`, then the project's `.rig/lsp.toml`, map file
//! extensions to server commands (`rs = "rust-analyzer"`). A server starts
//! the first time a call needs it, on a thread of the call's, and stays,
//! shared by every agent working in the same directory ([`LspServers`]).
//! Before each call the server is given every file it has open as it is on
//! disk, so whatever changed a file, results describe it as it is now.
//! While the server reports work in progress, a call waits for it (at
//! most `[lsp] timeout_secs` of `settings.toml`, 60 s), showing it as the
//! call's progress. A crashed server is reported with its last stderr
//! lines and started again by the next call; three crashes in five
//! minutes turn its language off for the session. The servers are shut
//! down before the app exits, which waits for them.

mod client;

use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Instant;

use bevy_ecs::system::SystemParam;
use bevy_tasks::{IoTaskPool, TaskPool};
use client::{Failure, Server, path_of, uri};
use rig_core::tool::ToolOutput;
use rig_harness::prelude::*;
use rig_tools::process::Launcher;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

/// The most items a result lists.
const MAX_ITEMS: usize = 50;
/// The most characters of hover text.
const MAX_HOVER: usize = 4000;
/// The most characters of a code snippet.
const MAX_SNIPPET: usize = 120;
/// How many crashes within [`CRASH_WINDOW`] turn a language off.
const MAX_CRASHES: usize = 3;
/// See [`MAX_CRASHES`].
const CRASH_WINDOW: Duration = Duration::from_secs(5 * 60);
/// How long each server gets for each step of its shutdown.
const SHUTDOWN_GRACE: Duration = Duration::from_secs(2);

/// The `[lsp]` table of `settings.toml`.
#[derive(Resource, Clone, Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LspSettings {
    /// How long a call waits for a server that is still indexing, and at
    /// most for any answer.
    pub timeout_secs: u64,
    /// How long a call waits for a file's diagnostics once the server is
    /// not busy.
    pub settle_secs: u64,
}

impl Default for LspSettings {
    fn default() -> Self {
        Self {
            timeout_secs: 60,
            settle_secs: 10,
        }
    }
}

/// The language tools.
#[derive(Default)]
pub struct LspToolsPlugin;

impl Plugin for LspToolsPlugin {
    fn build(&self, app: &mut App) {
        let options = || ToolOptions {
            footprint: Some(Touches::Reads(Scope::Anything)),
        };
        app.add_setting::<LspSettings>("lsp")
            .init_resource::<LspServers>()
            .init_resource::<ProjectRoot>()
            .init_resource::<RigHome>()
            .init_resource::<Processes>()
            .add_open_tool(
                "lsp_diagnostics",
                "The errors and warnings a language server reports for a file, or for every file it has open when no path is given. After editing typed code, check it with this instead of a full build.",
                options(),
                |called: On<ToolCalled<FileArgs>>, mut jobs: Jobs| {
                    jobs.start(&called, Request::Diagnostics(called.args.path.clone()));
                },
            )
            .add_open_tool(
                "lsp_definition",
                "Where the symbol at a position is defined, from a language server. Use it rather than `search` to find a definition.",
                options(),
                |called: On<ToolCalled<PositionArgs>>, mut jobs: Jobs| {
                    jobs.start(&called, Request::Definition(called.args.clone()));
                },
            )
            .add_open_tool(
                "lsp_references",
                "Every reference to the symbol at a position, from a language server. Before changing a function's signature, find every caller with it.",
                options(),
                |called: On<ToolCalled<PositionArgs>>, mut jobs: Jobs| {
                    jobs.start(&called, Request::References(called.args.clone()));
                },
            )
            .add_open_tool(
                "lsp_hover",
                "The type and documentation of the symbol at a position, from a language server.",
                options(),
                |called: On<ToolCalled<PositionArgs>>, mut jobs: Jobs| {
                    jobs.start(&called, Request::Hover(called.args.clone()));
                },
            )
            .add_open_tool(
                "lsp_symbols",
                "Symbols of the whole workspace whose names match a query, from the running language servers, or the one for `ext`.",
                options(),
                |called: On<ToolCalled<SymbolArgs>>, mut jobs: Jobs| {
                    jobs.start(&called, Request::Symbols(called.args.clone()));
                },
            )
            .add_observer(on_step)
            .add_observer(shut_down)
            .add_observer(give_up)
            .add_systems(AgentPass, hold_exit.in_set(Stage::Gate));
    }
}

/// `lsp_diagnostics`' arguments.
#[derive(Clone, Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FileArgs {
    /// The file; every open file when left out.
    #[serde(default)]
    pub path: Option<String>,
}

/// A position in a file: a line and a column, or a symbol on that line.
#[derive(Clone, Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PositionArgs {
    /// The file.
    pub path: String,
    /// The line, from 1.
    pub line: u32,
    /// The column, from 1, in characters.
    #[serde(default)]
    pub column: Option<u32>,
    /// A symbol on the line, in place of the column: its first occurrence.
    #[serde(default)]
    pub symbol: Option<String>,
}

/// `lsp_symbols`' arguments.
#[derive(Clone, Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SymbolArgs {
    /// What the symbols' names contain.
    pub query: String,
    /// The extension whose server to ask, such as `rs`, starting it if
    /// needed; every running server when left out.
    #[serde(default)]
    pub ext: Option<String>,
}

/// What a call asks.
enum Request {
    Diagnostics(Option<String>),
    Definition(PositionArgs),
    References(PositionArgs),
    Hover(PositionArgs),
    Symbols(SymbolArgs),
}

/// What a call's thread reports, as items of the stream on the call.
pub enum Step {
    /// The server's work in progress, shown on the call.
    Progress(String),
    /// A notice for the user.
    Notice(String),
    /// The call's result.
    Finished(ToolResult),
}

/// On a call this plugin answers: its agent.
#[derive(Component)]
struct LspCall(Entity);

/// The language servers of the session, shared by every agent: one per
/// command and directory, kept running across turns.
#[derive(Resource, Clone, Default)]
pub struct LspServers(Arc<Mutex<HashMap<(PathBuf, String), Arc<Mutex<Slot>>>>>);

/// One command's server in one directory, and its crashes.
#[derive(Default)]
struct Slot {
    server: Option<Arc<Server>>,
    crashes: Vec<Instant>,
    disabled: bool,
}

impl LspServers {
    /// The program name and process id of each running server.
    pub fn running(&self) -> Vec<(String, u32)> {
        self.slots()
            .into_iter()
            .filter_map(|(_, slot)| {
                let slot = slot.lock().unwrap_or_else(PoisonError::into_inner);
                slot.server
                    .as_ref()
                    .map(|server| (server.name.clone(), server.pid()))
            })
            .collect()
    }

    fn slots(&self) -> Vec<((PathBuf, String), Arc<Mutex<Slot>>)> {
        let slots = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        slots
            .iter()
            .map(|(key, slot)| (key.clone(), Arc::clone(slot)))
            .collect()
    }

    fn slot(&self, root: &Path, command: &str) -> Arc<Mutex<Slot>> {
        let mut slots = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        let key = (root.to_path_buf(), command.to_owned());
        Arc::clone(slots.entry(key).or_default())
    }

    /// Every running server, taken out of its slot.
    fn take_all(&self) -> Vec<Arc<Server>> {
        self.slots()
            .into_iter()
            .filter_map(|(_, slot)| {
                slot.lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .server
                    .take()
            })
            .collect()
    }
}

/// What starting a call needs.
#[derive(SystemParam)]
struct Jobs<'w, 's> {
    servers: Res<'w, LspServers>,
    settings: Res<'w, LspSettings>,
    root: Res<'w, ProjectRoot>,
    home: Res<'w, RigHome>,
    processes: Res<'w, Processes>,
    wake: Res<'w, Wake>,
    dirs: Query<'w, 's, &'static WorkDir>,
    calls: Query<'w, 's, &'static mut Decision, ToolCalls>,
    commands: Commands<'w, 's>,
}

impl Jobs<'_, '_> {
    /// Runs `request` for `called`'s call on a thread of its own, whose
    /// steps come back as a stream on the call.
    fn start<A: Send + Sync + 'static>(&mut self, called: &ToolCalled<A>, request: Request) {
        let base = self
            .dirs
            .get(called.agent)
            .map_or_else(|_| self.root.0.clone(), |dir| dir.0.clone());
        let base = std::fs::canonicalize(&base).unwrap_or(base);
        let (steps, received) = async_channel::unbounded();
        let job = Job {
            request,
            base,
            home: self.home.0.root().to_path_buf(),
            launcher: self.processes.launcher(),
            servers: self.servers.clone(),
            timeout: Duration::from_secs(self.settings.timeout_secs),
            settle: Duration::from_secs(self.settings.settle_secs),
            steps,
        };
        let thread = std::thread::Builder::new()
            .name("rig-lsp-call".to_owned())
            .spawn(move || {
                let result = job.run();
                job.steps.send_blocking(Step::Finished(result)).ok();
            });
        match thread {
            Ok(_) => {
                let pool = IoTaskPool::get_or_init(TaskPool::default);
                let running = Running::stream(pool, &self.wake, received);
                self.commands
                    .entity(called.call)
                    .insert((running, LspCall(called.agent)));
            }
            Err(failure) => {
                if let Ok(mut decision) = self.calls.get_mut(called.call) {
                    let error =
                        ToolExecutionError::other(format!("could not start the call: {failure}"));
                    decision.decide(Decider::plugin("fh-lsp-tools"), ToolResult::failed(error));
                }
            }
        }
    }
}

/// Shows a call's progress, passes its notices on, and answers it.
fn on_step(
    step: On<Insert<Done<Step>>>,
    steps: Query<(&Done<Step>, &LspCall)>,
    mut calls: Query<&mut Decision, ToolCalls>,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    let call = step.entity;
    let Ok((Done(step), &LspCall(agent))) = steps.get(call) else {
        return;
    };
    match step {
        Step::Progress(line) => {
            commands.entity(call).insert(ToolProgress(line.clone()));
        }
        Step::Notice(text) => {
            notices.write(Notice::warn(agent, text.clone()));
        }
        Step::Finished(result) => {
            if let Ok(mut decision) = calls.get_mut(call) {
                decision.decide(Decider::plugin("fh-lsp-tools"), result.clone());
            }
        }
    }
}

/// On the hold that keeps the app from exiting while the servers shut
/// down.
#[derive(Component)]
struct ShutdownHold;

/// Holds the first exit while the running servers shut down, at most a
/// few seconds.
fn hold_exit(
    exits: Query<Entity, (With<ExitRequest>, ReadyToStart, PluginEffects)>,
    servers: Res<LspServers>,
    wake: Res<Wake>,
    mut held: Local<bool>,
    mut commands: Commands,
) {
    let Some(exit) = exits.iter().next().filter(|_| !*held) else {
        return;
    };
    *held = true;
    let running = servers.take_all();
    if running.is_empty() {
        return;
    }
    let work = async move {
        blocking(move || {
            std::thread::scope(|scope| {
                for server in &running {
                    scope.spawn(|| server.shutdown(SHUTDOWN_GRACE));
                }
            });
            Ok(())
        })
        .await
        .ok();
    };
    let pool = IoTaskPool::get_or_init(TaskPool::default);
    commands.spawn((
        Name::new("shutting down the language servers"),
        ShutdownHold,
        HoldOf(exit),
        WakeAt::after(SHUTDOWN_GRACE * 3),
        Running::spawn(pool, &wake, work),
    ));
}

/// Lets the exit go once the servers shut down.
fn shut_down(
    done: On<Insert<Done<()>>>,
    holds: Query<(), With<ShutdownHold>>,
    mut commands: Commands,
) {
    if holds.contains(done.entity) {
        commands.entity(done.entity).despawn();
    }
}

/// Lets the exit go when the servers take too long; dropping them kills
/// what is left.
fn give_up(due: On<Add<Due>>, holds: Query<(), With<ShutdownHold>>, mut commands: Commands) {
    if holds.contains(due.entity) {
        commands.entity(due.entity).despawn();
    }
}

/// A call's work, on its own thread.
struct Job {
    request: Request,
    /// The directory the agent works in: relative paths start there, and
    /// its servers are rooted there.
    base: PathBuf,
    /// The `RIG_HOME` root.
    home: PathBuf,
    launcher: Launcher,
    servers: LspServers,
    timeout: Duration,
    settle: Duration,
    steps: async_channel::Sender<Step>,
}

/// A call's result before it is one: text, or a failure.
type Answer = Result<String, ToolExecutionError>;

impl Job {
    fn run(&self) -> ToolResult {
        let answer = match &self.request {
            Request::Diagnostics(path) => self.diagnostics(path.as_deref()),
            Request::Definition(at) => self.locate(at, "textDocument/definition", "definition"),
            Request::References(at) => self.locate(at, "textDocument/references", "reference"),
            Request::Hover(at) => self.hover(at),
            Request::Symbols(args) => self.symbols(args),
        };
        match answer {
            Ok(text) => ToolResult::success(ToolOutput::text(text)),
            Err(error) => ToolResult::failed(error),
        }
    }

    /// The extension-to-command map: `RIG_HOME/lsp.toml`, then the
    /// project's `.rig/lsp.toml` over it.
    fn config(&self) -> Result<HashMap<String, String>, ToolExecutionError> {
        let mut config = HashMap::new();
        for file in [self.home.join("lsp.toml"), self.base.join(".rig/lsp.toml")] {
            let Ok(text) = std::fs::read_to_string(&file) else {
                continue;
            };
            let table = text.parse::<toml::Table>().map_err(|failure| {
                ToolExecutionError::other(format!("{} is not TOML: {failure}", file.display()))
            })?;
            for (extension, command) in table {
                if let Some(command) = command.as_str() {
                    config.insert(
                        extension.trim_start_matches('.').to_owned(),
                        command.to_owned(),
                    );
                }
            }
        }
        Ok(config)
    }

    /// The server for files with `extension`, started if it is not
    /// running; a text for the model when there is none.
    fn server(&self, extension: &str) -> Result<Result<Arc<Server>, String>, ToolExecutionError> {
        let Some(command) = self.config()?.remove(extension) else {
            return Ok(Err(format!(
                "No language server is configured for `.{extension}` files: add `{extension} = \"<command>\"` to RIG_HOME/lsp.toml or the project's .rig/lsp.toml."
            )));
        };
        let slot = self.servers.slot(&self.base, &command);
        let mut slot = slot.lock().unwrap_or_else(PoisonError::into_inner);
        if slot.disabled {
            return Err(ToolExecutionError::other(format!(
                "`{command}` crashed {MAX_CRASHES} times in five minutes: language tools are off for `.{extension}` files for this session."
            )));
        }
        if let Some(server) = &slot.server {
            if server.alive() {
                return Ok(Ok(Arc::clone(server)));
            }
            return Err(self.crashed(&mut slot, extension));
        }
        let server = Server::start(&self.launcher, &command, &self.base, self.timeout)
            .map_err(ToolExecutionError::other)?;
        let server = Arc::new(server);
        slot.server = Some(Arc::clone(&server));
        Ok(Ok(server))
    }

    /// Takes a dead server out of its slot, so the next call starts it
    /// again, counts the crash, and reports it with the server's last
    /// stderr lines.
    fn crashed(&self, slot: &mut Slot, extension: &str) -> ToolExecutionError {
        let Some(server) = slot.server.take() else {
            return ToolExecutionError::other("the language server is gone");
        };
        let mut report = server.crash_report();
        slot.crashes.retain(|crash| crash.elapsed() < CRASH_WINDOW);
        slot.crashes.push(Instant::now());
        if slot.crashes.len() >= MAX_CRASHES {
            slot.disabled = true;
            let notice = format!(
                "`{}` crashed {MAX_CRASHES} times in five minutes: language tools are off for `.{extension}` files for this session.",
                server.name
            );
            self.steps.send_blocking(Step::Notice(notice.clone())).ok();
            report = format!("{report}\n{notice}");
        } else {
            report.push_str("\nThe next call starts it again.");
        }
        ToolExecutionError::provider(report)
    }

    /// The server's slot, to report a crash found in the middle of a call.
    fn crash_of(&self, server: &Server, extension: &str) -> ToolExecutionError {
        let slots = self.servers.slots();
        let slot = slots.iter().find(|(_, slot)| {
            let slot = slot.lock().unwrap_or_else(PoisonError::into_inner);
            slot.server
                .as_ref()
                .is_some_and(|running| std::ptr::eq(Arc::as_ptr(running), server))
        });
        match slot {
            Some((_, slot)) => self.crashed(
                &mut slot.lock().unwrap_or_else(PoisonError::into_inner),
                extension,
            ),
            None => ToolExecutionError::provider(server.crash_report()),
        }
    }

    /// Gives the server every file it has open as it is now, and `path`.
    fn sync(
        &self,
        server: &Server,
        path: Option<&Path>,
        extension: &str,
    ) -> Result<(), ToolExecutionError> {
        let mut paths = server.open_documents();
        paths.extend(path.map(Path::to_path_buf));
        for path in paths {
            if server.sync(&path).is_err() {
                return Err(self.crash_of(server, extension));
            }
        }
        Ok(())
    }

    /// Waits while the server reports work in progress, showing it, at
    /// most the timeout. Whether it is still busy.
    fn wait_until_idle(&self, server: &Server, started: Instant) -> bool {
        let mut shown = None;
        !server.wait(
            started + self.timeout,
            |state| !state.busy(),
            |state| self.show(server, state, &mut shown),
        )
    }

    /// Shows the server's progress on the call when it changed.
    fn show(&self, server: &Server, state: &client::State, shown: &mut Option<String>) {
        let line = state
            .progress_line()
            .map(|line| format!("{}: {line}", server.name));
        if line.is_some() && line != *shown {
            shown.clone_from(&line);
            if let Some(line) = line {
                self.steps.send_blocking(Step::Progress(line)).ok();
            }
        }
    }

    /// A request's answer, a crash or a timeout reported as the call's
    /// failure.
    fn ask(
        &self,
        server: &Server,
        method: &str,
        params: Value,
        extension: &str,
    ) -> Result<Value, ToolExecutionError> {
        server
            .request(method, params, self.timeout)
            .map_err(|failure| match failure {
                Failure::Died => self.crash_of(server, extension),
                Failure::Timeout => ToolExecutionError::timeout(format!(
                    "`{}` did not answer {method} within {} s",
                    server.name,
                    self.timeout.as_secs()
                )),
                Failure::Error(why) => {
                    ToolExecutionError::provider(format!("`{}`: {why}", server.name))
                }
            })
    }

    /// `path` from the agent's directory, and its extension.
    fn file(&self, path: &str) -> (PathBuf, String) {
        let path = rig_tools::fs::resolve(Some(&self.base), path);
        let extension = path
            .extension()
            .map(|ext| ext.to_string_lossy().into_owned())
            .unwrap_or_default();
        (path, extension)
    }

    fn diagnostics(&self, path: Option<&str>) -> Answer {
        let started = Instant::now();
        let targets: Vec<(Arc<Server>, Option<PathBuf>, String)> = match path {
            Some(path) => {
                let (path, extension) = self.file(path);
                if !path.is_file() {
                    return Err(ToolExecutionError::not_found(format!(
                        "{} is not a file",
                        self.show_path(&path)
                    )));
                }
                match self.server(&extension)? {
                    Ok(server) => vec![(server, Some(path), extension)],
                    Err(text) => return Ok(text),
                }
            }
            None => self
                .running()
                .into_iter()
                .map(|server| (server, None, String::new()))
                .collect(),
        };
        if targets.is_empty() {
            return Ok(
                "No language server is running, so no file is open: name a file.".to_owned(),
            );
        }
        let mut text = String::new();
        let mut busy = false;
        for (server, path, extension) in targets {
            self.sync(&server, path.as_deref(), &extension)?;
            busy |= self.wait_until_idle(&server, started);
            let uris: Vec<String> = match &path {
                Some(path) => vec![uri(path)],
                None => server
                    .open_documents()
                    .iter()
                    .map(|path| uri(path))
                    .collect(),
            };
            let mut shown = None;
            let settled = Instant::now() + self.settle;
            let fresh = server.wait(
                started + self.timeout,
                |state| {
                    uris.iter().all(|uri| state.fresh(uri))
                        || (!state.busy() && Instant::now() >= settled)
                },
                |state| self.show(&server, state, &mut shown),
            ) && server.with_state(|state| uris.iter().all(|uri| state.fresh(uri)));
            if !server.alive() {
                return Err(self.crash_of(&server, &extension));
            }
            let found = server.with_state(|state| {
                let mut found = Vec::new();
                for uri in &uris {
                    for diagnostic in state.diagnostics.get(uri).into_iter().flatten() {
                        // 1 is an error, 2 a warning; hints and information are left out.
                        let severity = diagnostic
                            .get("severity")
                            .and_then(Value::as_u64)
                            .unwrap_or(1);
                        if severity <= 2 {
                            found.push((uri.clone(), severity, diagnostic.clone()));
                        }
                    }
                }
                found
            });
            let errors = found
                .iter()
                .filter(|(_, severity, _)| *severity == 1)
                .count();
            let where_ = match &path {
                Some(path) => self.show_path(path),
                None => format!("{} open files", uris.len()),
            };
            if found.is_empty() {
                let _ = writeln!(text, "{}: no errors or warnings in {where_}.", server.name);
            } else {
                let warnings = found.len() - errors;
                let _ = writeln!(
                    text,
                    "{}: {} and {} in {where_}:",
                    server.name,
                    counted(errors, "error"),
                    counted(warnings, "warning")
                );
            }
            let mut items = Vec::new();
            for (uri, severity, diagnostic) in &found {
                let kind = if *severity == 1 { "error" } else { "warning" };
                let message = diagnostic
                    .get("message")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let start = diagnostic
                    .pointer("/range/start")
                    .cloned()
                    .unwrap_or_default();
                let (location, snippet) = self.location(uri, &start);
                items.push(format!("{location}: {kind}: {message}{snippet}"));
            }
            text.push_str(&capped(items));
            if !fresh {
                text.push_str(
                    "\nNote: the server has not reported on every file's latest content yet.",
                );
            }
        }
        if busy {
            text.push_str("\nNote: the server is still indexing; results may be incomplete.");
        }
        Ok(text.trim_end().to_owned())
    }

    /// Definitions or references of the symbol at `at`.
    fn locate(&self, at: &PositionArgs, method: &str, what: &str) -> Answer {
        let started = Instant::now();
        let (path, extension) = self.file(&at.path);
        let server = match self.server(&extension)? {
            Ok(server) => server,
            Err(text) => return Ok(text),
        };
        self.sync(&server, Some(&path), &extension)?;
        let busy = self.wait_until_idle(&server, started);
        let mut params = self.position(&path, at)?;
        if method == "textDocument/references"
            && let Some(params) = params.as_object_mut()
        {
            params.insert("context".to_owned(), json!({ "includeDeclaration": false }));
        }
        let answer = self.ask(&server, method, params, &extension)?;
        let found: Vec<Value> = match answer {
            Value::Array(items) => items,
            Value::Null => Vec::new(),
            item => vec![item],
        };
        let items: Vec<String> = found
            .iter()
            .map(|item| {
                // A `Location`, or a `LocationLink`.
                let uri = item.get("uri").or_else(|| item.get("targetUri"));
                let start = item
                    .pointer("/range/start")
                    .or_else(|| item.pointer("/targetSelectionRange/start"));
                let (location, snippet) = self.location(
                    uri.and_then(Value::as_str).unwrap_or_default(),
                    start.unwrap_or(&Value::Null),
                );
                format!("{location}{snippet}")
            })
            .collect();
        let mut text = match items.len() {
            0 => format!("No {what} found."),
            count => format!("{}:\n", counted(count, what)),
        };
        text.push_str(&capped(items));
        if busy {
            text.push_str("\nNote: the server is still indexing; results may be incomplete.");
        }
        Ok(text.trim_end().to_owned())
    }

    fn hover(&self, at: &PositionArgs) -> Answer {
        let started = Instant::now();
        let (path, extension) = self.file(&at.path);
        let server = match self.server(&extension)? {
            Ok(server) => server,
            Err(text) => return Ok(text),
        };
        self.sync(&server, Some(&path), &extension)?;
        let busy = self.wait_until_idle(&server, started);
        let params = self.position(&path, at)?;
        let answer = self.ask(&server, "textDocument/hover", params, &extension)?;
        let mut text = hover_text(answer.get("contents").unwrap_or(&Value::Null));
        if text.trim().is_empty() {
            text = "Nothing to show at that position.".to_owned();
        }
        if text.chars().count() > MAX_HOVER {
            text = text.chars().take(MAX_HOVER).collect();
            text.push_str("\n… (cut)");
        }
        if busy {
            text.push_str("\nNote: the server is still indexing; results may be incomplete.");
        }
        Ok(text)
    }

    fn symbols(&self, args: &SymbolArgs) -> Answer {
        let started = Instant::now();
        let servers: Vec<(Arc<Server>, String)> = match &args.ext {
            Some(extension) => {
                let extension = extension.trim_start_matches('.');
                match self.server(extension)? {
                    Ok(server) => vec![(server, extension.to_owned())],
                    Err(text) => return Ok(text),
                }
            }
            None => self
                .running()
                .into_iter()
                .map(|server| (server, String::new()))
                .collect(),
        };
        if servers.is_empty() {
            return Ok(
                "No language server is running yet: name one with `ext`, such as `rs`.".to_owned(),
            );
        }
        let mut items = Vec::new();
        let mut busy = false;
        for (server, extension) in servers {
            self.sync(&server, None, &extension)?;
            busy |= self.wait_until_idle(&server, started);
            let answer = self.ask(
                &server,
                "workspace/symbol",
                json!({ "query": args.query }),
                &extension,
            )?;
            for symbol in answer.as_array().into_iter().flatten() {
                let name = symbol
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let kind = symbol
                    .get("kind")
                    .and_then(Value::as_u64)
                    .map_or("symbol", symbol_kind);
                let uri = symbol
                    .pointer("/location/uri")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let start = symbol
                    .pointer("/location/range/start")
                    .cloned()
                    .unwrap_or_default();
                let (location, _) = self.location(uri, &start);
                items.push(format!("{kind} {name}  {location}"));
            }
        }
        let mut text = match items.len() {
            0 => format!("No symbol matches `{}`.", args.query),
            count => format!("{}:\n", counted(count, "symbol")),
        };
        text.push_str(&capped(items));
        if busy {
            text.push_str("\nNote: the server is still indexing; results may be incomplete.");
        }
        Ok(text.trim_end().to_owned())
    }

    /// The running servers rooted in the agent's directory.
    fn running(&self) -> Vec<Arc<Server>> {
        self.servers
            .slots()
            .into_iter()
            .filter(|((root, _), _)| *root == self.base)
            .filter_map(|(_, slot)| {
                slot.lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .server
                    .clone()
            })
            .collect()
    }

    /// The request's `textDocument` and `position` for `at`: a column, or
    /// the first occurrence of a symbol on the line, in UTF-16 units.
    fn position(&self, path: &Path, at: &PositionArgs) -> Result<Value, ToolExecutionError> {
        let text = std::fs::read_to_string(path).map_err(|failure| {
            ToolExecutionError::not_found(format!(
                "could not read {}: {failure}",
                self.show_path(path)
            ))
        })?;
        let line_text = at
            .line
            .checked_sub(1)
            .and_then(|line| text.lines().nth(line as usize))
            .ok_or_else(|| {
                ToolExecutionError::invalid_args(format!(
                    "{} has no line {}",
                    self.show_path(path),
                    at.line
                ))
            })?;
        let column = match (&at.symbol, at.column) {
            (Some(symbol), _) => find_symbol(line_text, symbol).ok_or_else(|| {
                ToolExecutionError::invalid_args(format!(
                    "`{symbol}` is not on line {} of {}",
                    at.line,
                    self.show_path(path)
                ))
            })?,
            (None, Some(column)) => column.saturating_sub(1) as usize,
            (None, None) => {
                return Err(ToolExecutionError::invalid_args(
                    "give a `column` or a `symbol`",
                ));
            }
        };
        let character: usize = line_text.chars().take(column).map(char::len_utf16).sum();
        Ok(json!({
            "textDocument": { "uri": uri(path) },
            "position": { "line": at.line - 1, "character": character },
        }))
    }

    /// `path:line:col` of an LSP position (counted from 1, in characters),
    /// and the line's code as a snippet when the file can be read.
    fn location(&self, uri: &str, start: &Value) -> (String, String) {
        let path = path_of(uri);
        let line = start
            .get("line")
            .and_then(Value::as_u64)
            .unwrap_or_default() as usize;
        let character = start
            .get("character")
            .and_then(Value::as_u64)
            .unwrap_or_default() as usize;
        let code = std::fs::read_to_string(&path)
            .ok()
            .and_then(|text| text.lines().nth(line).map(str::to_owned));
        let column = code.as_deref().map_or(character, |code| {
            let mut units = 0;
            code.chars()
                .take_while(|c| {
                    units += c.len_utf16();
                    units <= character
                })
                .count()
        });
        let location = format!("{}:{}:{}", self.show_path(&path), line + 1, column + 1);
        let snippet = code.map_or_else(String::new, |code| {
            let code: String = code.trim().chars().take(MAX_SNIPPET).collect();
            format!("\n    {code}")
        });
        (location, snippet)
    }

    /// `path` from the agent's directory when it is inside it.
    fn show_path(&self, path: &Path) -> String {
        path.strip_prefix(&self.base)
            .unwrap_or(path)
            .display()
            .to_string()
    }
}

/// The character index of the first whole-word occurrence of `symbol` in
/// `line`, else of any occurrence.
fn find_symbol(line: &str, symbol: &str) -> Option<usize> {
    let word = |c: Option<char>| c.is_some_and(|c| c.is_alphanumeric() || c == '_');
    let whole = line.match_indices(symbol).find(|(at, _)| {
        !word(
            line.get(..*at)
                .and_then(|before| before.chars().next_back()),
        ) && !word(
            line.get(at + symbol.len()..)
                .and_then(|after| after.chars().next()),
        )
    });
    let (at, _) = whole.or_else(|| line.match_indices(symbol).next())?;
    Some(line.get(..at).map_or(0, |before| before.chars().count()))
}

/// `count` and `noun`, made plural unless it is one.
fn counted(count: usize, noun: &str) -> String {
    if count == 1 {
        format!("1 {noun}")
    } else {
        format!("{count} {noun}s")
    }
}

/// At most [`MAX_ITEMS`] of `items`, one per line, and how many were left
/// out.
fn capped(items: Vec<String>) -> String {
    let count = items.len();
    let mut text: String = items
        .into_iter()
        .take(MAX_ITEMS)
        .map(|item| item + "\n")
        .collect();
    if count > MAX_ITEMS {
        let _ = writeln!(text, "… {} more omitted.", count - MAX_ITEMS);
    }
    text
}

/// The text of a hover's contents: a string, a `MarkedString`, a list of
/// them, or `MarkupContent`.
fn hover_text(contents: &Value) -> String {
    match contents {
        Value::String(text) => text.clone(),
        Value::Array(items) => items.iter().map(hover_text).collect::<Vec<_>>().join("\n"),
        Value::Object(_) => contents
            .get("value")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
        _ => String::new(),
    }
}

/// The name of an LSP `SymbolKind`.
fn symbol_kind(kind: u64) -> &'static str {
    match kind {
        1 => "file",
        2 => "module",
        3 => "namespace",
        5 => "class",
        6 => "method",
        8 => "field",
        9 => "constructor",
        10 => "enum",
        11 => "interface",
        12 => "function",
        13 => "variable",
        14 => "constant",
        22 => "enum member",
        23 => "struct",
        26 => "type parameter",
        _ => "symbol",
    }
}

#[cfg(test)]
mod tests;
