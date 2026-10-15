//! Plugin corpus (held-out): H29: Any program as a model (spec `H29-command-provider.md`).
//!
//! Each `[[model]]` of `RIG_HOME/command-models.toml` is a catalog model
//! served by a local program, so `--model`, `/model`, the remembered
//! default, subagents and plugin calls take it like any other:
//!
//! ```toml
//! [[model]]
//! id = "cmd/echo"                  # vendor/name; the vendor is not a built-in one
//! command = ["./echo-model.sh", "--fast"]
//! context_window = 32000           # optional
//! tools = true                     # whether it is offered tools (default true)
//! reasoning = false                # whether it reasons (default false)
//! persistent = false               # one process per request (default), or one kept running
//! ```
//!
//! **Request**: one JSON line on the program's stdin:
//!
//! ```json
//! {"id": 7, "model": "cmd/echo", "system": "…",
//!  "messages": [
//!    {"role": "user", "content": [{"type": "text", "text": "…"},
//!                                 {"type": "tool_result", "id": "…", "name": "read", "content": "…", "is_error": false}]},
//!    {"role": "assistant", "content": [{"type": "text", "text": "…"}, {"type": "reasoning", "text": "…"},
//!                                      {"type": "tool_call", "id": "…", "name": "read", "arguments": {}}]}],
//!  "tools": [{"name": "read", "description": "…", "parameters": {}}],
//!  "options": {"temperature": null, "max_tokens": null, "tool_choice": null, "additional_params": null}}
//! ```
//!
//! Other content (images, documents) is rig-core's own JSON for it.
//!
//! **Reply**: JSON lines on stdout, each streamed to the front as it
//! arrives: `{"text": "…"}`, `{"reasoning": "…"}`, `{"tool_call": {"id",
//! "name", "arguments"}}`, `{"usage": {"input_tokens", "output_tokens", …}}`
//! (rig-core's usage fields; the last one counts), `{"error": "…", "retry":
//! false}` and `{"done": true}`. Empty lines are skipped. A line that is
//! none of these fails the call with its line number and its start; an exit
//! before `done` fails it with the end of stderr; exit code 75 or an error
//! with `"retry": true` is a failure the kernel retries. The cost is 0. A
//! tool call's `id` is not kept: rig names the call, and the next request
//! carries rig's id with its result.
//!
//! **Interrupts**: a call's process is killed. A `persistent` program
//! serves one request at a time and is sent `{"cancel": <id>}` instead,
//! after which its lines up to its `done` or `error` are dropped.

use std::collections::HashMap;
use std::path::Path;
use std::process::Stdio;
use std::sync::Arc;

use futures::StreamExt;
use futures::channel::{mpsc, oneshot};
use futures::future::{Either, select};
use rig_harness::prelude::*;
use rig_harness::rig_core::catalog::{
    Catalog, ModelSpec, Pricing, Provide, ReasoningSupport, Transport,
};
use rig_harness::rig_core::completion::{CompletionRequest, Cost, ProviderCapabilities, Usage};
use rig_harness::rig_core::effect::{
    EffectKind, FamilyDescriptor, HandlerDescriptor, family, model_key,
};
use rig_harness::rig_core::error::{ErrorKind, ErrorReport};
use rig_harness::rig_core::message::{
    AssistantContent, Message as ChatMessage, Origin, ToolResultContent, UserContent,
};
use rig_harness::rig_core::operation::Finish;
use rig_harness::rig_core::providers::registry::{ConnectError, ProviderId};
use rig_harness::rig_core::serve::{Dispatch, ErasedHandler, Reply, Serve, StreamWriter};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader, Lines};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::runtime::Handle;
use tokio::task::JoinHandle;

/// The file under `RIG_HOME`.
pub const FILE: &str = "command-models.toml";

/// The exit code that asks for the request to be tried again
/// (`EX_TEMPFAIL`).
const TRY_AGAIN: i32 = 75;

/// How much of a malformed line a failure quotes, in characters.
const QUOTED: usize = 80;

/// How much of the end of stderr a failure quotes, in bytes.
const STDERR_TAIL: usize = 2000;

/// Adds the models of `RIG_HOME/command-models.toml` (or of a
/// [`CommandModels`] inserted before it) to the catalog.
#[derive(Default)]
pub struct CommandProviderPlugin;

impl Plugin for CommandProviderPlugin {
    fn build(&self, app: &mut App) {
        if !app.world().contains_resource::<CommandModels>() {
            let path = RigHome::of(app.world()).root().join(FILE);
            app.insert_resource(CommandModels::load(&path));
        }
        app.add_systems(Startup, report_problems);
    }

    /// Serves the models once every plugin's `build` ran, so the process
    /// launcher and tokio runtime exist.
    fn finish(&self, app: &mut App) {
        let world = app.world_mut();
        let processes = world.get_resource_or_init::<Processes>().clone();
        let runtime = world
            .get_resource::<TokioRuntime>()
            .map(|runtime| runtime.0.clone());
        let Some(mut configured) = world.get_resource_mut::<CommandModels>() else {
            return;
        };
        let Some(runtime) = runtime else {
            if !configured.models.is_empty() {
                configured
                    .problems
                    .push(format!("{FILE}: no tokio runtime to run the models on."));
            }
            return;
        };
        let mut vendors: HashMap<ProviderId, Vec<(String, Arc<Served>)>> = HashMap::new();
        let mut specs = Vec::new();
        for entry in &configured.models {
            let launch = Launch {
                processes: processes.clone(),
                command: entry.command.clone(),
            };
            let served = if entry.persistent {
                let (jobs, queue) = mpsc::unbounded();
                runtime.spawn(serve_persistent(launch, queue));
                Served::Persistent(jobs)
            } else {
                Served::Once(runtime.clone(), launch)
            };
            vendors
                .entry(entry.vendor)
                .or_default()
                .push((entry.name.clone(), Arc::new(served)));
            specs.push(entry.spec());
        }
        let mut models = world.get_resource_or_init::<Models>();
        for spec in specs {
            models.0.catalog_mut().insert(spec);
        }
        for (vendor, served) in vendors {
            models.0.provide(vendor, Programs(served));
        }
    }
}

/// The configured models, read once at startup, and what in the file could
/// not be used.
#[derive(Resource, Clone, Default)]
pub struct CommandModels {
    /// The models.
    pub models: Vec<CommandModel>,
    /// Each mistake found in the file.
    pub problems: Vec<String>,
}

/// One `[[model]]` of the file.
#[derive(Clone, Debug)]
pub struct CommandModel {
    vendor: ProviderId,
    name: String,
    command: Vec<String>,
    context_window: Option<u32>,
    tools: bool,
    reasoning: bool,
    persistent: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct File {
    #[serde(default)]
    model: Vec<Entry>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Entry {
    id: String,
    command: Vec<String>,
    context_window: Option<u32>,
    #[serde(default = "yes")]
    tools: bool,
    #[serde(default)]
    reasoning: bool,
    #[serde(default)]
    persistent: bool,
}

fn yes() -> bool {
    true
}

impl CommandModels {
    /// The models of the file at `path`; none when there is no file.
    pub fn load(path: &Path) -> Self {
        match std::fs::read_to_string(path) {
            Ok(text) => Self::parse(&text),
            Err(why) if why.kind() == std::io::ErrorKind::NotFound => Self::default(),
            Err(why) => Self {
                problems: vec![format!("Could not read {}: {why}.", path.display())],
                ..Self::default()
            },
        }
    }

    /// The models `text`, in the file's shape, lists.
    pub fn parse(text: &str) -> Self {
        let file: File = match toml::from_str(text) {
            Ok(file) => file,
            Err(why) => {
                return Self {
                    problems: vec![format!("{FILE}: {why}")],
                    ..Self::default()
                };
            }
        };
        let mut models = Self::default();
        for entry in file.model {
            let id = entry.id.clone();
            match CommandModel::new(entry) {
                Ok(model) => models.models.push(model),
                Err(why) => models
                    .problems
                    .push(format!("{FILE}: model `{id}` skipped: {why}.")),
            }
        }
        models
    }
}

impl CommandModel {
    fn new(entry: Entry) -> Result<Self, String> {
        let Some((vendor, name)) = entry.id.split_once('/') else {
            return Err("its id needs a vendor, as in `cmd/echo`".to_owned());
        };
        let vendor = ProviderId::declare(vendor).map_err(|why| why.to_string())?;
        if !vendor.is_declared() {
            return Err("its vendor is a built-in provider".to_owned());
        }
        if entry.command.is_empty() {
            return Err("its `command` is empty".to_owned());
        }
        Ok(Self {
            vendor,
            name: name.to_owned(),
            command: entry.command,
            context_window: entry.context_window,
            tools: entry.tools,
            reasoning: entry.reasoning,
            persistent: entry.persistent,
        })
    }

    /// The model as a catalog spec, free.
    fn spec(&self) -> ModelSpec {
        let mut spec = ModelSpec::new(self.vendor, &self.name)
            .with_tools(self.tools)
            .with_pricing(Pricing::new(0.0, 0.0));
        if let Some(tokens) = self.context_window {
            spec = spec.with_context_window(tokens);
        }
        if self.reasoning {
            spec = spec.with_reasoning(ReasoningSupport::Unknown { default: None });
        }
        spec
    }
}

/// What in `command-models.toml` could not be used, as notices.
fn report_problems(models: Res<CommandModels>, mut notices: MessageWriter<Notice>) {
    for problem in &models.problems {
        notices.write(Notice::error(None, problem.clone()));
    }
}

/// How a model's program starts: as every child process does.
#[derive(Clone)]
struct Launch {
    processes: Processes,
    command: Vec<String>,
}

impl Launch {
    /// The program with its pipes, killed when dropped.
    fn spawn(&self) -> std::io::Result<Child> {
        let Some((program, args)) = self.command.split_first() else {
            return Err(std::io::Error::other("no command"));
        };
        let mut command = self.processes.launcher().command(program);
        command.args(args);
        Command::from(command)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
    }
}

/// How a configured model is served.
enum Served {
    /// A process per request, on this runtime.
    Once(Handle, Launch),
    /// Requests for the one process [`serve_persistent`] keeps.
    Persistent(mpsc::UnboundedSender<Job>),
}

/// A vendor's configured models, by name.
struct Programs(Vec<(String, Arc<Served>)>);

impl Provide for Programs {
    fn handler(
        &self,
        spec: &ModelSpec,
        _: &Catalog,
        _: &Transport,
    ) -> Result<ErasedHandler, ConnectError> {
        let served = self.0.iter().find(|(name, _)| *name == spec.id);
        let Some((_, served)) = served else {
            let why = format!("{} is not in {FILE}", spec.reference());
            return Err(ConnectError::Provider(why));
        };
        Ok(ErasedHandler::new(ProgramHandler {
            reference: spec.reference(),
            name: spec.id.clone(),
            served: Arc::clone(served),
        }))
    }
}

/// A model call to a program.
struct ProgramHandler {
    reference: String,
    name: String,
    served: Arc<Served>,
}

/// One request for a program: its line, where its output goes, and what
/// drops once the call no longer listens.
struct Job {
    id: u64,
    line: String,
    events: mpsc::UnboundedSender<Event>,
    cancel: oneshot::Receiver<()>,
}

/// What a program did.
enum Event {
    /// It wrote this line.
    Line(String),
    /// It ended before it was done: its exit code and the end of stderr.
    Exited { code: Option<i32>, stderr: String },
    /// It could not be started.
    Failed(String),
}

impl Serve for ProgramHandler {
    type Family = family::Completion;

    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor {
            key: model_key(&self.reference),
            family: FamilyDescriptor::Completion {
                model: self.reference.as_str().into(),
                capabilities: ProviderCapabilities::new(),
            },
            layers: Vec::new(),
        }
    }

    async fn serve(&self, kind: EffectKind, dispatch: Dispatch) -> Reply {
        let EffectKind::Completion { request, .. } = kind else {
            let why = format!("{} serves completions only", self.reference);
            return Reply::Outcome(Err(ErrorReport::new(ErrorKind::Internal, why)));
        };
        let id = dispatch.id().as_u64();
        let line = request_line(id, &self.reference, &request);
        let (events, received) = mpsc::unbounded();
        let (listening, cancel) = oneshot::channel();
        let job = Job {
            id,
            line,
            events,
            cancel,
        };
        match &*self.served {
            Served::Once(runtime, launch) => {
                runtime.spawn(serve_once(launch.clone(), job));
            }
            Served::Persistent(jobs) => {
                if let Err(refused) = jobs.unbounded_send(job) {
                    let failed = Event::Failed("its process is gone".to_owned());
                    refused.into_inner().events.unbounded_send(failed).ok();
                }
            }
        }
        let origin = Origin::new("command", self.reference.as_str(), self.name.as_str());
        let reference = self.reference.clone();
        let name = self.name.clone();
        Reply::written(origin, move |writer| async move {
            relay(&reference, name, received, writer).await;
            // The program's work ends with the call.
            drop(listening);
        })
    }
}

/// The request as the program reads it: one JSON line.
fn request_line(id: u64, model: &str, request: &CompletionRequest) -> String {
    let mut system = Vec::new();
    let mut messages = Vec::new();
    for message in &request.chat_history {
        match &**message {
            ChatMessage::System { content } => system.push(content.as_str()),
            ChatMessage::User { content } => {
                let content: Vec<Value> = content.iter().map(user_content).collect();
                messages.push(json!({ "role": "user", "content": content }));
            }
            ChatMessage::Assistant(assistant) => {
                let content: Vec<Value> = assistant.content.iter().map(assistant_content).collect();
                messages.push(json!({ "role": "assistant", "content": content }));
            }
        }
    }
    let tools: Vec<Value> = request
        .tools
        .iter()
        .map(|tool| {
            json!({
                "name": tool.name.as_str(),
                "description": tool.description,
                "parameters": tool.parameters,
            })
        })
        .collect();
    let line = json!({
        "id": id,
        "model": model,
        "system": system.join("\n\n"),
        "messages": messages,
        "tools": tools,
        "options": {
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "tool_choice": serde_json::to_value(&request.tool_choice).unwrap_or_default(),
            "additional_params": request.additional_params,
        },
    });
    line.to_string()
}

fn user_content(content: &UserContent) -> Value {
    match content {
        UserContent::Text(text) => json!({ "type": "text", "text": text.text }),
        UserContent::ToolResult(result) => {
            let text: Vec<String> = result
                .content
                .iter()
                .map(|item| match item {
                    ToolResultContent::Text(text) => text.text.clone(),
                    other => serde_json::to_string(other).unwrap_or_default(),
                })
                .collect();
            json!({
                "type": "tool_result",
                "id": result.call.to_string(),
                "name": result.name.as_str(),
                "content": text.join("\n"),
                "is_error": result.is_error,
            })
        }
        other => serde_json::to_value(other).unwrap_or_default(),
    }
}

fn assistant_content(content: &AssistantContent) -> Value {
    match content {
        AssistantContent::Text(text) => json!({ "type": "text", "text": text.text }),
        AssistantContent::Reasoning(reasoning) => {
            json!({ "type": "reasoning", "text": reasoning.text })
        }
        AssistantContent::ToolCall(call) => json!({
            "type": "tool_call",
            "id": call.id.to_string(),
            "name": call.function.name.as_str(),
            "arguments": call.function.arguments,
        }),
        other => serde_json::to_value(other).unwrap_or_default(),
    }
}

/// One line of a program's reply.
#[derive(Deserialize)]
#[serde(untagged)]
enum Line {
    Text {
        text: String,
    },
    Reasoning {
        reasoning: String,
    },
    ToolCall {
        tool_call: ToolCallLine,
    },
    Usage {
        usage: Usage,
    },
    Error {
        error: String,
        #[serde(default)]
        retry: bool,
    },
    Done {
        done: bool,
    },
}

#[derive(Deserialize)]
struct ToolCallLine {
    /// The program's id for the call, unused: the stream writer issues
    /// rig's own, which later requests carry.
    #[serde(default, rename = "id")]
    _id: Option<String>,
    name: String,
    #[serde(default)]
    arguments: Value,
}

/// Whether `line` ends the program's reply to a request.
fn ends_reply(line: &str) -> bool {
    matches!(
        serde_json::from_str::<Line>(line),
        Ok(Line::Done { done: true } | Line::Error { .. })
    )
}

/// Writes the program's lines into the call's reply as they come.
async fn relay(
    model: &str,
    name: String,
    mut events: mpsc::UnboundedReceiver<Event>,
    mut writer: StreamWriter,
) {
    let mut usage = Usage::default();
    let mut number = 0usize;
    while let Some(event) = events.next().await {
        let line = match event {
            Event::Line(line) => line,
            Event::Exited { code, stderr } => {
                writer.error(exited(model, code, &stderr)).await.ok();
                return;
            }
            Event::Failed(why) => {
                let report = ErrorReport::new(ErrorKind::Provider, format!("`{model}`: {why}"));
                writer.error(report).await.ok();
                return;
            }
        };
        number += 1;
        if line.trim().is_empty() {
            continue;
        }
        let written = match serde_json::from_str::<Line>(&line) {
            Ok(Line::Text { text }) => writer.text(text).await,
            Ok(Line::Reasoning { reasoning }) => writer.reasoning(reasoning).await,
            Ok(Line::ToolCall { tool_call }) => {
                writer.tool_call(tool_call.name, tool_call.arguments).await
            }
            Ok(Line::Usage { usage: reported }) => {
                usage = reported;
                Ok(())
            }
            Ok(Line::Error { error, retry }) => {
                let report = ErrorReport::new(ErrorKind::Provider, format!("`{model}`: {error}"));
                writer.error(report.with_retryable(retry)).await.ok();
                return;
            }
            Ok(Line::Done { done: false }) => Ok(()),
            Ok(Line::Done { done: true }) => {
                usage.cost = Some(Cost::default());
                let finish = Finish {
                    usage,
                    model: Some(name),
                    ..Finish::default()
                };
                writer.finish(finish).await.ok();
                return;
            }
            Err(_) => {
                let quoted: String = line.chars().take(QUOTED).collect();
                let why = format!("`{model}` wrote a malformed line {number}: {quoted}");
                writer
                    .error(ErrorReport::new(ErrorKind::Response, why))
                    .await
                    .ok();
                return;
            }
        };
        if written.is_err() {
            return;
        }
    }
    let why = format!("`{model}` stopped without an answer");
    writer
        .error(ErrorReport::new(ErrorKind::Provider, why))
        .await
        .ok();
}

/// The failure of a program that ended before it was done.
fn exited(model: &str, code: Option<i32>, stderr: &str) -> ErrorReport {
    let status = code.map_or_else(|| "a signal".to_owned(), |code| format!("code {code}"));
    if code == Some(TRY_AGAIN) {
        let why = format!("`{model}` asked to be tried again (exit {status})");
        return ErrorReport::new(ErrorKind::Provider, why).with_retryable(true);
    }
    let why = format!(
        "`{model}` exited with {status} before `done`: {}",
        stderr.trim()
    );
    ErrorReport::new(ErrorKind::Provider, why)
}

/// A running program: its stdin, its stdout's lines and the end of its
/// stderr, read until it closes.
struct Program {
    child: Child,
    /// Closed by taking it.
    stdin: Option<ChildStdin>,
    lines: Lines<BufReader<ChildStdout>>,
    stderr: JoinHandle<String>,
}

impl Program {
    fn start(launch: &Launch) -> Result<Self, String> {
        let mut child = launch
            .spawn()
            .map_err(|why| format!("`{}` did not start: {why}", launch.command.join(" ")))?;
        let (Some(stdin), Some(stdout), Some(mut stderr)) =
            (child.stdin.take(), child.stdout.take(), child.stderr.take())
        else {
            return Err("its pipes could not be opened".to_owned());
        };
        let stderr = tokio::spawn(async move {
            let mut tail = Vec::new();
            let mut chunk = [0u8; 4096];
            while let Ok(read @ 1..) = stderr.read(&mut chunk).await {
                tail.extend(chunk.iter().take(read));
                let cut = tail.len().saturating_sub(STDERR_TAIL);
                tail.drain(..cut);
            }
            String::from_utf8_lossy(&tail).into_owned()
        });
        Ok(Self {
            child,
            stdin: Some(stdin),
            lines: BufReader::new(stdout).lines(),
            stderr,
        })
    }

    /// Sends one line to the program.
    async fn send(&mut self, line: &str) -> std::io::Result<()> {
        let Some(stdin) = &mut self.stdin else {
            return Err(std::io::ErrorKind::BrokenPipe.into());
        };
        stdin.write_all(line.as_bytes()).await?;
        stdin.write_all(b"\n").await?;
        stdin.flush().await
    }

    /// The program's end once its stdout closed.
    async fn exited(mut self) -> Event {
        let code = self
            .child
            .wait()
            .await
            .ok()
            .and_then(|status| status.code());
        let stderr = self.stderr.await.unwrap_or_default();
        Event::Exited { code, stderr }
    }
}

/// A process for one request: it gets the request and its stdin closes;
/// it is killed when the call stops listening.
async fn serve_once(launch: Launch, job: Job) {
    let Job {
        line,
        events,
        cancel,
        ..
    } = job;
    let mut program = match Program::start(&launch) {
        Ok(program) => program,
        Err(why) => {
            events.unbounded_send(Event::Failed(why)).ok();
            return;
        }
    };
    let work = async move {
        // A program that does not read its stdin still answers.
        program.send(&line).await.ok();
        program.stdin = None;
        while let Ok(Some(line)) = program.lines.next_line().await {
            if events.unbounded_send(Event::Line(line)).is_err() {
                return;
            }
        }
        events.unbounded_send(program.exited().await).ok();
    };
    // Dropping `work` drops the child, which kills it.
    select(Box::pin(work), cancel).await;
}

/// The one process of a `persistent` model: started for the first request
/// and again after it exited, sent one request at a time. A call that stops
/// listening is cancelled with `{"cancel": id}`, and the program's lines up
/// to its `done` or `error` are dropped.
async fn serve_persistent(launch: Launch, mut jobs: mpsc::UnboundedReceiver<Job>) {
    let mut running: Option<Program> = None;
    while let Some(job) = jobs.next().await {
        if job.events.is_closed() {
            continue;
        }
        let mut program = match running.take() {
            Some(program) => program,
            None => match Program::start(&launch) {
                Ok(program) => program,
                Err(why) => {
                    job.events.unbounded_send(Event::Failed(why)).ok();
                    continue;
                }
            },
        };
        if program.send(&job.line).await.is_err() {
            job.events.unbounded_send(program.exited().await).ok();
            continue;
        }
        running = answer(program, job).await;
    }
}

/// Relays the program's lines for `job` until its reply ended; the
/// program, unless it exited.
async fn answer(mut program: Program, job: Job) -> Option<Program> {
    let Job {
        id,
        events,
        mut cancel,
        ..
    } = job;
    loop {
        let next = match select(Box::pin(program.lines.next_line()), &mut cancel).await {
            Either::Left((line, _)) => Some(line),
            Either::Right(_) => None,
        };
        match next {
            Some(Ok(Some(line))) => {
                let ended = ends_reply(&line);
                if events.unbounded_send(Event::Line(line)).is_err() {
                    break;
                }
                if ended {
                    return Some(program);
                }
            }
            Some(_) => {
                events.unbounded_send(program.exited().await).ok();
                return None;
            }
            None => break,
        }
    }
    // The call stopped listening before the reply ended.
    let cancel = json!({ "cancel": id }).to_string();
    if program.send(&cancel).await.is_err() {
        return None;
    }
    while let Ok(Some(line)) = program.lines.next_line().await {
        if ends_reply(&line) {
            return Some(program);
        }
    }
    None
}

#[cfg(test)]
mod tests;
