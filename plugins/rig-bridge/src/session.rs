//! A harness kept running for one conversation, and the handler that makes
//! it a model. A session answers one request at a time: it streams what the
//! model says until the model calls a tool or its turn ends. A tool call is
//! handed to Rig as an ordinary tool call and its MCP request stays open; the
//! request that carries the results answers it, and the same harness turn
//! goes on.
//!
//! Rig owns the conversation. A request continues a session only when its
//! history is the one the session answered plus what came since; any other
//! request starts a new harness with the history written out as text.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use base64::Engine;
use rig_harness::prelude::Processes;
use rig_harness::rig_core::completion::{
    AssistantContent, CompletionRequest, Effort, FinishReason, Message, Reasoning, ToolDefinition,
    UnsupportedOption, Usage,
};
use rig_harness::rig_core::effect::{self, EffectKind, FamilyDescriptor, HandlerDescriptor};
use rig_harness::rig_core::error::{ErrorKind, ErrorReport};
use rig_harness::rig_core::message::{
    Api, CallId, DocumentSourceKind, Image, MimeType, Origin, ToolResult, ToolResultContent,
    UserContent,
};
use rig_harness::rig_core::operation::Finish;
use rig_harness::rig_core::serve::{Dispatch, Reply, Serve};
use serde_json::{Value, json};
use tokio::sync::{OnceCell, mpsc, oneshot};

use crate::harness::{Event, Harness, Launch, Part, Running};
use crate::mcp::{Call, Entry, Outcome, Server};

#[cfg(test)]
mod tests;

/// How long a reply stays open after a tool call for the calls made beside
/// it. A harness makes parallel calls one request each, so the first cannot
/// say how many follow; a call that arrives later starts the next reply. A
/// reply also stays open until the inference that made the call is counted.
const SETTLE: Duration = Duration::from_millis(50);

/// How long a harness told that the tools changed has to list them again.
const RELIST: Duration = Duration::from_secs(5);

/// What the bridge's models share: where harnesses run and the sessions
/// waiting for their next request.
pub(crate) struct Bridge {
    runtime: tokio::runtime::Handle,
    processes: Processes,
    server: OnceCell<Server>,
    /// Waiting sessions, the most recently used last.
    waiting: Mutex<VecDeque<Session>>,
    /// How many sessions may wait. The least recently used is ended first.
    limit: usize,
    /// The project Rig works in, when it is known.
    project: Option<PathBuf>,
}

impl Bridge {
    pub(crate) fn new(
        runtime: tokio::runtime::Handle,
        processes: Processes,
        limit: usize,
        project: Option<PathBuf>,
    ) -> Self {
        Self {
            project,
            runtime,
            processes,
            server: OnceCell::new(),
            waiting: Mutex::default(),
            limit,
        }
    }

    fn waiting(&self) -> std::sync::MutexGuard<'_, VecDeque<Session>> {
        self.waiting.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The waiting session `history` continues, with what to send it.
    fn resume(&self, setup: &Setup, history: &[Arc<Message>]) -> Option<(Session, Next)> {
        let mut waiting = self.waiting();
        let (index, next) = waiting
            .iter()
            .enumerate()
            .rev()
            .find_map(|(index, session)| {
                (session.setup.fits(setup))
                    .then(|| session.next(history))
                    .flatten()
                    .map(|next| (index, next))
            })?;
        waiting.remove(index).map(|session| (session, next))
    }

    /// Keeps `session` for its next request, ending those past the limit.
    fn keep(&self, session: Session) {
        let ended: Vec<Session> = {
            let mut waiting = self.waiting();
            waiting.push_back(session);
            let over = waiting.len().saturating_sub(self.limit);
            waiting.drain(..over).collect()
        };
        drop(ended);
    }
}

/// What a session was started with. A request with another setup needs
/// another session.
#[derive(Clone, PartialEq)]
struct Setup {
    program: PathBuf,
    model: String,
    effort: Option<Effort>,
    system: String,
    tools: Arc<[Arc<ToolDefinition>]>,
}

impl Setup {
    /// Whether a session with this setup can serve `other`: the same in
    /// all but the tools, which a running harness can be given anew.
    fn fits(&self, other: &Self) -> bool {
        (&self.program, &self.model, self.effort, &self.system)
            == (&other.program, &other.model, other.effort, &other.system)
    }
}

/// A harness's model as an effect handler.
pub(crate) struct Model {
    pub bridge: Arc<Bridge>,
    pub harness: Arc<dyn Harness>,
    /// The harness's program.
    pub program: PathBuf,
    /// The catalog vendor the model is filed under.
    pub vendor: &'static str,
    /// The model's id, as the harness names it.
    pub model: String,
}

impl Serve for Model {
    type Family = effect::family::Completion;

    fn descriptor(&self) -> HandlerDescriptor {
        let label = format!("{}/{}", self.vendor, self.model);
        HandlerDescriptor {
            key: effect::model_key(&label),
            family: FamilyDescriptor::Completion {
                model: label.into(),
                capabilities: Default::default(),
            },
            layers: Vec::new(),
        }
    }

    /// Answers a completion on the session its history continues, or a new
    /// one. Dropping the reply ends the harness.
    async fn serve(&self, kind: EffectKind, _dispatch: Dispatch) -> Reply {
        let EffectKind::Completion { request, .. } = kind else {
            return Reply::Outcome(Err(ErrorReport::new(
                ErrorKind::Internal,
                format!("a model was sent {}", kind.name()),
            )));
        };
        let (out, mut written) = mpsc::unbounded_channel();
        let step = Step {
            bridge: self.bridge.clone(),
            harness: self.harness.clone(),
            program: self.program.clone(),
            model: self.model.clone(),
        };
        self.bridge.runtime.spawn(step.run(request, out));
        let origin = Origin::new(Api::from_static("bridge"), self.vendor, self.model.clone());
        Reply::written(origin, |mut writer| async move {
            while let Some(wrote) = written.recv().await {
                let sent = match wrote {
                    Wrote::Text(text) => writer.text(text).await,
                    Wrote::Reasoning(text) => writer.reasoning(text).await,
                    Wrote::Call { name, arguments } => writer.tool_call(name, arguments).await,
                    Wrote::Finish(finish) => {
                        let _ = writer.finish(*finish).await;
                        return;
                    }
                    Wrote::Failed(why) => {
                        let _ = writer
                            .error(ErrorReport::new(ErrorKind::Provider, why))
                            .await;
                        return;
                    }
                };
                if sent.is_err() {
                    return;
                }
            }
        })
    }

    /// A harness takes a prompt, a model and an effort: the sampling,
    /// length, tool choice, output schema and reasoning budget of a request
    /// are left out.
    fn check(&self, kind: &EffectKind) -> Vec<UnsupportedOption> {
        let EffectKind::Completion { request, .. } = kind else {
            return Vec::new();
        };
        left_out(request)
            .into_iter()
            .map(|option| {
                UnsupportedOption::new(
                    option,
                    self.vendor,
                    self.model.clone(),
                    "the program takes no such option",
                )
            })
            .collect()
    }
}

/// The options of `request` a harness has no way to take.
fn left_out(request: &CompletionRequest) -> Vec<&'static str> {
    let budgeted = matches!(request.options.reasoning, Some(Reasoning::Budget { .. }));
    [
        ("temperature", request.temperature.is_some()),
        ("max_tokens", request.max_tokens.is_some()),
        ("tool_choice", request.tool_choice.is_some()),
        ("output_schema", request.output_schema.is_some()),
        ("reasoning", budgeted),
    ]
    .into_iter()
    .filter_map(|(option, set)| set.then_some(option))
    .collect()
}

/// What a step writes into its reply.
enum Wrote {
    Text(String),
    Reasoning(String),
    Call { name: String, arguments: Value },
    Finish(Box<Finish>),
    Failed(String),
}

/// One request's work, run on the tokio runtime.
struct Step {
    bridge: Arc<Bridge>,
    harness: Arc<dyn Harness>,
    program: PathBuf,
    model: String,
}

impl Step {
    async fn run(self, request: Arc<CompletionRequest>, out: mpsc::UnboundedSender<Wrote>) {
        let answered = tokio::select! {
            // The reply was dropped: the session ends with this future.
            () = out.closed() => return,
            answered = self.answer(&request, &out) => answered,
        };
        let _ = out.send(match answered {
            Ok(finish) => Wrote::Finish(Box::new(finish)),
            Err(why) => Wrote::Failed(why),
        });
    }

    /// Streams one reply into `out` and keeps the session for the next.
    async fn answer(
        &self,
        request: &CompletionRequest,
        out: &mpsc::UnboundedSender<Wrote>,
    ) -> Result<Finish, String> {
        let (system, history) = split(&request.chat_history);
        let setup = Setup {
            program: self.program.clone(),
            model: self.model.clone(),
            effort: effort(request),
            system,
            tools: request.tools.iter().cloned().collect(),
        };
        // A session that cannot take what is next, its harness gone or a
        // call given up, is ended: the conversation goes on in a new one.
        let resumed = match self.bridge.resume(&setup, &history) {
            // The harness is told of other tools before it goes on, or is
            // ended when it cannot take them.
            Some((mut session, next)) => (session.retool(&setup.tools).await
                && session.send(next).is_ok())
            .then_some(session),
            None => None,
        };
        let mut session = match resumed {
            Some(session) => session,
            None => self.start(setup, &history).await?,
        };
        let finish = session.answer(out).await?;
        session.seen = history;
        self.bridge.keep(session);
        Ok(finish)
    }

    /// A new session, sent `history` as its first prompt.
    async fn start(&self, setup: Setup, history: &[Arc<Message>]) -> Result<Session, String> {
        let server = self
            .bridge
            .server
            .get_or_try_init(Server::start)
            .await
            .map_err(|why| format!("could not serve the tools: {why}"))?;
        let (entry, calls) = server.open(setup.tools.clone());
        let dir = Dir::create().map_err(|why| format!("could not make a directory: {why}"))?;
        let note = self.harness.note(self.bridge.project.as_deref());
        let system = match (setup.system.as_str(), note.as_str()) {
            (system, "") => system.to_owned(),
            ("", note) => note.to_owned(),
            (system, note) => format!("{system}\n\n{note}"),
        };
        let running = self.harness.start(Launch {
            command: self.bridge.processes.launcher().command(&setup.program),
            dir: &dir.0,
            system: &system,
            model: &setup.model,
            effort: setup.effort,
            mcp: &entry,
            tools: &setup.tools,
        })?;
        let mut session = Session {
            setup,
            relists: self.harness.relists(),
            seen: Vec::new(),
            said: Said::default(),
            open: Vec::new(),
            calls,
            running,
            entry,
            _dir: dir,
        };
        session.send(Next::Prompt(opening(history)))?;
        Ok(session)
    }
}

/// The reasoning effort a request asks for.
fn effort(request: &CompletionRequest) -> Option<Effort> {
    match request.options.reasoning {
        Some(Reasoning::Effort(effort)) => Some(effort),
        Some(Reasoning::Off | Reasoning::Budget { .. }) | None => None,
        Some(_) => None,
    }
}

/// A request's system prompt and the conversation after it.
fn split(messages: &[Arc<Message>]) -> (String, Vec<Arc<Message>>) {
    let mut system = String::new();
    let mut history = Vec::with_capacity(messages.len());
    for message in messages {
        match &**message {
            Message::System { content } => {
                if !system.is_empty() {
                    system.push_str("\n\n");
                }
                system.push_str(content);
            }
            Message::User { .. } | Message::Assistant(_) => history.push(message.clone()),
        }
    }
    (system, history)
}

/// A session's private directory, removed with it.
struct Dir(PathBuf);

impl Dir {
    fn create() -> std::io::Result<Self> {
        let path =
            std::env::temp_dir().join(format!("rig-bridge-{}", uuid::Uuid::new_v4().simple()));
        let mut builder = std::fs::DirBuilder::new();
        #[cfg(unix)]
        std::os::unix::fs::DirBuilderExt::mode(&mut builder, 0o700);
        builder.create(&path)?;
        Ok(Self(path))
    }
}

impl Drop for Dir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// What a waiting session is sent to go on.
#[derive(Debug, PartialEq)]
enum Next {
    /// The next prompt.
    Prompt(Vec<Part>),
    /// The results of its open calls, in the calls' order.
    Results(Vec<Outcome>),
}

/// The text and tool calls of a reply.
#[derive(Default, PartialEq)]
struct Said {
    text: String,
    calls: Vec<(String, serde_json::Map<String, Value>)>,
}

impl Said {
    /// What an assistant message in a history says.
    fn of(content: &[AssistantContent]) -> Self {
        let mut said = Self::default();
        for content in content {
            match content {
                AssistantContent::Text(text) => said.text.push_str(&text.text),
                AssistantContent::ToolCall(call) => said.calls.push((
                    call.function.name.as_str().to_owned(),
                    call.function.arguments.clone(),
                )),
                _ => {}
            }
        }
        said
    }
}

/// A running harness and where its conversation stands.
struct Session {
    setup: Setup,
    /// Whether the harness takes a changed tool list while it runs.
    relists: bool,
    /// The history of the last request this session answered. The reply to
    /// it is the message after.
    seen: Vec<Arc<Message>>,
    /// What the last reply said: its text, and each tool call's name and
    /// arguments.
    said: Said,
    /// The tool calls Rig is running, in the order the reply made them.
    open: Vec<oneshot::Sender<Outcome>>,
    calls: mpsc::UnboundedReceiver<Call>,
    running: Running,
    entry: Entry,
    _dir: Dir,
}

impl Session {
    /// What to send this session for `history`, if `history` is the one it
    /// answered, its reply, and then only what answers that reply: the
    /// results of its open calls, or else the next user messages.
    fn next(&self, history: &[Arc<Message>]) -> Option<Next> {
        let (seen, after) = history.split_at_checked(self.seen.len())?;
        let same = seen
            .iter()
            .zip(&self.seen)
            .all(|(now, then)| Arc::ptr_eq(now, then) || now == then);
        let (reply, since) = after.split_first()?;
        let Message::Assistant(reply) = &**reply else {
            return None;
        };
        // Two conversations can share a history: the reply tells them apart.
        if !same || since.is_empty() || Said::of(&reply.content) != self.said {
            return None;
        }
        let mut contents = Vec::new();
        for message in since {
            let Message::User { content } = &**message else {
                return None;
            };
            contents.extend(content);
        }
        if self.open.is_empty() {
            let has_result = contents
                .iter()
                .any(|content| matches!(content, UserContent::ToolResult(_)));
            return (!has_result).then(|| Next::Prompt(parts(contents)));
        }
        let calls: Vec<&CallId> = reply
            .content
            .iter()
            .filter_map(|content| match content {
                AssistantContent::ToolCall(call) => Some(&call.id),
                _ => None,
            })
            .collect();
        if calls.len() != self.open.len() {
            return None;
        }
        let mut outcomes: Vec<Option<Outcome>> = vec![None; calls.len()];
        let mut said = Vec::new();
        for content in contents {
            match content {
                UserContent::ToolResult(result) => {
                    let index = calls.iter().position(|call| **call == result.call)?;
                    *outcomes.get_mut(index)? = Some(outcome(result));
                }
                other => said.push(other),
            }
        }
        let mut outcomes = outcomes.into_iter().collect::<Option<Vec<_>>>()?;
        // What Rig says beside the results reaches the model with the last.
        if let Some(last) = outcomes.last_mut() {
            last.content.extend(parts(said).iter().map(block));
        }
        Some(Next::Results(outcomes))
    }

    /// Offers `tools` in place of the session's, if they differ: whether
    /// the harness now has them. It has once it listed the tools again
    /// after being told they changed; one that is not told, or does not
    /// list them in time, has not.
    async fn retool(&mut self, tools: &Arc<[Arc<ToolDefinition>]>) -> bool {
        if self.setup.tools == *tools {
            return true;
        }
        if !self.relists {
            return false;
        }
        let Some(mut listed) = self.entry.retool(tools.clone()) else {
            return false;
        };
        let relisted = tokio::time::timeout(RELIST, listed.changed()).await;
        if !matches!(relisted, Ok(Ok(()))) {
            return false;
        }
        self.setup.tools = tools.clone();
        true
    }

    fn send(&mut self, next: Next) -> Result<(), String> {
        match next {
            Next::Prompt(prompt) => self
                .running
                .prompts
                .send(prompt)
                .map_err(|_| "the harness is gone".to_owned()),
            Next::Results(outcomes) => {
                let mut given_up = false;
                for (answer, outcome) in self.open.drain(..).zip(outcomes) {
                    given_up |= answer.send(outcome).is_err();
                }
                if given_up {
                    return Err("the harness gave up a tool call".to_owned());
                }
                Ok(())
            }
        }
    }

    /// Streams the harness into `out` until its turn ends or it calls
    /// tools, which stay open.
    async fn answer(&mut self, out: &mpsc::UnboundedSender<Wrote>) -> Result<Finish, String> {
        self.said = Said::default();
        let mut usage = Usage::new();
        let mut settled = None;
        // Whether an inference began whose counters have not arrived: a
        // harness may call a tool before it finishes reading the message.
        let mut counting = false;
        loop {
            let settle = async {
                match settled {
                    Some(at) => tokio::time::sleep_until(at).await,
                    None => std::future::pending().await,
                }
            };
            tokio::select! {
                event = self.running.events.recv() => match event {
                    Some(Event::Text(text)) => {
                        self.said.text.push_str(&text);
                        let _ = out.send(Wrote::Text(text));
                    }
                    Some(Event::Reasoning(text)) => drop(out.send(Wrote::Reasoning(text))),
                    Some(Event::Began) => counting = true,
                    Some(Event::Usage(counted)) => {
                        usage += counted;
                        counting = false;
                    }
                    // A turn that ends with calls open gave them up: Rig
                    // must not run them for a harness that no longer waits.
                    Some(Event::Done(Ok(()))) if !self.open.is_empty() => {
                        return Err("the harness ended its turn before its tools answered".to_owned());
                    }
                    Some(Event::Done(Ok(()))) => return Ok(finish(usage, FinishReason::Stop)),
                    Some(Event::Done(Err(why))) => return Err(why),
                    None => return Err("the harness exited before it answered".to_owned()),
                },
                call = self.calls.recv() => {
                    let Some(Call { name, arguments, answer }) = call else {
                        return Err("the tool server closed".to_owned());
                    };
                    self.said.calls.push((name.clone(), arguments.clone()));
                    let _ = out.send(Wrote::Call { name, arguments: Value::Object(arguments) });
                    self.open.push(answer);
                    settled.get_or_insert_with(|| tokio::time::Instant::now() + SETTLE);
                }
                () = settle, if !counting => return Ok(finish(usage, FinishReason::ToolCalls)),
            }
        }
    }
}

fn finish(usage: Usage, reason: FinishReason) -> Finish {
    Finish {
        usage,
        reason: Some(reason),
        ..Finish::default()
    }
}

/// A tool result as MCP content.
fn outcome(result: &ToolResult) -> Outcome {
    let content = result
        .content
        .iter()
        .map(|content| match content {
            ToolResultContent::Text(text) => block(&Part::Text(text.text.clone())),
            ToolResultContent::Image(image) => block(&picture(image)),
            ToolResultContent::Json { value } => block(&Part::Text(value.to_string())),
        })
        .collect();
    Outcome {
        content,
        is_error: result.is_error,
    }
}

/// A part as an MCP content block.
fn block(part: &Part) -> Value {
    match part {
        Part::Text(text) => json!({ "type": "text", "text": text }),
        Part::Image { media_type, data } => {
            json!({ "type": "image", "data": data, "mimeType": media_type })
        }
    }
}

/// An image as a part, or a note in its place when its bytes are elsewhere.
fn picture(image: &Image) -> Part {
    let data = match &image.data {
        DocumentSourceKind::Base64(data) => Some(data.clone()),
        DocumentSourceKind::Raw(bytes) => {
            Some(base64::engine::general_purpose::STANDARD.encode(bytes))
        }
        _ => None,
    };
    match (data, &image.media_type) {
        (Some(data), Some(media_type)) => Part::Image {
            media_type: media_type.to_mime_type().to_owned(),
            data,
        },
        _ => Part::Text("[an image that could not be sent]".to_owned()),
    }
}

/// User content as prompt parts. Content with no text form is named.
fn parts<'a>(contents: impl IntoIterator<Item = &'a UserContent>) -> Vec<Part> {
    contents
        .into_iter()
        .map(|content| match content {
            UserContent::Text(text) => Part::Text(text.text.clone()),
            UserContent::Image(image) => picture(image),
            UserContent::ToolResult(result) => Part::Text(told(result)),
            UserContent::Audio(_) => Part::Text("[audio that could not be sent]".to_owned()),
            UserContent::Video(_) => Part::Text("[a video that could not be sent]".to_owned()),
            UserContent::Document(_) => {
                Part::Text("[a document that could not be sent]".to_owned())
            }
        })
        .collect()
}

/// A tool result as transcript text.
fn told(result: &ToolResult) -> String {
    let mut text = format!("[result of `{}`", result.name.as_str());
    text.push_str(if result.is_error {
        ", an error]\n"
    } else {
        "]\n"
    });
    for content in &result.content {
        match content {
            ToolResultContent::Text(said) => text.push_str(&said.text),
            ToolResultContent::Image(_) => text.push_str("[an image]"),
            ToolResultContent::Json { value } => text.push_str(&value.to_string()),
        }
        text.push('\n');
    }
    text
}

/// The first prompt of a new session for `history`: the user's last
/// messages, after the conversation before them written out as text.
fn opening(history: &[Arc<Message>]) -> Vec<Part> {
    let asked = |message: &Arc<Message>| match &**message {
        Message::User { content } => !content
            .iter()
            .any(|content| matches!(content, UserContent::ToolResult(_))),
        _ => false,
    };
    let latest = history
        .iter()
        .rev()
        .take_while(|message| asked(message))
        .count();
    let (before, latest) = history.split_at(history.len() - latest);
    let mut prompt = Vec::new();
    if !before.is_empty() {
        let mut text = String::from(
            "The conversation so far follows between the markers. Continue it: the tool calls \
             in it were yours.\n\n<conversation>\n",
        );
        for message in before {
            match &**message {
                Message::User { content } => {
                    text.push_str("## User\n");
                    for part in parts(content) {
                        match part {
                            Part::Text(said) => text.push_str(&said),
                            Part::Image { .. } => text.push_str("[an image]"),
                        }
                        text.push('\n');
                    }
                }
                Message::Assistant(reply) => {
                    text.push_str("## Assistant\n");
                    for content in &reply.content {
                        match content {
                            AssistantContent::Text(said) => text.push_str(&said.text),
                            AssistantContent::ToolCall(call) => text.push_str(&format!(
                                "[called `{}` with {}]",
                                call.function.name.as_str(),
                                Value::Object(call.function.arguments.clone()),
                            )),
                            _ => continue,
                        }
                        text.push('\n');
                    }
                }
                Message::System { .. } => {}
            }
            text.push('\n');
        }
        text.push_str("</conversation>\n");
        if latest.is_empty() {
            text.push_str("\nGo on from where the conversation stops.");
        }
        prompt.push(Part::Text(text));
    }
    for message in latest {
        if let Message::User { content } = &**message {
            prompt.extend(parts(content));
        }
    }
    prompt
}
