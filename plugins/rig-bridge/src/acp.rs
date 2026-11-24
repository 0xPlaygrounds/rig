//! A harness that speaks the Agent Client Protocol on its stdin and stdout,
//! as a model: one process and one ACP session for a whole conversation.
//! The client offers the agent no file system and no terminal, names Rig's
//! MCP server as the session's only one, and refuses every permission the
//! agent asks for that is not for a tool of that server. What differs
//! between agents is an [`Agent`].

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use rig_harness::rig_core::completion::{Effort, FinishReason, Usage};
use serde_json::{Value, json};
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::sync::{mpsc, oneshot};

use crate::harness::{
    ContextSnapshot, Event, HANDSHAKE_TIMEOUT, Harness, Launch, Part, Prompt, Running, Stderr,
    Steer, parse_effort, spawn,
};
use crate::rpc::{Transport, method_not_found, positive_u32};

#[cfg(test)]
mod tests;

/// The MCP server name Rig's tools are under.
pub(crate) const SERVER: &str = "rig";

/// The JSON-RPC code an agent refuses a session with before a sign-in.
const AUTH_REQUIRED: i64 = -32000;

/// What one ACP agent needs that the protocol does not say.
pub(crate) trait Agent: Send + Sync + 'static {
    /// The agent's name, for messages.
    fn name(&self) -> &'static str;

    /// Adds the arguments, environment and files that start the program as
    /// an ACP agent with nothing of its own turned on.
    fn prepare(&self, command: &mut Command, opening: &Opening) -> Result<(), String>;

    /// The `_meta` of `session/new`, where an agent takes a system prompt.
    /// `None` when it takes none there: the first prompt then opens with it.
    fn session_meta(&self, initialized: &Value, system: &str) -> Option<Value>;

    /// The requests that choose the model and effort once the session is
    /// open, as method and parameters. `opened` is the `session/new` result.
    fn choose(
        &self,
        session: &str,
        opened: &Value,
        opening: &Opening,
    ) -> Vec<(&'static str, Value)>;

    /// The `authenticate` method that signs in with the login the program
    /// already has, if it has one to offer.
    fn sign_in(&self, initialized: &Value) -> Option<String>;

    /// What the model is told about the agent, after Rig's system prompt.
    fn note(&self, project: Option<&Path>) -> String;

    /// Whether the agent offers its model a tool added while it runs.
    fn relists(&self) -> bool {
        true
    }

    /// Whether a session the agent saved offers its model the tools a new
    /// process is given, as [`Harness::retools_saved`] says.
    fn retools_saved(&self) -> bool {
        true
    }

    /// Wire value for ACP `session/set_config_option` (`reasoning_effort`)
    /// when this agent supports live in-session effort updates.
    fn effort_value(&self, _effort: Effort) -> Option<&'static str> {
        None
    }

    /// What a person runs to sign in.
    fn login_hint(&self) -> &'static str;

    /// Whether the tool call a permission request describes is a call of
    /// one of Rig's tools, given any `title` previously announced for the
    /// same `toolCallId` in a `tool_call` update.
    fn is_rig_tool(&self, tool_call: &Value, recorded_title: Option<&str>) -> bool;

    /// Whether an `agent_message_chunk` text fragment should be forwarded to
    /// Rig. Agents that emit CLI status banners as message chunks return
    /// `false` for those lines.
    fn keep_text(&self, text: &str) -> bool {
        let _ = text;
        true
    }

    /// Whether `session/new` names Rig's MCP server. An agent that is told
    /// of it another way, in [`prepare`](Self::prepare), says no.
    fn serves_in_session(&self) -> bool {
        true
    }

    /// Checks, once the session is open, that the agent runs as
    /// [`prepare`](Self::prepare) set it up.
    fn opened(&self, opening: &Opening) -> Result<(), String> {
        let _ = opening;
        Ok(())
    }

    /// Whether a prompt result's input counter leaves out the tokens read
    /// from and written to the cache.
    fn input_leaves_out_cache(&self) -> bool {
        false
    }

    /// Whether `session/prompt`'s `result.usage` is cumulative across the
    /// whole session instead of reset for each prompt turn.
    fn cumulative_usage(&self) -> bool {
        false
    }

    /// The `_meta` of `initialize`, for an agent that reads one.
    fn initialize_meta(&self) -> Option<Value> {
        None
    }

    /// The `_meta` of the `session/prompt` the client calls `prompt`, for
    /// an agent that names its prompts.
    fn prompt_meta(&self, prompt: &str) -> Option<Value> {
        let _ = prompt;
        None
    }

    /// The result of the prompt the client calls `prompt`, when the
    /// notification `method` with `params` is the agent's own word that it
    /// ended: some agents say so and leave the request unanswered.
    fn ended(&self, method: &str, params: &Value, prompt: &str) -> Option<Value> {
        let _ = (method, params, prompt);
        None
    }

    /// The token counters of one inference, when the notification `method`
    /// with `params` is the agent's own report of them. An agent that
    /// reports none is counted by its prompt results, a turn at a time.
    fn counted(&self, method: &str, params: &Value) -> Option<Usage> {
        let _ = (method, params);
        None
    }

    /// How the agent that answered `initialize` with `started` takes
    /// `parts` as a user message for the prompt it is answering. The
    /// protocol has no way: `None` unless the agent has one of its own.
    fn steer(&self, started: &Value, session: &str, parts: &[Part]) -> Option<Steering> {
        let _ = (started, session, parts);
        None
    }

    /// Whether `result`, the answer to a [`Steering::Request`], says the
    /// agent holds the message for its running prompt.
    fn steered(&self, result: &Value) -> bool {
        let _ = result;
        true
    }
}

/// How an agent takes a user message for the prompt it is answering.
pub(crate) enum Steering {
    /// By a request of its own beside the prompt, as method and parameters.
    Request(&'static str, Value),
    /// As the session's next prompt, sent while this one is answered: the
    /// agent ends this one at once and goes on with its work, tool calls
    /// in flight included, under the next.
    Prompt,
}

/// An [`Agent`] as a harness.
pub(crate) struct Acp<A>(pub Arc<A>);

impl<A: Agent> Harness for Acp<A> {
    fn note(&self, project: Option<&Path>) -> String {
        self.0.note(project)
    }

    fn relists(&self) -> bool {
        self.0.relists()
    }

    fn retools_saved(&self) -> bool {
        self.0.retools_saved()
    }

    fn changes_effort(&self) -> bool {
        self.0.effort_value(Effort::High).is_some()
    }

    fn start(&self, launch: Launch<'_>) -> Result<Running, String> {
        let Launch {
            mut command,
            dir,
            resume,
            system,
            model,
            effort,
            mcp,
            tools,
        } = launch;
        let opening = Opening {
            dir: dir.to_path_buf(),
            resume: resume.map(str::to_owned),
            system: system.to_owned(),
            model: model.to_owned(),
            effort,
            mcp_url: mcp.url().to_owned(),
            mcp_token: mcp.token().to_owned(),
            tools: tools
                .iter()
                .map(|tool| tool.name.as_str().to_owned())
                .collect(),
        };
        command.current_dir(dir);
        self.0.prepare(&mut command, &opening)?;
        let (process, pipes) = spawn(command)?;
        let (prompts, prompted) = mpsc::unbounded_channel();
        let (steers, steered) = mpsc::unbounded_channel();
        let (stops, stopped) = mpsc::unbounded_channel();
        let (events, reported) = mpsc::unbounded_channel();
        let stderr = Stderr::follow(pipes.stderr);
        let agent = self.0.clone();
        tokio::spawn(async move {
            let name = agent.name();
            let mut peer = Peer::new(pipes.stdout, pipes.stdin, events.clone(), agent);
            let conversed = converse(&mut peer, opening, prompted, steered, stopped);
            if let Err(why) = conversed.await {
                let why = stderr.explain(&format!("{name}: {why}")).await;
                let _ = events.send(Event::Done(Err(why)));
            }
        });
        Ok(Running {
            prompts,
            steers: Some(steers),
            stops,
            events: reported,
            _alive: Box::new(process),
        })
    }
}

/// What a session opens with.
pub(crate) struct Opening {
    /// The session's private directory, the agent's working directory.
    pub dir: PathBuf,
    /// The session a process before this one saved, to go on from.
    pub resume: Option<String>,
    /// The system prompt.
    pub system: String,
    /// The model's id.
    pub model: String,
    /// The effort asked for.
    pub effort: Option<Effort>,
    /// Where Rig's MCP server is.
    pub mcp_url: String,
    /// The bearer token that names the session to it.
    pub mcp_token: String,
    /// The names of Rig's tools.
    pub tools: Vec<String>,
}

/// Why a request failed.
#[derive(Debug)]
enum Failed {
    /// The agent is gone.
    Closed,
    /// The agent answered with an error.
    Refused { code: i64, message: String },
}

impl std::fmt::Display for Failed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Closed => f.write_str("the program exited"),
            Self::Refused { code, message } => write!(f, "{message} ({code})"),
        }
    }
}

/// The client's end of the connection.
pub(crate) struct Peer<R, W> {
    transport: Transport<R, W>,
    events: mpsc::UnboundedSender<Event>,
    agent: Arc<dyn Agent>,
    /// The client's name for the prompt being answered.
    prompt: Option<String>,
    /// Whether the agent reported an inference's counters since the last
    /// prompt was answered.
    counted: bool,
    /// `toolCallId` -> `title` seen in `tool_call` updates, for agents that
    /// strip the MCP server prefix in `session/request_permission`.
    call_titles: HashMap<String, String>,
    /// What the agent answered `initialize` with.
    started: Value,
    /// Whether the agent is loading a saved session: it then reports the
    /// conversation over again, which is not an answer to anything.
    loading: bool,
    /// The counters of the last prompt result, for an agent whose results
    /// count the whole session.
    totals: Usage,
}

impl<R: AsyncRead + Unpin, W: AsyncWrite + Unpin> Peer<R, W> {
    pub(crate) fn new(
        reader: R,
        writer: W,
        events: mpsc::UnboundedSender<Event>,
        agent: Arc<dyn Agent>,
    ) -> Self {
        Self {
            transport: Transport::new(reader, writer),
            events,
            agent,
            prompt: None,
            counted: false,
            call_titles: HashMap::new(),
            started: Value::Null,
            loading: false,
            totals: Usage::new(),
        }
    }

    async fn read(&mut self) -> Result<Value, Failed> {
        self.transport
            .next_object()
            .await
            .map_err(|()| Failed::Closed)
    }

    /// Sends a request and waits for its answer, handling what the agent
    /// sends meanwhile.
    async fn request(&mut self, method: &str, params: Value) -> Result<Value, Failed> {
        let id = self
            .transport
            .send_request(method, params, true)
            .await
            .map_err(|()| Failed::Closed)?;
        loop {
            let message = self.read().await?;
            let answers = message.get("method").is_none()
                && message.get("id").and_then(Value::as_u64) == Some(id);
            if !answers {
                let ended = self.prompt.as_deref().and_then(|prompt| {
                    let method = message.get("method")?.as_str()?;
                    let params = message.get("params").unwrap_or(&Value::Null);
                    self.agent.ended(method, params, prompt)
                });
                if let Some(result) = ended {
                    return Ok(result);
                }
                self.incoming(&message).await?;
                continue;
            }
            return answered(&message);
        }
    }

    /// Sends `asked` as the session's next prompt and waits for its answer,
    /// as [`request`](Self::request) does. A steer that comes meanwhile is
    /// given to the agent its own way, and told whether the agent took it;
    /// a stop for this prompt, the `turn`th, cancels it, as one that came
    /// before the prompt did (`stopped`) does at once, and the agent then
    /// answers it as cancelled.
    async fn prompted(
        &mut self,
        session: &str,
        asked: Value,
        steers: &mut mpsc::UnboundedReceiver<Steer>,
        (turn, stopped, stops): (u64, bool, &mut mpsc::UnboundedReceiver<u64>),
    ) -> Result<Value, Failed> {
        let closed = |()| Failed::Closed;
        let send = self.transport.send_request("session/prompt", asked, true);
        let mut id = send.await.map_err(closed)?;
        // The steer the agent has not answered: its request's id, who waits
        // to hear of it, and whether it is a prompt.
        let mut steering: Option<(u64, oneshot::Sender<Result<(), String>>, bool)> = None;
        let mut cancelled = stopped;
        loop {
            if std::mem::take(&mut cancelled) {
                let cancel = json!({
                    "jsonrpc": "2.0",
                    "method": "session/cancel",
                    "params": { "sessionId": session },
                });
                self.transport.write(&cancel).await.map_err(closed)?;
            }
            let message = tokio::select! {
                message = self.read() => message?,
                Some(stop) = stops.recv() => {
                    cancelled = stop == turn;
                    continue;
                }
                Some(steer) = steers.recv(), if steering.is_none() => {
                    let (method, params, prompt) =
                        match self.agent.steer(&self.started, session, &steer.parts) {
                            Some(Steering::Request(method, params)) => (method, params, false),
                            Some(Steering::Prompt) => {
                                let prompt = blocks(&steer.parts);
                                let asked = json!({ "sessionId": session, "prompt": prompt });
                                ("session/prompt", asked, true)
                            }
                            None => {
                                let name = self.agent.name();
                                let _ = steer.taken.send(Err(format!("{name} takes none")));
                                continue;
                            }
                        };
                    let send = self.transport.send_request(method, params, true);
                    steering = Some((send.await.map_err(closed)?, steer.taken, prompt));
                    continue;
                }
            };
            let answers = message
                .get("method")
                .is_none()
                .then(|| message.get("id").and_then(Value::as_u64))
                .flatten();
            if answers == Some(id) {
                match (steering.take(), answered(&message)) {
                    // The agent ended this prompt to go on under the next.
                    (Some((next, taken, true)), Ok(result)) => {
                        self.count(&result);
                        let _ = taken.send(Ok(()));
                        id = next;
                        continue;
                    }
                    (_, result) => return result,
                }
            }
            if let Some((_, taken, prompt)) =
                steering.take_if(|(steer, ..)| answers == Some(*steer))
            {
                let _ = taken.send(match answered(&message) {
                    Ok(result) if prompt || self.agent.steered(&result) => Ok(()),
                    Ok(_) => Err(format!("{} did not take the message", self.agent.name())),
                    Err(refused) => Err(refused.to_string()),
                });
                continue;
            }
            let ended = self.prompt.as_deref().and_then(|prompt| {
                let method = message.get("method")?.as_str()?;
                let params = message.get("params").unwrap_or(&Value::Null);
                self.agent.ended(method, params, prompt)
            });
            if let Some(result) = ended {
                return Ok(result);
            }
            self.incoming(&message).await?;
        }
    }

    /// Reports the counters of a prompt's `result`, the prompt's totals,
    /// unless the agent counted each inference of it.
    fn count(&mut self, result: &Value) {
        let was_counted = std::mem::take(&mut self.counted);
        let cache_apart = self.agent.input_leaves_out_cache();
        let Some(reported) = usage(result, cache_apart).filter(|_| !was_counted) else {
            return;
        };
        let counted = if self.agent.cumulative_usage() {
            let since = usage_delta(&self.totals, reported);
            self.totals = reported;
            since
        } else {
            reported
        };
        if counted.is_reported() {
            let _ = self.events.send(Event::Usage(counted));
        }
    }

    /// Handles a message that answers nothing: an update is reported, a
    /// permission is decided, any other request is refused.
    async fn incoming(&mut self, message: &Value) -> Result<(), Failed> {
        let Some(method) = message.get("method").and_then(Value::as_str) else {
            return Ok(());
        };
        let params = message.get("params").unwrap_or(&Value::Null);
        let Some(id) = message.get("id").filter(|id| !id.is_null()) else {
            if method == "session/update" {
                if !self.loading {
                    self.update(params.get("update").unwrap_or(&Value::Null));
                }
            } else if self.loading {
                // What a loaded session counts again was counted before.
            } else if let Some(usage) = self.agent.counted(method, params) {
                self.counted = true;
                let _ = self.events.send(Event::Usage(usage));
            }
            return Ok(());
        };
        let answer = match method {
            "session/request_permission" => {
                let tool_call = params.get("toolCall").unwrap_or(&Value::Null);
                let recorded = tool_call
                    .get("toolCallId")
                    .and_then(Value::as_str)
                    .and_then(|id| self.call_titles.get(id))
                    .map(String::as_str);
                let ours = self.agent.is_rig_tool(tool_call, recorded);
                json!({ "result": { "outcome": decide(ours, params.get("options")) } })
            }
            _ => method_not_found(method),
        };
        self.transport
            .reply(id, answer, true)
            .await
            .map_err(|()| Failed::Closed)
    }

    /// Reports the text of a `session/update`.
    fn update(&mut self, update: &Value) {
        let kind = update.get("sessionUpdate").and_then(Value::as_str);
        if kind == Some("tool_call") {
            if let (Some(id), Some(title)) = (
                update.get("toolCallId").and_then(Value::as_str),
                update.get("title").and_then(Value::as_str),
            ) {
                self.call_titles.insert(id.to_owned(), title.to_owned());
            }
            return;
        }
        if kind == Some("usage_update") {
            let used = self
                .prompt
                .is_some()
                .then(|| {
                    update
                        .get("used")
                        .and_then(Value::as_u64)
                        .filter(|used| *used > 0)
                })
                .flatten();
            let window = positive_u32(update.get("size"));
            if used.is_some() || window.is_some() {
                let _ = self
                    .events
                    .send(Event::Context(ContextSnapshot { used, window }));
            }
            return;
        }
        let text = || {
            update
                .get("content")
                .and_then(|content| content.get("text"))
                .and_then(Value::as_str)
                .filter(|text| !text.is_empty())
                .map(str::to_owned)
        };
        let event = match kind {
            Some("agent_message_chunk") => text()
                .filter(|text| self.agent.keep_text(text))
                .map(Event::Text),
            Some("agent_thought_chunk") => text().map(Event::Reasoning),
            _ => None,
        };
        if let Some(event) = event {
            let _ = self.events.send(event);
        }
    }
}

/// A prompt's parts as content blocks.
fn blocks(parts: &[Part]) -> Vec<Value> {
    let block = |part: &Part| match part {
        Part::Text(text) => json!({ "type": "text", "text": text }),
        Part::Image { media_type, data } => {
            json!({ "type": "image", "mimeType": media_type, "data": data })
        }
    };
    parts.iter().map(block).collect()
}

/// What a response says: its result, or how the agent refused.
fn answered(message: &Value) -> Result<Value, Failed> {
    match message.get("error") {
        Some(error) => Err(Failed::Refused {
            code: error
                .get("code")
                .and_then(Value::as_i64)
                .unwrap_or_default(),
            message: error
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("the request failed")
                .to_owned(),
        }),
        None => Ok(message.get("result").cloned().unwrap_or(Value::Null)),
    }
}

/// The outcome of a permission request: the first option that allows the
/// call once for one of Rig's tools, and a refusal for anything else.
fn decide(ours: bool, options: Option<&Value>) -> Value {
    let kind =
        |option: &Value, kind: &str| option.get("kind").and_then(Value::as_str) == Some(kind);
    let options = options
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default();
    let wanted: &[&str] = if ours {
        &["allow_once", "allow_always"]
    } else {
        &["reject_once", "reject_always"]
    };
    let chosen = wanted
        .iter()
        .find_map(|wanted| options.iter().find(|option| kind(option, wanted)))
        .and_then(|option| option.get("optionId"));
    match chosen {
        Some(option) => json!({ "outcome": "selected", "optionId": option }),
        None => json!({ "outcome": "cancelled" }),
    }
}

/// Builds an ACP `session/set_config_option` request for `reasoning_effort`.
pub(crate) fn set_effort_request(session: &str, value: &str) -> (&'static str, Value) {
    (
        "session/set_config_option",
        json!({
            "sessionId": session,
            "configId": "reasoning_effort",
            "value": value,
        }),
    )
}

/// Whether a session supports runtime `reasoning_effort` updates, and what
/// default effort to restore when a prompt asks for `None` (provider default).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EffortConfig {
    /// `session/new` returned a `configOptions` list without `reasoning_effort`.
    Unsupported,
    /// Runtime updates are allowed; the `Option<Effort>` is the model's default.
    Supported(Option<Effort>),
}

/// Opens the session, a new one or the saved one, then answers each prompt
/// until the prompts end. A steer goes into the prompt being answered, and
/// a stop cancels it.
pub(crate) async fn converse<R, W>(
    peer: &mut Peer<R, W>,
    opening: Opening,
    mut prompts: mpsc::UnboundedReceiver<Prompt>,
    mut steers: mpsc::UnboundedReceiver<Steer>,
    mut stops: mpsc::UnboundedReceiver<u64>,
) -> Result<(), String>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    let agent = peer.agent.clone();
    let mut current_effort = opening.effort;
    let (session, mut system, effort_config) =
        tokio::time::timeout(HANDSHAKE_TIMEOUT, handshake(peer, &opening))
            .await
            .map_err(|_| "it did not finish starting in time".to_owned())??;

    // The number of the prompt being answered, and of the last one stopped.
    let (mut turn, mut stopped) = (0u64, 0u64);
    loop {
        // Between prompts the agent may still send requests and updates. A
        // prompt is taken before a stop, which may be the one for it. A
        // stop with no prompt being answered is for one that is over, or
        // for one that is only now on its way: that one is cancelled once
        // it is sent, so that the agent has it.
        let prompt = loop {
            tokio::select! {
                biased;
                prompt = prompts.recv() => break prompt,
                Some(steer) = steers.recv() => {
                    let _ = steer.taken.send(Err("no prompt is being answered".to_owned()));
                }
                Some(stop) = stops.recv() => stopped = stopped.max(stop),
                message = peer.read() => {
                    let message = message.map_err(|why| why.to_string())?;
                    peer.incoming(&message).await.map_err(|why| why.to_string())?;
                }
            }
        };
        let Some(prompt) = prompt else {
            return Ok(());
        };
        turn += 1;
        if prompt.effort != current_effort
            && let EffortConfig::Supported(default_effort) = effort_config
            && let Some(value) = prompt
                .effort
                .or(default_effort)
                .and_then(|effort| agent.effort_value(effort))
        {
            let (method, params) = set_effort_request(&session, value);
            match peer.request(method, params).await {
                Ok(_) => current_effort = prompt.effort,
                Err(Failed::Closed) => return Err(Failed::Closed.to_string()),
                Err(refused) => {
                    let _ = peer.events.send(Event::Done(Err(refused.to_string())));
                    continue;
                }
            }
        }
        let mut blocks: Vec<Value> = Vec::new();
        if let Some(system) = system.take().filter(|system| !system.is_empty()) {
            blocks.push(json!({
                "type": "text",
                "text": format!("<instructions>\n{system}\n</instructions>\n"),
            }));
        }
        blocks.extend(self::blocks(&prompt.parts));
        let named = uuid::Uuid::new_v4().to_string();
        let mut asked = json!({ "sessionId": session, "prompt": blocks });
        if let (Some(meta), Some(asked)) = (agent.prompt_meta(&named), asked.as_object_mut()) {
            asked.insert("_meta".to_owned(), meta);
        }
        peer.prompt = Some(named);
        peer.call_titles.clear();
        let answered = peer
            .prompted(
                &session,
                asked,
                &mut steers,
                (turn, turn <= stopped, &mut stops),
            )
            .await;
        peer.prompt = None;
        let done = match answered {
            Ok(result) => {
                peer.count(&result);
                match result.get("stopReason").and_then(Value::as_str) {
                    Some("max_tokens") => Event::Done(Ok(FinishReason::Length)),
                    Some("refusal") => Event::Done(Err("the model refused to answer".to_owned())),
                    Some("cancelled") => Event::Stopped,
                    _ => Event::Done(Ok(FinishReason::Stop)),
                }
            }
            Err(Failed::Closed) => return Err(Failed::Closed.to_string()),
            Err(refused) => {
                peer.counted = false;
                Event::Done(Err(refused.to_string()))
            }
        };
        let _ = peer.events.send(done);
    }
}

/// Runs the ACP startup handshake (`initialize`, `session/new`, optional auth,
/// and model/effort selection).
async fn handshake<R, W>(
    peer: &mut Peer<R, W>,
    opening: &Opening,
) -> Result<(String, Option<String>, EffortConfig), String>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    let failed = |what: &str| {
        let what = what.to_owned();
        move |why: Failed| format!("{what}: {why}")
    };
    let agent = peer.agent.clone();
    let mut hello = json!({
        "protocolVersion": 1,
        "clientCapabilities": {
            "fs": { "readTextFile": false, "writeTextFile": false },
            "terminal": false,
        },
        "clientInfo": { "name": "rig", "title": "Rig", "version": env!("CARGO_PKG_VERSION") },
    });
    if let (Some(meta), Some(hello)) = (agent.initialize_meta(), hello.as_object_mut()) {
        hello.insert("_meta".to_owned(), meta);
    }
    let initialized = peer
        .request("initialize", hello)
        .await
        .map_err(failed("it did not start"))?;
    peer.started.clone_from(&initialized);
    let meta = agent.session_meta(&initialized, &opening.system);
    // A saved session has the system prompt its first prompt opened with.
    let system = (meta.is_none() && opening.resume.is_none()).then(|| opening.system.clone());
    let servers = if agent.serves_in_session() {
        json!([{
            "type": "http",
            "name": SERVER,
            "url": opening.mcp_url,
            "headers": [{ "name": "Authorization", "value": format!("Bearer {}", opening.mcp_token) }],
        }])
    } else {
        json!([])
    };
    let mut new = json!({ "cwd": opening.dir.to_string_lossy(), "mcpServers": servers });
    if let (Some(meta), Some(new)) = (meta, new.as_object_mut()) {
        new.insert("_meta".to_owned(), meta);
    }
    // A saved session is resumed where the agent can, and else loaded: the
    // agent then reports the whole conversation again before it answers.
    let open = match (&opening.resume, new.as_object_mut()) {
        (Some(saved), Some(new)) => {
            new.insert("sessionId".to_owned(), saved.as_str().into());
            let resumes = initialized.pointer("/agentCapabilities/sessionCapabilities/resume");
            let resumes =
                resumes.is_some_and(|able| able.is_object() || able.as_bool() == Some(true));
            if resumes {
                "session/resume"
            } else if initialized.pointer("/agentCapabilities/loadSession") == Some(&true.into()) {
                "session/load"
            } else {
                return Err("it cannot go on from a saved session".to_owned());
            }
        }
        _ => "session/new",
    };
    peer.loading = open == "session/load";
    let opened = match peer.request(open, new.clone()).await {
        Err(Failed::Refused {
            code: AUTH_REQUIRED,
            ..
        }) => {
            let not_signed_in = || format!("it is not signed in: run `{}`", agent.login_hint());
            let method = agent.sign_in(&initialized).ok_or_else(not_signed_in)?;
            peer.request("authenticate", json!({ "methodId": method }))
                .await
                .map_err(|_| not_signed_in())?;
            peer.request(open, new).await
        }
        opened => opened,
    }
    .map_err(failed("it opened no session"))?;
    peer.loading = false;
    // An agent names a new session; a saved one keeps its name.
    let session = opened
        .get("sessionId")
        .and_then(Value::as_str)
        .or(opening.resume.as_deref())
        .ok_or("it opened a session without an id")?
        .to_owned();
    let _ = peer.events.send(Event::Saved(session.clone()));
    agent.opened(opening)?;
    if let Some(window) = model_context_window(&opened, &opening.model) {
        let _ = peer.events.send(Event::Context(ContextSnapshot {
            used: None,
            window: Some(window),
        }));
    }
    for (method, params) in agent.choose(&session, &opened, opening) {
        peer.request(method, params)
            .await
            .map_err(failed(&format!("it refused the model `{}`", opening.model)))?;
    }
    let effort_config = session_effort_config(&opened, &opening.model);
    Ok((session, system, effort_config))
}

/// Looks up one `configOptions` entry by `id` in a `session/new` response.
pub(crate) fn config_option<'a>(opened: &'a Value, id: &str) -> Option<&'a Value> {
    opened
        .get("configOptions")
        .and_then(Value::as_array)?
        .iter()
        .find(|opt| opt.get("id").and_then(Value::as_str) == Some(id))
}

/// Inspects `session/new` metadata for whether the session supports runtime
/// `reasoning_effort` configuration and what its unmodified default effort is.
fn session_effort_config(opened: &Value, model: &str) -> EffortConfig {
    let model_default = opened
        .pointer("/models/availableModels")
        .and_then(Value::as_array)
        .and_then(|models| {
            models
                .iter()
                .find(|entry| entry.get("modelId").and_then(Value::as_str) == Some(model))
        })
        .and_then(|entry| entry.pointer("/_meta/reasoningEfforts"))
        .and_then(Value::as_array)
        .and_then(|efforts| {
            efforts.iter().find_map(|item| {
                (item.get("default").and_then(Value::as_bool) == Some(true))
                    .then(|| item.get("value").and_then(Value::as_str))
                    .flatten()
                    .and_then(parse_effort)
            })
        });
    if opened
        .get("configOptions")
        .and_then(Value::as_array)
        .is_none()
    {
        return EffortConfig::Supported(model_default);
    }
    let Some(option) = config_option(opened, "reasoning_effort") else {
        return EffortConfig::Unsupported;
    };
    let current = option
        .get("currentValue")
        .and_then(Value::as_str)
        .and_then(parse_effort);
    EffortConfig::Supported(model_default.or(current))
}

/// Extracts a model's declared context window from `session/new` metadata
/// (`models.availableModels[*]._meta.totalContextTokens`), when present.
fn model_context_window(opened: &Value, model: &str) -> Option<u32> {
    positive_u32(
        opened
            .pointer("/models/availableModels")
            .and_then(Value::as_array)?
            .iter()
            .find(|entry| entry.get("modelId").and_then(Value::as_str) == Some(model))?
            .pointer("/_meta/totalContextTokens"),
    )
}

/// The token counters of a `session/prompt` result: the turn's totals.
fn usage(result: &Value, cache_apart: bool) -> Option<Usage> {
    let counted = result
        .get("usage")
        .or_else(|| result.get("_meta").and_then(|meta| meta.get("usage")))?;
    let count = |names: &[&str]| {
        names
            .iter()
            .find_map(|name| counted.get(name).and_then(Value::as_u64))
    };
    let mut usage = Usage::new();
    usage.input_tokens = count(&["inputTokens", "input_tokens"]);
    usage.output_tokens = count(&["outputTokens", "output_tokens"]);
    usage.cached_input_tokens = count(&[
        "cachedReadTokens",
        "cacheReadTokens",
        "cache_read_input_tokens",
    ]);
    usage.cache_creation_input_tokens = count(&[
        "cachedWriteTokens",
        "cacheCreationTokens",
        "cache_creation_input_tokens",
    ]);
    usage.reasoning_tokens = count(&["thoughtTokens", "reasoningTokens", "reasoning_tokens"]);
    if cache_apart {
        let cached =
            usage.cached_input_tokens.unwrap_or(0) + usage.cache_creation_input_tokens.unwrap_or(0);
        usage.input_tokens = usage.input_tokens.map(|fresh| fresh + cached);
    }
    usage.total_tokens = usage
        .input_tokens
        .zip(usage.output_tokens)
        .map(|(input, output)| input + output)
        .or_else(|| count(&["totalTokens", "total_tokens"]));
    usage.is_reported().then_some(usage)
}

/// Subtracts the prior session-cumulative [`Usage`] from `current` to obtain
/// this turn's delta. If the primary counters decreased (for example after a
/// backend reset or a CLI change to per-turn reporting), returns `current`
/// unchanged instead of clamping to zero.
fn usage_delta(prev: &Usage, current: Usage) -> Usage {
    let decreased =
        |cur: Option<u64>, old: Option<u64>| matches!((cur, old), (Some(c), Some(o)) if c < o);
    if decreased(current.input_tokens, prev.input_tokens)
        || decreased(current.output_tokens, prev.output_tokens)
    {
        return current;
    }
    let sub = |cur: Option<u64>, old: Option<u64>| cur.map(|c| c.saturating_sub(old.unwrap_or(0)));
    let mut delta = Usage::new();
    delta.input_tokens = sub(current.input_tokens, prev.input_tokens);
    delta.output_tokens = sub(current.output_tokens, prev.output_tokens);
    delta.cached_input_tokens = sub(current.cached_input_tokens, prev.cached_input_tokens);
    delta.cache_creation_input_tokens = sub(
        current.cache_creation_input_tokens,
        prev.cache_creation_input_tokens,
    );
    delta.reasoning_tokens = sub(current.reasoning_tokens, prev.reasoning_tokens);
    delta.total_tokens = delta
        .input_tokens
        .zip(delta.output_tokens)
        .map(|(input, output)| input + output)
        .or_else(|| sub(current.total_tokens, prev.total_tokens));
    delta
}
