//! A harness that speaks the Agent Client Protocol on its stdin and stdout,
//! as a model: one process and one ACP session for a whole conversation.
//! The client offers the agent no file system and no terminal, names Rig's
//! MCP server as the session's only one, and refuses every permission the
//! agent asks for that is not for a tool of that server. What differs
//! between agents is an [`Agent`].

use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;

use rig_harness::rig_core::completion::{Effort, Usage};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncWrite, AsyncWriteExt, BufReader, Lines};
use tokio::sync::mpsc;

use crate::harness::{Event, Harness, Launch, Part, Running, Stderr, spawn};

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

    /// What a person runs to sign in.
    fn login_hint(&self) -> &'static str;

    /// Whether the tool call a permission request describes is a call of
    /// one of Rig's tools.
    fn is_rig_tool(&self, tool_call: &Value) -> bool;

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
}

/// An [`Agent`] as a harness.
pub(crate) struct Acp<A>(pub Arc<A>);

impl<A: Agent> Harness for Acp<A> {
    fn start(&self, launch: Launch<'_>) -> Result<Running, String> {
        let Launch {
            mut command,
            dir,
            system,
            model,
            effort,
            mcp,
            tools,
        } = launch;
        let opening = Opening {
            dir: dir.to_path_buf(),
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
        let (events, reported) = mpsc::unbounded_channel();
        let stderr = Stderr::follow(pipes.stderr);
        let agent = self.0.clone();
        tokio::spawn(async move {
            let name = agent.name();
            let mut peer = Peer::new(pipes.stdout, pipes.stdin, events.clone(), agent);
            if let Err(why) = converse(&mut peer, opening, prompted).await {
                let why = stderr.explain(&format!("{name}: {why}"));
                let _ = events.send(Event::Done(Err(why)));
            }
        });
        Ok(Running {
            prompts,
            events: reported,
            _alive: Box::new(process),
        })
    }
}

/// What a session opens with.
pub(crate) struct Opening {
    /// The session's private directory, the agent's working directory.
    pub dir: PathBuf,
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
    lines: Lines<BufReader<R>>,
    writer: W,
    next_id: u64,
    events: mpsc::UnboundedSender<Event>,
    agent: Arc<dyn Agent>,
    /// The client's name for the prompt being answered.
    prompt: Option<String>,
    /// Whether the agent reported an inference's counters since the last
    /// prompt was answered.
    counted: bool,
}

impl<R: AsyncRead + Unpin, W: AsyncWrite + Unpin> Peer<R, W> {
    pub(crate) fn new(
        reader: R,
        writer: W,
        events: mpsc::UnboundedSender<Event>,
        agent: Arc<dyn Agent>,
    ) -> Self {
        Self {
            lines: BufReader::new(reader).lines(),
            writer,
            next_id: 0,
            events,
            agent,
            prompt: None,
            counted: false,
        }
    }

    async fn write(&mut self, message: &Value) -> Result<(), Failed> {
        let mut line = message.to_string();
        line.push('\n');
        self.writer
            .write_all(line.as_bytes())
            .await
            .map_err(|_| Failed::Closed)
    }

    /// The next JSON message. A line that is not JSON is skipped.
    async fn read(&mut self) -> Result<Value, Failed> {
        loop {
            let line = self
                .lines
                .next_line()
                .await
                .ok()
                .flatten()
                .ok_or(Failed::Closed)?;
            if let Ok(message) = serde_json::from_str::<Value>(&line)
                && message.is_object()
            {
                return Ok(message);
            }
        }
    }

    /// Sends a request and waits for its answer, handling what the agent
    /// sends meanwhile.
    async fn request(&mut self, method: &str, params: Value) -> Result<Value, Failed> {
        self.next_id += 1;
        let id = self.next_id;
        self.write(&json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params }))
            .await?;
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
            return match message.get("error") {
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
            };
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
                self.update(params.get("update").unwrap_or(&Value::Null));
            } else if let Some(usage) = self.agent.counted(method, params) {
                self.counted = true;
                let _ = self.events.send(Event::Usage(usage));
            }
            return Ok(());
        };
        let answer = match method {
            "session/request_permission" => {
                let tool_call = params.get("toolCall").unwrap_or(&Value::Null);
                let ours = self.agent.is_rig_tool(tool_call);
                json!({ "result": { "outcome": decide(ours, params.get("options")) } })
            }
            _ => json!({ "error": { "code": -32601, "message": format!("no method `{method}`") } }),
        };
        let mut answer = answer;
        if let Some(answer) = answer.as_object_mut() {
            answer.insert("jsonrpc".to_owned(), "2.0".into());
            answer.insert("id".to_owned(), id.clone());
        }
        self.write(&answer).await
    }

    /// Reports the text of a `session/update`.
    fn update(&self, update: &Value) {
        let text = || {
            update
                .get("content")
                .and_then(|content| content.get("text"))
                .and_then(Value::as_str)
                .filter(|text| !text.is_empty())
                .map(str::to_owned)
        };
        let event = match update.get("sessionUpdate").and_then(Value::as_str) {
            Some("agent_message_chunk") => text().map(Event::Text),
            Some("agent_thought_chunk") => text().map(Event::Reasoning),
            _ => None,
        };
        if let Some(event) = event {
            let _ = self.events.send(event);
        }
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

/// Opens the session, then answers each prompt until the prompts end.
pub(crate) async fn converse<R, W>(
    peer: &mut Peer<R, W>,
    opening: Opening,
    mut prompts: mpsc::UnboundedReceiver<Vec<Part>>,
) -> Result<(), String>
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
    let meta = agent.session_meta(&initialized, &opening.system);
    let mut system = meta.is_none().then(|| opening.system.clone());
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
    let opened = match peer.request("session/new", new.clone()).await {
        Err(Failed::Refused {
            code: AUTH_REQUIRED,
            ..
        }) => {
            let not_signed_in = || format!("it is not signed in: run `{}`", agent.login_hint());
            let method = agent.sign_in(&initialized).ok_or_else(not_signed_in)?;
            peer.request("authenticate", json!({ "methodId": method }))
                .await
                .map_err(|_| not_signed_in())?;
            peer.request("session/new", new).await
        }
        opened => opened,
    }
    .map_err(failed("it opened no session"))?;
    let session = opened
        .get("sessionId")
        .and_then(Value::as_str)
        .ok_or("it opened a session without an id")?
        .to_owned();
    agent.opened(&opening)?;
    for (method, params) in agent.choose(&session, &opened, &opening) {
        peer.request(method, params)
            .await
            .map_err(failed(&format!("it refused the model `{}`", opening.model)))?;
    }

    loop {
        // Between prompts the agent may still send requests and updates.
        let prompt = loop {
            tokio::select! {
                prompt = prompts.recv() => break prompt,
                message = peer.read() => {
                    let message = message.map_err(|why| why.to_string())?;
                    peer.incoming(&message).await.map_err(|why| why.to_string())?;
                }
            }
        };
        let Some(prompt) = prompt else {
            return Ok(());
        };
        let mut blocks: Vec<Value> = Vec::new();
        if let Some(system) = system.take().filter(|system| !system.is_empty()) {
            blocks.push(json!({
                "type": "text",
                "text": format!("<instructions>\n{system}\n</instructions>\n"),
            }));
        }
        blocks.extend(prompt.iter().map(|part| match part {
            Part::Text(text) => json!({ "type": "text", "text": text }),
            Part::Image { media_type, data } => {
                json!({ "type": "image", "mimeType": media_type, "data": data })
            }
        }));
        let named = uuid::Uuid::new_v4().to_string();
        let mut asked = json!({ "sessionId": session, "prompt": blocks });
        if let (Some(meta), Some(asked)) = (agent.prompt_meta(&named), asked.as_object_mut()) {
            asked.insert("_meta".to_owned(), meta);
        }
        peer.prompt = Some(named);
        let answered = peer.request("session/prompt", asked).await;
        peer.prompt = None;
        let done = match answered {
            Ok(result) => {
                // The turn's totals, unless each inference was counted.
                let cache_apart = agent.input_leaves_out_cache();
                if let Some(usage) = usage(&result, cache_apart).filter(|_| !peer.counted) {
                    let _ = peer.events.send(Event::Usage(usage));
                }
                peer.counted = false;
                match result.get("stopReason").and_then(Value::as_str) {
                    Some("refusal") => Err("the model refused to answer".to_owned()),
                    Some("cancelled") => Err("the turn was cancelled".to_owned()),
                    _ => Ok(()),
                }
            }
            Err(Failed::Closed) => return Err(Failed::Closed.to_string()),
            Err(refused) => Err(refused.to_string()),
        };
        let _ = peer.events.send(Event::Done(done));
    }
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
        .map(|(input, output)| input + output);
    usage.is_reported().then_some(usage)
}
