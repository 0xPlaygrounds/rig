//! Meta AI Muse (`muse`) as a model, over `muse serve` (MSP JSON-RPC).
//! The program runs with private XDG directories (sharing only `auth.json`),
//! `--preset empty`, and workspace tools disabled, so it
//! has none of its own tools, instructions, memory or skills. Rig's system
//! prompt is passed via `session/start` `config.developerPrompt` and Rig's
//! MCP server via `config.mcpServers`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use rig_harness::rig_core::completion::{FinishReason, Usage};
use serde_json::{Value, json};
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::sync::mpsc;

use crate::harness::{
    Event, HANDSHAKE_TIMEOUT, Harness, Launch, Part, Prompt, Running, Stderr, Steer,
    clear_nested_tmpdir, env_path, link, note, spawn,
};
use crate::rpc::{Transport, method_not_found};

#[cfg(test)]
mod tests;

/// The MCP server name Rig's tools are under: the model sees a tool as
/// `mcp__rig.<name>`.
const SERVER: &str = "rig";

/// Variables removed so Muse answers with its own sign-in.
const REMOVED_ENV: &[&str] = &[
    "META_API_KEY",
    "MUSE_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
];

/// Meta AI Muse.
pub(crate) struct Muse;

/// Where the user's Muse sign-in lives.
fn signed_in() -> Option<PathBuf> {
    let config = env_path("XDG_CONFIG_HOME").or_else(|| Some(env_path("HOME")?.join(".config")))?;
    Some(config.join("muse").join("auth.json"))
}

impl Harness for Muse {
    fn note(&self, project: Option<&Path>) -> String {
        note(&format!("mcp__{SERVER}.read"), "", Some(project))
    }

    fn changes_effort(&self) -> bool {
        true
    }

    fn start(&self, launch: Launch<'_>) -> Result<Running, String> {
        let Launch {
            mut command,
            dir,
            resume,
            system,
            model,
            effort: _,
            mcp,
            tools: _,
        } = launch;
        let xdg_config = dir.join("xdg_config");
        let muse_config = xdg_config.join("muse");
        let xdg_data = dir.join("xdg_data");
        let xdg_state = dir.join("xdg_state");
        std::fs::create_dir_all(&muse_config)
            .and_then(|()| std::fs::create_dir_all(&xdg_data))
            .and_then(|()| std::fs::create_dir_all(&xdg_state))
            .map_err(|why| format!("could not write the session's files: {why}"))?;
        if let Some(auth) = signed_in().filter(|auth| auth.is_file()) {
            link(&auth, &muse_config.join("auth.json"))
                .map_err(|why| format!("could not reach the sign-in: {why}"))?;
        }
        command
            .current_dir(dir)
            .args([
                "serve",
                "--preset",
                "empty",
                "--disable-shell",
                "--disable-write",
                "--disable-request-user-input",
                "--model",
                model,
            ])
            .env("XDG_CONFIG_HOME", &xdg_config)
            .env("XDG_DATA_HOME", &xdg_data)
            .env("XDG_STATE_HOME", &xdg_state)
            .env("MUSE_NO_AUTO_UPDATE", "1");
        clear_nested_tmpdir(&mut command, dir);
        for name in REMOVED_ENV {
            command.env_remove(name);
        }
        let (process, pipes) = spawn(command)?;
        let (prompts, prompted) = mpsc::unbounded_channel();
        let (steers, steered) = mpsc::unbounded_channel();
        let (stops, stopped) = mpsc::unbounded_channel();
        let (events, reported) = mpsc::unbounded_channel();
        let stderr = Stderr::follow(pipes.stderr);
        let opening = Opening {
            system: system.to_owned(),
            model: model.to_owned(),
            resume: resume.map(str::to_owned),
            mcp_url: mcp.url().to_owned(),
            mcp_token: mcp.token().to_owned(),
        };
        tokio::spawn(async move {
            let mut wire = Wire::new(pipes.stdout, pipes.stdin, events.clone());
            let conversed = converse(&mut wire, opening, prompted, steered, stopped);
            if let Err(why) = conversed.await {
                let why = stderr.explain(&format!("muse: {why}")).await;
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

/// What a Muse MSP session opens with.
pub(crate) struct Opening {
    pub system: String,
    pub model: String,
    /// The session a process before this one saved, to go on from.
    pub resume: Option<String>,
    pub mcp_url: String,
    pub mcp_token: String,
}

/// A UUIDv7 string for MSP `commandId` fields (48-bit Unix ms timestamp +
/// version 7 + random payload from `Uuid::new_v4()`).
pub(crate) fn uuidv7() -> String {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or_default();
    let mut bytes = *uuid::Uuid::new_v4().as_bytes();
    let ts = millis.to_be_bytes();
    bytes[0] = ts[2];
    bytes[1] = ts[3];
    bytes[2] = ts[4];
    bytes[3] = ts[5];
    bytes[4] = ts[6];
    bytes[5] = ts[7];
    bytes[6] = (bytes[6] & 0x0f) | 0x70;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    uuid::Uuid::from_bytes(bytes).to_string()
}

/// The client's end of a Muse MSP stdio session.
pub(crate) struct Wire<R, W> {
    transport: Transport<R, W>,
    events: mpsc::UnboundedSender<Event>,
    /// `itemId` -> item `kind` (`agentMessage`, `reasoning`, ...).
    items: HashMap<String, String>,
    /// How the active turn ended, once `turn/completed` arrives.
    ended: Option<Event>,
}

impl<R: AsyncRead + Unpin, W: AsyncWrite + Unpin> Wire<R, W> {
    pub(crate) fn new(reader: R, writer: W, events: mpsc::UnboundedSender<Event>) -> Self {
        Self {
            transport: Transport::new(reader, writer),
            events,
            items: HashMap::new(),
            ended: None,
        }
    }

    /// Reads the next JSON object. Cancel-safe for use in `tokio::select!`.
    async fn next_message(&mut self) -> Result<Value, String> {
        self.transport.next_message().await
    }

    async fn handle(&mut self, message: Value) -> Result<(), String> {
        let events = &self.events;
        let items = &mut self.items;
        let ended = &mut self.ended;
        let _ = self
            .transport
            .dispatch(message, None, true, decide, |method, params| {
                notify(events, items, ended, method, params);
            })
            .await?;
        Ok(())
    }

    async fn request(&mut self, method: &str, params: Value) -> Result<Value, String> {
        let events = &self.events;
        let items = &mut self.items;
        let ended = &mut self.ended;
        self.transport
            .request(method, params, true, decide, |method, params| {
                notify(events, items, ended, method, params);
            })
            .await
    }

    async fn notify(&mut self, method: &str, params: Value) -> Result<(), String> {
        self.transport
            .write(&json!({
                "jsonrpc": "2.0",
                "method": method,
                "params": params,
            }))
            .await
            .map_err(|()| "the program exited".to_owned())
    }
}

fn notify(
    events: &mpsc::UnboundedSender<Event>,
    items: &mut HashMap<String, String>,
    ended: &mut Option<Event>,
    method: &str,
    params: &Value,
) {
    match method {
        "item/started" => {
            if let Some(item) = params.get("item")
                && let (Some(id), Some(kind)) = (
                    item.get("itemId").and_then(Value::as_str),
                    item.get("kind").and_then(Value::as_str),
                )
            {
                items.insert(id.to_owned(), kind.to_owned());
            }
        }
        "item/delta" => {
            let Some(id) = params.get("itemId").and_then(Value::as_str) else {
                return;
            };
            let Some(delta) = params
                .get("delta")
                .and_then(Value::as_str)
                .filter(|text| !text.is_empty())
            else {
                return;
            };
            let field = params.get("field").and_then(Value::as_str);
            match items.get(id).map(String::as_str) {
                Some("agentMessage") if field.is_none() || field == Some("text") => {
                    let _ = events.send(Event::Text(delta.to_owned()));
                }
                Some("reasoning") => {
                    let _ = events.send(Event::Reasoning(delta.to_owned()));
                }
                _ => {}
            }
        }
        "item/completed" => {
            if let Some(id) = params.pointer("/item/itemId").and_then(Value::as_str) {
                items.remove(id);
            }
        }
        "turn/modelAttemptStarted" => {
            let _ = events.send(Event::Began);
        }
        "session/tokenUsage" => {
            if let Some(usage) = token_usage(params) {
                let _ = events.send(Event::Usage(usage));
            }
        }
        "turn/completed" => {
            items.clear();
            *ended = Some(match params.get("terminal").and_then(Value::as_str) {
                Some("completed") => Event::Done(Ok(FinishReason::Stop)),
                Some("maxTokens" | "max_tokens" | "length") => {
                    Event::Done(Ok(FinishReason::Length))
                }
                Some("cancelled") => Event::Stopped,
                _ => Event::Done(Err(params
                    .pointer("/error/message")
                    .or_else(|| params.get("reason"))
                    .and_then(Value::as_str)
                    .unwrap_or("the turn failed")
                    .to_owned())),
            });
        }
        _ => {}
    }
}

/// Refuses unexpected server requests (`userInput/request` or non-Rig approvals).
fn decide(method: &str, params: &Value) -> Value {
    match method {
        "approval/request" => {
            let tool = params
                .pointer("/subject/tool")
                .or_else(|| params.pointer("/subject/toolName"))
                .and_then(Value::as_str);
            let ours = tool.is_some_and(|name| {
                name.starts_with(&format!("mcp__{SERVER}__"))
                    || name.starts_with(&format!("mcp__{SERVER}."))
            });
            let wanted = if ours { "approved" } else { "denied" };
            let choice = params
                .get("availableChoices")
                .and_then(Value::as_array)
                .and_then(|choices| {
                    choices.iter().find(|choice| {
                        choice.get("decision").and_then(Value::as_str) == Some(wanted)
                    })
                })
                .and_then(|choice| choice.get("choiceId"))
                .and_then(Value::as_str);
            match choice {
                Some(choice_id) => json!({ "result": { "choiceId": choice_id } }),
                None => json!({ "error": { "code": -32000, "message": "approval declined" } }),
            }
        }
        _ => method_not_found(method),
    }
}

/// Parses a `session/tokenUsage` notification into Rig's [`Usage`].
pub(crate) fn token_usage(params: &Value) -> Option<Usage> {
    let raw = params.get("usage")?;
    let count = |obj: &Value, key: &str| obj.get(key).and_then(Value::as_u64);
    let mut usage = Usage::new();
    usage.input_tokens = count(params, "promptTokens").or_else(|| count(raw, "inputTokens"));
    usage.output_tokens = count(raw, "outputTokens");
    usage.cached_input_tokens =
        count(raw, "cacheReadTokens").or_else(|| count(raw, "cachedTokens"));
    usage.cache_creation_input_tokens = count(raw, "cacheWriteTokens");
    usage.reasoning_tokens = count(raw, "reasoningTokens");
    usage.total_tokens = count(params, "totalTokens").or_else(|| {
        usage
            .input_tokens
            .zip(usage.output_tokens)
            .map(|(input, output)| input + output)
    });
    Some(usage)
}

/// A prompt's parts as a turn's input.
fn input(parts: &[Part]) -> Vec<Value> {
    parts
        .iter()
        .map(|part| match part {
            Part::Text(text) => json!({ "type": "text", "text": text }),
            Part::Image { media_type, data } => json!({
                "type": "image",
                "mediaType": media_type,
                "base64Data": data,
            }),
        })
        .collect()
}

/// Runs one Muse MSP session, a new one or the saved one, from
/// `initialize` through every prompt. A steer goes into the turn being
/// answered, and a stop interrupts it.
pub(crate) async fn converse<R, W>(
    wire: &mut Wire<R, W>,
    opening: Opening,
    mut prompts: mpsc::UnboundedReceiver<Prompt>,
    mut steers: mpsc::UnboundedReceiver<Steer>,
    mut stops: mpsc::UnboundedReceiver<u64>,
) -> Result<(), String>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    let session_id = tokio::time::timeout(HANDSHAKE_TIMEOUT, async {
        let init = wire
            .request(
                "initialize",
                json!({
                    "clientInfo": { "name": "rig", "version": env!("CARGO_PKG_VERSION") },
                    "capabilities": {
                        "requestedCapabilities": ["sessionMcp"],
                        "userInputDialogs": false,
                    },
                }),
            )
            .await
            .map_err(|why| format!("it did not start: {why}"))?;
        let has_mcp = init
            .get("grantedCapabilities")
            .and_then(Value::as_array)
            .is_some_and(|caps| caps.iter().any(|cap| cap.as_str() == Some("sessionMcp")));
        if !has_mcp {
            return Err("muse did not grant the `sessionMcp` capability".to_owned());
        }
        wire.notify("initialized", json!({})).await?;

        let mut config = json!({
            "mcpServers": {
                SERVER: {
                    "transport": "streamableHttp",
                    "url": opening.mcp_url,
                    "headers": {
                        "Authorization": format!("Bearer {}", opening.mcp_token),
                    },
                    "mode": "required",
                },
            },
        });
        if !opening.system.trim().is_empty()
            && let Some(obj) = config.as_object_mut()
        {
            obj.insert("developerPrompt".to_owned(), opening.system.into());
        }
        let mut session = json!({
            "commandId": uuidv7(),
            "modelId": opening.model,
            "config": config,
        });
        let open = match (&opening.resume, session.as_object_mut()) {
            (Some(saved), Some(session)) => {
                session.insert("sessionId".to_owned(), saved.as_str().into());
                // Its history is the program's to keep, not ours to read.
                session.insert("excludeItems".to_owned(), true.into());
                "session/resume"
            }
            (None, Some(session)) => {
                session.insert("approvalMode".to_owned(), "allowAll".into());
                "session/start"
            }
            (_, None) => "session/start",
        };
        let started = wire.request(open, session).await.map_err(|why| {
            format!("it opened no session (run `muse login` if signed out): {why}")
        })?;
        started
            .pointer("/session/sessionId")
            .or_else(|| started.get("sessionId"))
            .and_then(Value::as_str)
            .ok_or_else(|| "it opened a session without an id".to_owned())
            .map(str::to_owned)
    })
    .await
    .map_err(|_| "it did not finish starting in time".to_owned())??;

    let _ = wire.events.send(Event::Saved(session_id.clone()));

    // The number of the prompt being answered, and of the last one stopped.
    let (mut turn, mut stopped) = (0u64, 0u64);
    loop {
        // A prompt is taken before a stop, which may be the one for it. A
        // stop with no turn running is for one that is over, or for a
        // prompt that is only now on its way: that one is not started on.
        let prompt = loop {
            tokio::select! {
                biased;
                prompt = prompts.recv() => break prompt,
                Some(steer) = steers.recv() => {
                    let _ = steer.taken.send(Err("no turn is running".to_owned()));
                }
                Some(stop) = stops.recv() => stopped = stopped.max(stop),
                msg = wire.next_message() => wire.handle(msg?).await?,
            }
        };
        let Some(prompt) = prompt else {
            return Ok(());
        };
        turn += 1;
        if turn <= stopped {
            let _ = wire.events.send(Event::Stopped);
            continue;
        }
        let mut params = json!({
            "commandId": uuidv7(),
            "sessionId": session_id,
            "input": input(&prompt.parts),
        });
        if let (Some(effort), Some(obj)) = (prompt.effort, params.as_object_mut()) {
            obj.insert("reasoningEffort".to_owned(), effort.as_str().into());
        }
        wire.ended = None;
        let started = wire
            .request("turn/start", params)
            .await
            .map_err(|why| format!("the turn did not start: {why}"))?;
        let done = loop {
            if let Some(done) = wire.ended.take() {
                break done;
            }
            tokio::select! {
                msg = wire.next_message() => wire.handle(msg?).await?,
                Some(steer) = steers.recv() => {
                    let taken = match started.get("turnId").and_then(Value::as_str) {
                        Some(turn) => {
                            let steered = json!({
                                "commandId": uuidv7(),
                                "sessionId": session_id,
                                "expectedTurnId": turn,
                                "input": input(&steer.parts),
                            });
                            wire.request("turn/steer", steered).await.map(drop)
                        }
                        None => Err("the turn has no id".to_owned()),
                    };
                    let _ = steer.taken.send(taken);
                }
                Some(stop) = stops.recv() => {
                    if stop != turn {
                        continue;
                    }
                    // The turn then completes as cancelled.
                    let mut asked = json!({ "commandId": uuidv7(), "sessionId": session_id });
                    if let (Some(id), Some(asked)) = (started.get("turnId"), asked.as_object_mut())
                    {
                        asked.insert("turnId".to_owned(), id.clone());
                    }
                    let _ = wire.request("turn/interrupt", asked).await;
                }
            }
        };
        if wire.events.send(done).is_err() {
            return Ok(());
        }
    }
}

/// The model ids `muse exec --list-models` prints, skipping diagnostic lines.
pub(crate) fn models(listing: &str) -> Vec<String> {
    let mut ids = Vec::new();
    for line in listing.lines() {
        let id = line.trim();
        if id.is_empty() || id.contains(char::is_whitespace) || id.contains(':') {
            continue;
        }
        if !ids.iter().any(|seen| seen == id) {
            ids.push(id.to_owned());
        }
    }
    ids
}
