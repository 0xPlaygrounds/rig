//! Meta AI Muse (`muse`) as a model, over `muse serve` (MSP JSON-RPC).
//! The program runs with private XDG directories (sharing only `auth.json`),
//! `--no-session-log`, `--preset empty`, and workspace tools disabled, so it
//! has none of its own tools, instructions, memory or skills. Rig's system
//! prompt is passed via `session/start` `config.developerPrompt` and Rig's
//! MCP server via `config.mcpServers`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use rig_harness::rig_core::completion::{Effort, FinishReason, Usage};
use serde_json::{Value, json};
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::sync::mpsc;

use crate::harness::{
    Event, HANDSHAKE_TIMEOUT, Harness, Launch, Part, Running, Stderr, link, note, spawn,
};
use crate::rpc::Transport;

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
    let config = std::env::var_os("XDG_CONFIG_HOME")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .or_else(|| Some(PathBuf::from(std::env::var_os("HOME")?).join(".config")))?;
    Some(config.join("muse").join("auth.json"))
}

impl Harness for Muse {
    fn note(&self, project: Option<&Path>) -> String {
        note(&format!("mcp__{SERVER}.read"), "", Some(project))
    }

    fn start(&self, launch: Launch<'_>) -> Result<Running, String> {
        let Launch {
            mut command,
            dir,
            system,
            model,
            effort,
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
                "--no-session-log",
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
        for name in REMOVED_ENV {
            command.env_remove(name);
        }
        let (process, pipes) = spawn(command)?;
        let (prompts, prompted) = mpsc::unbounded_channel();
        let (events, reported) = mpsc::unbounded_channel();
        let stderr = Stderr::follow(pipes.stderr);
        let opening = Opening {
            system: system.to_owned(),
            model: model.to_owned(),
            effort,
            mcp_url: mcp.url().to_owned(),
            mcp_token: mcp.token().to_owned(),
        };
        tokio::spawn(async move {
            let mut wire = Wire::new(pipes.stdout, pipes.stdin, events.clone());
            if let Err(why) = converse(&mut wire, opening, prompted).await {
                let why = stderr.explain(&format!("muse: {why}")).await;
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

/// What a Muse MSP session opens with.
pub(crate) struct Opening {
    pub system: String,
    pub model: String,
    pub effort: Option<Effort>,
    pub mcp_url: String,
    pub mcp_token: String,
}

/// An effort level as Muse MSP names it.
fn level(effort: Effort) -> &'static str {
    match effort {
        Effort::Minimal => "minimal",
        Effort::Low => "low",
        Effort::Medium => "medium",
        Effort::High => "high",
        Effort::XHigh => "xhigh",
        _ => "max",
    }
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
    ended: Option<Result<FinishReason, String>>,
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
        self.transport
            .next_object()
            .await
            .map_err(|()| "the program exited".to_owned())
    }

    async fn handle(&mut self, message: Value, id: Option<u64>) -> Result<Option<Value>, String> {
        let Some(method) = message.get("method").and_then(Value::as_str) else {
            let answers = id.is_some() && message.get("id").and_then(Value::as_u64) == id;
            return Ok(answers.then_some(message));
        };
        let params = message.get("params").unwrap_or(&Value::Null);
        match message.get("id").filter(|id| !id.is_null()) {
            Some(asked) => {
                self.transport
                    .reply(asked, decide(method, params), true)
                    .await
                    .map_err(|()| "the program exited".to_owned())?;
            }
            None => self.notified(method, params),
        }
        Ok(None)
    }

    async fn request(&mut self, method: &str, params: Value) -> Result<Value, String> {
        let id = self
            .transport
            .send_request(method, params, true)
            .await
            .map_err(|()| "the program exited".to_owned())?;
        loop {
            let msg = self.next_message().await?;
            let Some(message) = self.handle(msg, Some(id)).await? else {
                continue;
            };
            return match message.get("error") {
                Some(error) => {
                    let text = error
                        .get("message")
                        .and_then(Value::as_str)
                        .unwrap_or("the request failed");
                    Err(text.to_owned())
                }
                None => Ok(message.get("result").cloned().unwrap_or(Value::Null)),
            };
        }
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

    fn notified(&mut self, method: &str, params: &Value) {
        match method {
            "item/started" => {
                if let Some(item) = params.get("item")
                    && let (Some(id), Some(kind)) = (
                        item.get("itemId").and_then(Value::as_str),
                        item.get("kind").and_then(Value::as_str),
                    )
                {
                    self.items.insert(id.to_owned(), kind.to_owned());
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
                match self.items.get(id).map(String::as_str) {
                    Some("agentMessage") if field.is_none() || field == Some("text") => {
                        let _ = self.events.send(Event::Text(delta.to_owned()));
                    }
                    Some("reasoning") => {
                        let _ = self.events.send(Event::Reasoning(delta.to_owned()));
                    }
                    _ => {}
                }
            }
            "item/completed" => {
                if let Some(id) = params.pointer("/item/itemId").and_then(Value::as_str) {
                    self.items.remove(id);
                }
            }
            "turn/modelAttemptStarted" => {
                let _ = self.events.send(Event::Began);
            }
            "session/tokenUsage" => {
                if let Some(usage) = token_usage(params) {
                    let _ = self.events.send(Event::Usage(usage));
                }
            }
            "turn/completed" => {
                self.items.clear();
                let outcome = match params.get("terminal").and_then(Value::as_str) {
                    Some("completed") => Ok(FinishReason::Stop),
                    Some("maxTokens" | "max_tokens" | "length") => Ok(FinishReason::Length),
                    Some("cancelled") => Err("the turn was cancelled".to_owned()),
                    _ => {
                        let why = params
                            .pointer("/error/message")
                            .or_else(|| params.get("reason"))
                            .and_then(Value::as_str)
                            .unwrap_or("the turn failed");
                        Err(why.to_owned())
                    }
                };
                self.ended = Some(outcome);
            }
            _ => {}
        }
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
        _ => json!({ "error": { "code": -32601, "message": format!("no method `{method}`") } }),
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

/// Runs one Muse MSP session from `initialize` through every prompt.
pub(crate) async fn converse<R, W>(
    wire: &mut Wire<R, W>,
    opening: Opening,
    mut prompts: mpsc::UnboundedReceiver<Vec<Part>>,
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
        let started = wire
            .request(
                "session/start",
                json!({
                    "commandId": uuidv7(),
                    "modelId": opening.model,
                    "approvalMode": "allowAll",
                    "config": config,
                }),
            )
            .await
            .map_err(|why| {
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

    loop {
        let prompt = loop {
            tokio::select! {
                prompt = prompts.recv() => break prompt,
                msg = wire.next_message() => drop(wire.handle(msg?, None).await?),
            }
        };
        let Some(prompt) = prompt else {
            return Ok(());
        };
        let input: Vec<Value> = prompt
            .iter()
            .map(|part| match part {
                Part::Text(text) => json!({ "type": "text", "text": text }),
                Part::Image { media_type, data } => json!({
                    "type": "image",
                    "mediaType": media_type,
                    "base64Data": data,
                }),
            })
            .collect();
        let mut params = json!({
            "commandId": uuidv7(),
            "sessionId": session_id,
            "input": input,
        });
        if let (Some(effort), Some(obj)) = (opening.effort, params.as_object_mut()) {
            obj.insert("reasoningEffort".to_owned(), level(effort).into());
        }
        wire.ended = None;
        wire.request("turn/start", params)
            .await
            .map_err(|why| format!("the turn did not start: {why}"))?;
        let done = loop {
            if let Some(done) = wire.ended.take() {
                break done;
            }
            let msg = wire.next_message().await?;
            wire.handle(msg, None).await?;
        };
        if wire.events.send(Event::Done(done)).is_err() {
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
