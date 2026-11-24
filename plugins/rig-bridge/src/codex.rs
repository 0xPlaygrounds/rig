//! Codex as a model, over `codex app-server`: one process and one thread
//! for a whole conversation. The program runs on a home directory of its
//! own, so it finds no settings, instructions, hooks or MCP servers of the
//! user's, with each of its tools and features switched off, Rig's system
//! prompt as its base instructions, and Rig's MCP server as its only one.
//! Only its sign-in is the user's.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use base64::Engine;
use rig_harness::rig_core::catalog::ReasoningSupport;
use rig_harness::rig_core::completion::{FinishReason, Usage};
use serde_json::{Value, json};
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::sync::mpsc;

use crate::harness::{
    ContextSnapshot, Event, HANDSHAKE_TIMEOUT, Harness, Launch, Part, Prompt, Running, Stderr,
    Steer, effort_wire, env_path, link, note, parse_effort, spawn,
};
use crate::rpc::{Transport, method_not_found, positive_u32};

#[cfg(test)]
mod tests;

/// The MCP server name Rig's tools are under: the model sees a tool as
/// `mcp__rig.<name>`.
const SERVER: &str = "rig";

/// The variable the program reads the MCP server's bearer token from.
const TOKEN_ENV: &str = "RIG_MCP_TOKEN";

/// How long the program waits for one of Rig's tools, in seconds.
const TOOL_TIMEOUT_SECS: u64 = 86_400;

/// The features of Codex's own that a model does not need.
const DISABLED: &[&str] = &[
    "apps",
    "plugins",
    "hooks",
    "shell_tool",
    "unified_exec",
    "view_image",
    "image_generation",
    "browser_use",
    "browser_use_external",
    "browser_use_full_cdp_access",
    "browser_annotation_api",
    "computer_use",
    "in_app_browser",
    "in_app_chat",
    "in_app_dictation",
    "in_app_local_automation",
    "in_app_updates",
    "in_app_voice",
    "daemon_auto_start",
    "sleep_tool",
    "multi_agent",
    "multi_agent_v2",
    "code_mode_host",
    "memories",
    "goals",
    "tool_suggest",
    "skill_search",
    "skill_mcp_dependency_install",
    "remote_plugin",
    "plugin_sharing",
    "workspace_dependencies",
    "shell_snapshot",
];

/// Settings that leave out what Codex adds to a conversation by itself.
/// Each value is TOML.
const SETTINGS: &[&str] = &[
    "web_search=\"disabled\"",
    "project_doc_max_bytes=0",
    "include_apps_instructions=false",
    "include_collaboration_mode_instructions=false",
    "include_environment_context=false",
    "include_permissions_instructions=false",
    "skills.include_instructions=false",
    "skills.bundled.enabled=false",
    "tools.experimental_request_user_input.enabled=false",
    "analytics.enabled=false",
    "history.persistence=\"none\"",
    "check_for_update_on_startup=false",
    "thread_unload_delay_secs=0",
];

/// A model the program lists.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Listed {
    /// The model's id.
    pub id: String,
    /// The name it is shown under.
    pub name: String,
    /// Its context window, in tokens.
    pub context_window: Option<u32>,
    /// The reasoning effort levels the model advertises.
    pub reasoning: ReasoningSupport,
}

/// Codex, with its model catalog rewritten so that no model brings tools.
pub(crate) struct Codex {
    catalog: Arc<str>,
}

impl Codex {
    /// Reads what `codex debug models` printed: the models to list, and
    /// Codex for them. `None` when the listing is not a catalog.
    pub(crate) fn new(listing: &str) -> Option<(Self, Vec<Listed>)> {
        let mut catalog: Value = serde_json::from_str(listing).ok()?;
        let models = catalog.get_mut("models")?.as_array_mut()?;
        let mut listed = Vec::new();
        for model in models.iter_mut() {
            let model = model.as_object_mut()?;
            // A model's catalog entry decides which tools it is sent, and
            // no setting overrides it: the code-running wrapper around MCP
            // tools, the multi-agent tools, patching and tool search.
            model.insert("tool_mode".to_owned(), "direct".into());
            model.insert("multi_agent_version".to_owned(), Value::Null);
            model.insert("experimental_supported_tools".to_owned(), json!([]));
            model.insert("apply_patch_tool_type".to_owned(), Value::Null);
            model.insert("supports_search_tool".to_owned(), false.into());
            let text = |key: &str| model.get(key).and_then(Value::as_str);
            if let (Some(id), Some("list")) = (text("slug"), text("visibility")) {
                let window = model
                    .get("context_window")
                    .and_then(Value::as_u64)
                    .map(|raw| {
                        match model
                            .get("effective_context_window_percent")
                            .and_then(Value::as_u64)
                        {
                            Some(pct @ 1..=100) => raw.saturating_mul(pct) / 100,
                            _ => raw,
                        }
                    });
                let mut levels = Vec::new();
                if let Some(supported) = model
                    .get("supported_reasoning_levels")
                    .and_then(Value::as_array)
                {
                    for entry in supported {
                        if let Some(effort) = entry
                            .get("effort")
                            .and_then(Value::as_str)
                            .and_then(parse_effort)
                            && !levels.contains(&effort)
                        {
                            levels.push(effort);
                        }
                    }
                }
                let default = text("default_reasoning_level").and_then(parse_effort);
                let reasoning = if levels.is_empty() {
                    ReasoningSupport::Unknown { default }
                } else {
                    ReasoningSupport::Listed {
                        levels,
                        budget: None,
                        can_disable: false,
                        default,
                    }
                };
                listed.push(Listed {
                    id: id.to_owned(),
                    name: text("display_name").unwrap_or(id).to_owned(),
                    context_window: window.and_then(|tokens| u32::try_from(tokens).ok()),
                    reasoning,
                });
            }
        }
        let codex = Self {
            catalog: catalog.to_string().into(),
        };
        Some((codex, listed))
    }
}

/// Where the user's sign-in is kept.
pub(crate) fn signed_in() -> Option<PathBuf> {
    let home = env_path("CODEX_HOME").or_else(|| Some(env_path("HOME")?.join(".codex")))?;
    Some(home.join("auth.json"))
}

impl Harness for Codex {
    fn note(&self, _: Option<&Path>) -> String {
        note(&format!("mcp__{SERVER}.read"), "", None)
    }

    /// Codex 0.171 disables `mcp_tool_list_changed` by default, so waiting on
    /// `tools/list` after `notifications/tools/list_changed` would time out.
    fn relists(&self) -> bool {
        false
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
        // The program's home, where it saves the thread.
        let home = dir.join("codex");
        let catalog = dir.join("models.json");
        std::fs::create_dir_all(&home)
            .and_then(|()| std::fs::write(&catalog, &*self.catalog))
            .map_err(|why| format!("could not write the session's files: {why}"))?;
        // The sign-in is shared, so a token the program renews is renewed
        // for the user's own sessions too.
        if let Some(auth) = signed_in().filter(|auth| auth.is_file()) {
            link(&auth, &home.join("auth.json"))
                .map_err(|why| format!("could not reach the sign-in: {why}"))?;
        }
        command.current_dir(dir).arg("app-server");
        for feature in DISABLED {
            command.args(["--disable", feature]);
        }
        let server = format!("mcp_servers.{SERVER}");
        let settings = SETTINGS.iter().map(|setting| (*setting).to_owned()).chain([
            format!("model_catalog_json={}", json!(catalog.to_string_lossy())),
            format!("{server}.url={}", json!(mcp.url())),
            format!("{server}.bearer_token_env_var={}", json!(TOKEN_ENV)),
            format!("{server}.tool_timeout_sec={TOOL_TIMEOUT_SECS}"),
            // Rig gates its own tools.
            format!("{server}.default_tools_approval_mode=\"approve\""),
        ]);
        for setting in settings {
            command.args(["-c", &setting]);
        }
        command
            .env("CODEX_HOME", &home)
            .env(TOKEN_ENV, mcp.token())
            // The sign-in answers, not an API key.
            .env_remove("OPENAI_API_KEY")
            .env_remove("OPENAI_BASE_URL");
        let (process, pipes) = spawn(command)?;
        let (prompts, prompted) = mpsc::unbounded_channel();
        let (steers, steered) = mpsc::unbounded_channel();
        let (stops, stopped) = mpsc::unbounded_channel();
        let (events, reported) = mpsc::unbounded_channel();
        let stderr = Stderr::follow(pipes.stderr);
        let opening = Opening {
            dir: dir.to_path_buf(),
            system: system.to_owned(),
            model: model.to_owned(),
            resume: resume.map(str::to_owned),
        };
        tokio::spawn(async move {
            let mut wire = Wire::new(pipes.stdout, pipes.stdin, events.clone());
            let conversed = converse(&mut wire, opening, prompted, steered, stopped);
            if let Err(why) = conversed.await {
                let why = stderr.explain(&format!("codex: {why}")).await;
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

/// What a thread opens with.
pub(crate) struct Opening {
    /// The session's private directory, the program's working directory.
    pub dir: PathBuf,
    /// The system prompt.
    pub system: String,
    /// The model's id.
    pub model: String,
    /// The thread a process before this one saved, to go on from.
    pub resume: Option<String>,
}

/// The client's end of the connection: JSON lines, each request answered
/// under its id. The program's own requests carry a method and ids of
/// their own numbering.
pub(crate) struct Wire<R, W> {
    transport: Transport<R, W>,
    events: mpsc::UnboundedSender<Event>,
    /// Whether the program reached Rig's MCP server, once it said.
    server: Option<Result<(), String>>,
    /// How the turn being answered ended, once it did.
    ended: Option<Event>,
}

impl<R: AsyncRead + Unpin, W: AsyncWrite + Unpin> Wire<R, W> {
    pub(crate) fn new(reader: R, writer: W, events: mpsc::UnboundedSender<Event>) -> Self {
        Self {
            transport: Transport::new(reader, writer),
            events,
            server: None,
            ended: None,
        }
    }

    async fn write(&mut self, message: &Value) -> Result<(), String> {
        self.transport
            .write(message)
            .await
            .map_err(|()| "the program exited".to_owned())
    }

    /// Reads the next JSON object. Cancel-safe for use in `tokio::select!`.
    async fn next_message(&mut self) -> Result<Value, String> {
        self.transport.next_message().await
    }

    /// Handles one incoming notification or server request.
    async fn handle(&mut self, message: Value) -> Result<(), String> {
        let events = &self.events;
        let server = &mut self.server;
        let ended = &mut self.ended;
        let _ = self
            .transport
            .dispatch(message, None, false, decide, |method, params| {
                notify(events, server, ended, method, params);
            })
            .await?;
        Ok(())
    }

    /// Sends a request and waits for its result.
    async fn request(&mut self, method: &str, params: Value) -> Result<Value, String> {
        let events = &self.events;
        let server = &mut self.server;
        let ended = &mut self.ended;
        self.transport
            .request(method, params, false, decide, |method, params| {
                notify(events, server, ended, method, params);
            })
            .await
    }
}

/// The answer to a request of the program's: yes to a call of one of
/// Rig's tools, and a refusal of anything else.
fn decide(method: &str, params: &Value) -> Value {
    let ours = params.get("serverName").and_then(Value::as_str) == Some(SERVER);
    if method == "mcpServer/elicitation/request" && ours {
        json!({ "result": { "action": "accept", "content": null } })
    } else {
        method_not_found(method)
    }
}

/// Reports what a notification says.
fn notify(
    events: &mpsc::UnboundedSender<Event>,
    server: &mut Option<Result<(), String>>,
    ended: &mut Option<Event>,
    method: &str,
    params: &Value,
) {
    let delta = || {
        params
            .get("delta")
            .and_then(Value::as_str)
            .filter(|delta| !delta.is_empty())
            .map(str::to_owned)
    };
    let event = match method {
        "item/agentMessage/delta" => delta().map(Event::Text),
        "item/reasoning/summaryTextDelta" | "item/reasoning/textDelta" => {
            delta().map(Event::Reasoning)
        }
        // A new part of a reasoning summary is a new paragraph.
        "item/reasoning/summaryPartAdded" => (params.get("summaryIndex").and_then(Value::as_u64)
            > Some(0))
        .then(|| Event::Reasoning("\n\n".to_owned())),
        "thread/tokenUsage/updated" => params.get("tokenUsage").and_then(|counted| {
            if let Some(window) = positive_u32(counted.get("modelContextWindow")) {
                let _ = events.send(Event::Context(ContextSnapshot {
                    used: None,
                    window: Some(window),
                }));
            }
            counted.get("last").map(|last| Event::Usage(usage(last)))
        }),
        "mcpServer/startupStatus/updated" => {
            if params.get("name").and_then(Value::as_str) == Some(SERVER) {
                match params.get("status").and_then(Value::as_str) {
                    Some("ready") => *server = Some(Ok(())),
                    Some("failed") => {
                        let why = params.get("error").and_then(Value::as_str);
                        *server = Some(Err(why.unwrap_or("it failed").to_owned()));
                    }
                    _ => {}
                }
            }
            None
        }
        "turn/completed" => {
            let turn = params.get("turn").unwrap_or(&Value::Null);
            *ended = Some(match turn.get("status").and_then(Value::as_str) {
                Some("completed") => Event::Done(Ok(FinishReason::Stop)),
                Some("interrupted") => Event::Stopped,
                _ => Event::Done(Err(turn
                    .pointer("/error/message")
                    .and_then(Value::as_str)
                    .unwrap_or("the turn failed")
                    .to_owned())),
            });
            None
        }
        _ => None,
    };
    if let Some(event) = event {
        let _ = events.send(event);
    }
}

/// The counters of one model call. Its input counts what was read from
/// the cache.
fn usage(last: &Value) -> Usage {
    let count = |name: &str| last.get(name).and_then(Value::as_u64);
    let mut usage = Usage::new();
    usage.input_tokens = count("inputTokens");
    usage.output_tokens = count("outputTokens");
    usage.cached_input_tokens = count("cachedInputTokens");
    usage.cache_creation_input_tokens = count("cacheWriteInputTokens");
    usage.reasoning_tokens = count("reasoningOutputTokens");
    usage.total_tokens = usage
        .input_tokens
        .zip(usage.output_tokens)
        .map(|(input, output)| input + output);
    usage
}

/// Maps a MIME type to a safe filename extension for `localImage`.
pub(crate) fn image_extension(media_type: &str) -> &str {
    let subtype = media_type
        .split(';')
        .next()
        .unwrap_or_default()
        .trim()
        .split_once('/')
        .map(|(_, sub)| sub)
        .unwrap_or_default();
    match subtype {
        "png" => "png",
        "jpeg" | "jpg" => "jpg",
        "gif" => "gif",
        "webp" => "webp",
        other if !other.is_empty() && other.bytes().all(|b| b.is_ascii_alphanumeric()) => other,
        _ => "png",
    }
}

/// A prompt as a turn's input. An image is a file in `dir`, numbered from
/// `images`.
fn input(prompt: &[Part], dir: &Path, images: &mut usize) -> Result<Vec<Value>, String> {
    let mut items = Vec::new();
    for part in prompt {
        items.push(match part {
            Part::Text(text) => json!({ "type": "text", "text": text }),
            Part::Image { media_type, data } => {
                let bytes = base64::engine::general_purpose::STANDARD.decode(data);
                let bytes = bytes.map_err(|why| format!("an image is not base 64: {why}"))?;
                let kind = image_extension(media_type);
                *images += 1;
                // A process that went on from a saved thread counts from 1
                // again: its files are told apart from the ones before.
                let id = uuid::Uuid::new_v4().simple();
                let path = dir.join(format!("image-{images}-{id}.{kind}"));
                std::fs::write(&path, bytes)
                    .map_err(|why| format!("could not write an image: {why}"))?;
                json!({ "type": "localImage", "path": path.to_string_lossy() })
            }
        });
    }
    Ok(items)
}

/// Opens the thread, a new one or the saved one, then answers each prompt
/// until the prompts end. A steer goes into the turn being answered, and a
/// stop interrupts it.
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
    let thread = tokio::time::timeout(HANDSHAKE_TIMEOUT, async {
        let hello = json!({
            "clientInfo": { "name": "rig", "title": "Rig", "version": env!("CARGO_PKG_VERSION") },
            "capabilities": {},
        });
        wire.request("initialize", hello).await?;
        wire.write(&json!({ "method": "initialized" })).await?;
        let account = wire
            .request("account/read", json!({ "refreshToken": false }))
            .await?;
        if account.get("account").is_none_or(Value::is_null) {
            return Err("it is not signed in: run `codex login`".to_owned());
        }
        let mut thread = json!({
            "model": opening.model,
            "cwd": opening.dir.to_string_lossy(),
            // Nothing of Codex's own is left to approve or to sandbox.
            "approvalPolicy": "never",
            "sandbox": "read-only",
            "baseInstructions": opening.system,
        });
        let open = match (&opening.resume, thread.as_object_mut()) {
            (Some(saved), Some(thread)) => {
                thread.insert("threadId".to_owned(), saved.as_str().into());
                // Its turns are the program's to keep, not ours to read.
                thread.insert("excludeTurns".to_owned(), true.into());
                "thread/resume"
            }
            _ => "thread/start",
        };
        let started = wire.request(open, thread).await?;
        let thread = started
            .pointer("/thread/id")
            .and_then(Value::as_str)
            .ok_or("it started a thread without an id")?
            .to_owned();
        // A turn started before the server is reached runs without Rig's tools.
        loop {
            if let Some(reached) = wire.server.take() {
                reached.map_err(|why| format!("it did not reach rig's tools: {why}"))?;
                break;
            }
            let msg = wire.next_message().await?;
            wire.handle(msg).await?;
        }
        Ok::<_, String>(thread)
    })
    .await
    .map_err(|_| "it did not finish starting in time".to_owned())??;

    let _ = wire.events.send(Event::Saved(thread.clone()));

    let mut images = 0;
    // The number of the prompt being answered, and of the last one stopped.
    let (mut number, mut stopped) = (0u64, 0u64);
    loop {
        // Between prompts the program still sends notifications. A prompt
        // is taken before a stop, which may be the one for it. A stop with
        // no turn running is for one that is over, or for a prompt that is
        // only now on its way: that one is not started on.
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
        number += 1;
        if number <= stopped {
            let _ = wire.events.send(Event::Stopped);
            continue;
        }
        let mut turn = json!({
            "threadId": thread,
            "input": input(&prompt.parts, &opening.dir, &mut images)?,
            "summary": "auto",
        });
        if let (Some(effort), Some(turn)) = (prompt.effort, turn.as_object_mut()) {
            turn.insert("effort".to_owned(), effort_wire(effort, true).into());
        }
        wire.ended = None;
        let done = match wire.request("turn/start", turn).await {
            Ok(started) => loop {
                if let Some(ended) = wire.ended.take() {
                    break ended;
                }
                tokio::select! {
                    msg = wire.next_message() => wire.handle(msg?).await?,
                    Some(steer) = steers.recv() => {
                        let id = started.pointer("/turn/id").and_then(Value::as_str);
                        let input = input(&steer.parts, &opening.dir, &mut images);
                        let taken = match (id, input) {
                            (Some(id), Ok(input)) => {
                                let steered = json!({
                                    "threadId": thread,
                                    "expectedTurnId": id,
                                    "input": input,
                                });
                                wire.request("turn/steer", steered).await.map(drop)
                            }
                            (None, _) => Err("the turn has no id".to_owned()),
                            (_, Err(why)) => Err(why),
                        };
                        let _ = steer.taken.send(taken);
                    }
                    Some(stop) = stops.recv() => {
                        // The turn then completes as interrupted.
                        if let (true, Some(id)) = (stop == number, started.pointer("/turn/id")) {
                            let interrupted = json!({ "threadId": thread, "turnId": id });
                            let _ = wire.request("turn/interrupt", interrupted).await;
                        }
                    }
                }
            },
            Err(why) => Event::Done(Err(why)),
        };
        let _ = wire.events.send(done);
    }
}
