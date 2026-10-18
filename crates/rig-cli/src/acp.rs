//! `rig acp`: an Agent Client Protocol (v1) agent on stdin and stdout, for
//! editors such as Zed. It is a client of `rigd` like every other front:
//! an ACP session is a rig session (the same id), `session/prompt` sends
//! the prompt and blocks until the turn ends, the session's events become
//! `session/update` notifications, and an `ask.opened` becomes a
//! `session/request_permission` request whose answer goes back as
//! `rig/session/answer`. Frames are JSON-RPC 2.0, one per line; stdout
//! carries only frames.

use std::collections::{HashMap, HashSet};
use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use rig_client::{Client, ClientError};
use rig_protocol::methods::{self, WaitStatus};
use rig_protocol::{AskId, EventRecord, Key, Seq, SessionId};
use serde_json::{Value, json};

use crate::out::Failure;
use crate::verbs::Ctx;

/// The ACP version this adapter speaks.
pub const PROTOCOL_VERSION: u64 = 1;

/// The answer to a free-text question asked through an editor, which ACP
/// gives no way to answer: the user can reply in the next prompt.
const FREE_TEXT_REPLY: &str = "The editor can't answer free-text questions. Make the reasonable choice, say which, and the user can correct it in the next message.";

/// JSON-RPC error codes ACP uses.
const INVALID_PARAMS: i64 = -32602;
const METHOD_NOT_FOUND: i64 = -32601;
const INTERNAL: i64 = -32603;

/// What the threads share.
struct Shared {
    client: Client,
    out: Mutex<std::io::Stdout>,
    next_id: AtomicU64,
    /// Our `session/request_permission` requests awaiting the editor:
    /// by request id, the session and ask they answer and its options.
    asking: Mutex<HashMap<u64, Asking>>,
    /// Sessions whose events are being forwarded.
    following: Mutex<HashSet<String>>,
    /// Counts each session's resolved asks, so a prompt waiting out an ask
    /// wakes when it is answered.
    resolved: Mutex<HashMap<String, u64>>,
    /// The last event forwarded per session, so a prompt answers only
    /// after the updates of its turn are out.
    forwarded: Mutex<HashMap<String, Seq>>,
    changed: Condvar,
}

struct Asking {
    session: SessionId,
    ask: AskId,
    /// The ask's reject option, for an editor that cancels.
    reject: Option<String>,
}

impl Shared {
    fn send(&self, frame: &Value) {
        if let Ok(mut out) = self.out.lock() {
            writeln!(out, "{frame}").ok();
            out.flush().ok();
        }
    }

    fn reply(&self, id: &Value, result: Value) {
        self.send(&json!({"jsonrpc": "2.0", "id": id, "result": result}));
    }

    fn error(&self, id: &Value, code: i64, message: &str) {
        self.send(&json!({
            "jsonrpc": "2.0",
            "id": id,
            "error": {"code": code, "message": message},
        }));
    }

    fn update(&self, session: &str, update: Value) {
        self.send(&json!({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {"sessionId": session, "update": update},
        }));
    }
}

/// Runs the adapter until stdin ends.
pub fn run(ctx: &Ctx) -> Result<u8, Failure> {
    let client = ctx.connect()?;
    let shared = Arc::new(Shared {
        client,
        out: Mutex::new(std::io::stdout()),
        next_id: AtomicU64::new(1),
        asking: Mutex::new(HashMap::new()),
        following: Mutex::new(HashSet::new()),
        resolved: Mutex::new(HashMap::new()),
        forwarded: Mutex::new(HashMap::new()),
        changed: Condvar::new(),
    });
    let stdin = std::io::stdin();
    let mut line = String::new();
    let mut input = stdin.lock();
    loop {
        line.clear();
        match input.read_line(&mut line) {
            Ok(0) | Err(_) => break,
            Ok(_) => {}
        }
        if line.trim().is_empty() {
            continue;
        }
        let Ok(frame) = serde_json::from_str::<Value>(&line) else {
            shared.send(&json!({
                "jsonrpc": "2.0",
                "id": null,
                "error": {"code": -32700, "message": "not JSON"},
            }));
            continue;
        };
        handle(&shared, frame);
    }
    Ok(0)
}

fn handle(shared: &Arc<Shared>, frame: Value) {
    let id = frame.get("id").cloned();
    let method = frame
        .get("method")
        .and_then(Value::as_str)
        .map(str::to_owned);
    let params = frame.get("params").cloned().unwrap_or(Value::Null);
    match (method, id) {
        (Some(method), Some(id)) => {
            let shared = Arc::clone(shared);
            // A prompt blocks until its turn ends; every request gets its
            // own thread so the editor's cancel and answers still arrive.
            std::thread::spawn(move || match request(&shared, &method, &params) {
                Ok(result) => shared.reply(&id, result),
                Err((code, message)) => shared.error(&id, code, &message),
            });
        }
        (Some(method), None) => notification(shared, &method, &params),
        (None, Some(id)) => response(shared, &id, &frame),
        (None, None) => {}
    }
}

type Outcome = Result<Value, (i64, String)>;

fn rpc(error: ClientError) -> (i64, String) {
    let message = match &error {
        ClientError::Rpc(rpc) => rpc.message.clone(),
        other => other.to_string(),
    };
    (INTERNAL, message)
}

fn session_param(params: &Value) -> Result<SessionId, (i64, String)> {
    params
        .get("sessionId")
        .and_then(Value::as_str)
        .map(|id| SessionId(id.to_owned()))
        .ok_or((INVALID_PARAMS, "sessionId is missing".to_owned()))
}

fn request(shared: &Arc<Shared>, method: &str, params: &Value) -> Outcome {
    match method {
        "initialize" => Ok(json!({
            "protocolVersion": PROTOCOL_VERSION,
            "agentCapabilities": {
                "loadSession": true,
                "promptCapabilities": {"image": false, "audio": false, "embeddedContext": true},
            },
            "authMethods": [],
        })),
        "authenticate" => Ok(json!({})),
        "session/new" => {
            let cwd = params
                .get("cwd")
                .and_then(Value::as_str)
                .map(PathBuf::from)
                .ok_or((INVALID_PARAMS, "cwd is missing".to_owned()))?;
            let started = shared
                .client
                .call::<methods::Start>(&methods::Start {
                    cwd,
                    prompt: None,
                    name: None,
                    host: None,
                    build: crate::verbs::own_build(),
                    grant: None,
                    policy: None,
                    budget: None,
                    context: None,
                    unload: None,
                    view: false,
                    key: Key(uuid::Uuid::new_v4().to_string()),
                })
                .map_err(rpc)?;
            follow(shared, &started.session_id, started.seq);
            Ok(json!({"sessionId": started.session_id.0}))
        }
        "session/load" => {
            let id = session_param(params)?;
            // The history replays as updates before the answer, as ACP
            // asks; live ones follow.
            let history = follow(shared, &id, Seq(0));
            forwarded_through(shared, &id, history);
            Ok(Value::Null)
        }
        "session/prompt" => prompt(shared, params),
        _ => Err((METHOD_NOT_FOUND, format!("rig acp has no method {method}"))),
    }
}

fn notification(shared: &Arc<Shared>, method: &str, params: &Value) {
    if method == "session/cancel"
        && let Ok(id) = session_param(params)
    {
        shared
            .client
            .call::<methods::Stop>(&methods::Stop {
                session_id: id,
                scope: methods::StopScope::Turn,
                key: Key(uuid::Uuid::new_v4().to_string()),
            })
            .ok();
    }
}

/// The editor's answer to one of our permission requests.
fn response(shared: &Arc<Shared>, id: &Value, frame: &Value) {
    let Some(asking) = id
        .as_u64()
        .and_then(|id| shared.asking.lock().ok()?.remove(&id))
    else {
        return;
    };
    let outcome = frame.pointer("/result/outcome");
    let chosen = match outcome
        .and_then(|o| o.get("outcome"))
        .and_then(Value::as_str)
    {
        Some("selected") => outcome
            .and_then(|o| o.get("optionId"))
            .and_then(Value::as_str)
            .map(str::to_owned),
        // Cancelled, or an error: refuse.
        _ => asking.reject,
    };
    shared
        .client
        .call::<methods::Answer>(&methods::Answer {
            session_id: asking.session,
            ask_id: asking.ask,
            option_id: chosen,
            text: None,
            key: Key(uuid::Uuid::new_v4().to_string()),
        })
        .ok();
}

/// The text of an ACP prompt: its text blocks, and embedded resources'
/// text, in order.
pub fn prompt_text(blocks: &[Value]) -> String {
    let mut parts: Vec<String> = Vec::new();
    for block in blocks {
        match block.get("type").and_then(Value::as_str) {
            Some("text") => {
                if let Some(text) = block.get("text").and_then(Value::as_str) {
                    parts.push(text.to_owned());
                }
            }
            Some("resource") => {
                let resource = block.get("resource");
                let uri = resource.and_then(|r| r.get("uri")).and_then(Value::as_str);
                if let Some(text) = resource.and_then(|r| r.get("text")).and_then(Value::as_str) {
                    parts.push(format!("{}:\n{text}", uri.unwrap_or("resource")));
                }
            }
            Some("resource_link") => {
                if let Some(uri) = block.get("uri").and_then(Value::as_str) {
                    parts.push(format!("@{}", uri.trim_start_matches("file://")));
                }
            }
            _ => {}
        }
    }
    parts.join("\n\n")
}

/// `session/prompt`: sends the prompt, then waits until the turn ends,
/// waiting out each ask while the editor answers it.
fn prompt(shared: &Arc<Shared>, params: &Value) -> Outcome {
    let id = session_param(params)?;
    let blocks = params
        .get("prompt")
        .and_then(Value::as_array)
        .ok_or((INVALID_PARAMS, "prompt is missing".to_owned()))?;
    let sent = shared
        .client
        .call::<methods::Send>(&methods::Send {
            session_id: id.clone(),
            text: Some(prompt_text(blocks)),
            content: None,
            mode: None,
            unload: None,
            key: Key(uuid::Uuid::new_v4().to_string()),
        })
        .map_err(rpc)?;
    follow(shared, &id, Seq(0));
    loop {
        let seen = resolved(shared, &id);
        let result = shared
            .client
            .call::<methods::Wait>(&methods::Wait {
                session_id: id.clone(),
                after_command: Some(sent.command_seq),
                timeout_ms: None,
            })
            .map_err(rpc)?;
        if !matches!(
            result.status,
            WaitStatus::NeedsApproval | WaitStatus::NeedsAnswer | WaitStatus::TimedOut
        ) {
            forwarded_through(shared, &id, result.seq);
        }
        match result.status {
            WaitStatus::Answered => return Ok(json!({"stopReason": "end_turn"})),
            WaitStatus::Stopped => return Ok(json!({"stopReason": "cancelled"})),
            WaitStatus::Failed => {
                return Err((
                    INTERNAL,
                    result.error.unwrap_or_else(|| "the turn failed".to_owned()),
                ));
            }
            WaitStatus::TimedOut => {}
            WaitStatus::NeedsApproval | WaitStatus::NeedsAnswer => {
                wait_resolved(shared, &id, seen);
            }
        }
    }
}

/// Blocks until `id`'s events through `seq` went to the editor, at most a
/// few seconds (a stream that stalls longer has ended).
fn forwarded_through(shared: &Shared, id: &SessionId, seq: Seq) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    let Ok(mut forwarded) = shared.forwarded.lock() else {
        return;
    };
    while forwarded.get(&id.0).is_none_or(|last| *last < seq) {
        let left = deadline.saturating_duration_since(std::time::Instant::now());
        if left.is_zero() {
            return;
        }
        match shared.changed.wait_timeout(forwarded, left) {
            Ok((next, _)) => forwarded = next,
            Err(_) => return,
        }
    }
}

fn resolved(shared: &Shared, id: &SessionId) -> u64 {
    shared
        .resolved
        .lock()
        .ok()
        .and_then(|counts| counts.get(&id.0).copied())
        .unwrap_or(0)
}

/// Blocks until `id` has resolved more asks than `seen`.
fn wait_resolved(shared: &Shared, id: &SessionId, seen: u64) {
    let Ok(mut counts) = shared.resolved.lock() else {
        return;
    };
    while counts.get(&id.0).copied().unwrap_or(0) <= seen {
        match shared.changed.wait(counts) {
            Ok(next) => counts = next,
            Err(_) => return,
        }
    }
}

/// Forwards `id`'s events after `after` to the editor, once per session.
fn follow(shared: &Arc<Shared>, id: &SessionId, after: Seq) -> Seq {
    // Where the history ends, and which of its asks are still open: a
    // replayed ask already answered is not asked again.
    let (history, open) = shared
        .client
        .call::<methods::Read>(&methods::Read {
            session_id: id.clone(),
        })
        .map(|read| {
            let open: HashSet<String> = read
                .open_asks
                .iter()
                .map(|ask| ask.ask.ask_id.0.clone())
                .collect();
            (read.last_seq, open)
        })
        .unwrap_or_default();
    let first = shared
        .following
        .lock()
        .is_ok_and(|mut following| following.insert(id.0.clone()));
    if !first {
        return history;
    }
    let Ok(events) = shared.client.subscribe(id, after) else {
        return history;
    };
    let shared = Arc::clone(shared);
    let id = id.clone();
    std::thread::spawn(move || {
        let mut streamed: HashSet<String> = HashSet::new();
        for record in events.flatten() {
            let replayed_ask = record.kind == "ask.opened"
                && record.seq <= history
                && !record
                    .data
                    .get("askId")
                    .and_then(Value::as_str)
                    .is_some_and(|ask| open.contains(ask));
            if !replayed_ask {
                forward(&shared, &id, &record, &mut streamed);
            }
            if let Ok(mut forwarded) = shared.forwarded.lock() {
                forwarded.insert(id.0.clone(), record.seq);
            }
            shared.changed.notify_all();
        }
    });
    history
}

/// The ACP `ToolKind` of a rig tool.
fn tool_kind(tool: &str) -> &'static str {
    match tool {
        "read" | "read_file" => "read",
        "edit" | "write" | "edit_file" | "write_file" => "edit",
        "shell" | "bash" => "execute",
        "search" | "grep" | "glob" => "search",
        "fetch" => "fetch",
        _ => "other",
    }
}

/// One rig event as what the editor sees. `streamed` holds the messages
/// whose text went out as deltas, so their final `message` is not sent
/// twice.
fn forward(shared: &Shared, id: &SessionId, record: &EventRecord, streamed: &mut HashSet<String>) {
    let data = &record.data;
    let field = |name: &str| data.get(name).and_then(Value::as_str).unwrap_or_default();
    let session = id.0.as_str();
    match record.kind.as_str() {
        "message.delta" => {
            streamed.insert(field("messageId").to_owned());
            if let Some(text) = data.get("text").and_then(Value::as_str) {
                shared.update(
                    session,
                    json!({"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": text}}),
                );
            }
            if let Some(text) = data.get("reasoning").and_then(Value::as_str) {
                shared.update(
                    session,
                    json!({"sessionUpdate": "agent_thought_chunk", "content": {"type": "text", "text": text}}),
                );
            }
        }
        "message" if !streamed.contains(field("messageId")) => {
            let kind = if field("role") == "user" {
                "user_message_chunk"
            } else {
                "agent_message_chunk"
            };
            shared.update(
                session,
                json!({"sessionUpdate": kind, "content": {"type": "text", "text": field("text")}}),
            );
        }
        "call.started" => shared.update(
            session,
            json!({
                "sessionUpdate": "tool_call",
                "toolCallId": field("callId"),
                "title": field("tool"),
                "kind": tool_kind(field("tool")),
                "status": "in_progress",
                "rawInput": data.get("args").cloned().unwrap_or(Value::Null),
            }),
        ),
        "call.ended" => {
            let ok = data.get("ok").and_then(Value::as_bool).unwrap_or(false);
            shared.update(
                session,
                json!({
                    "sessionUpdate": "tool_call_update",
                    "toolCallId": field("callId"),
                    "status": if ok { "completed" } else { "failed" },
                    "rawOutput": data.get("result").cloned().unwrap_or(Value::Null),
                }),
            );
        }
        "ask.opened" => ask(shared, id, data),
        "ask.resolved" => {
            if let Ok(mut counts) = shared.resolved.lock() {
                *counts.entry(id.0.clone()).or_default() += 1;
            }
            shared.changed.notify_all();
        }
        _ => {}
    }
}

/// An ask's options as ACP `PermissionOption`s.
/// ACP v1 knows four kinds; a question's choice, or any other kind, is
/// picked as `allow_once`.
pub fn permission_options(options: &[Value]) -> Vec<Value> {
    options
        .iter()
        .map(|option| {
            let kind = option
                .get("kind")
                .and_then(Value::as_str)
                .filter(|kind| {
                    matches!(
                        *kind,
                        "allow_once" | "allow_always" | "reject_once" | "reject_always"
                    )
                })
                .unwrap_or("allow_once");
            json!({
                "optionId": option.get("optionId"),
                "name": option.get("label"),
                "kind": kind,
            })
        })
        .collect()
}

/// An open ask as a `session/request_permission` request to the editor.
fn ask(shared: &Shared, id: &SessionId, data: &Value) {
    let Some(ask_id) = data.get("askId").and_then(Value::as_str) else {
        return;
    };
    let choices = data
        .get("options")
        .and_then(Value::as_array)
        .is_some_and(|options| !options.is_empty());
    if !choices {
        // ACP has no free-text request: the question shows in the reply,
        // and the model is told to decide, rather than wait 10 minutes.
        let question = data
            .get("title")
            .and_then(Value::as_str)
            .unwrap_or_default();
        shared.update(
            &id.0,
            json!({"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": format!("\n\n(asked: {question})\n\n")}}),
        );
        shared
            .client
            .call::<methods::Answer>(&methods::Answer {
                session_id: id.clone(),
                ask_id: AskId(ask_id.to_owned()),
                option_id: None,
                text: Some(FREE_TEXT_REPLY.to_owned()),
                key: Key(uuid::Uuid::new_v4().to_string()),
            })
            .ok();
        return;
    }
    let options = permission_options(
        data.get("options")
            .and_then(Value::as_array)
            .map(Vec::as_slice)
            .unwrap_or_default(),
    );
    let reject = data
        .get("options")
        .and_then(Value::as_array)
        .and_then(|options| {
            options.iter().find(|option| {
                option
                    .get("kind")
                    .and_then(Value::as_str)
                    .is_some_and(|kind| kind.starts_with("reject"))
            })
        })
        .and_then(|option| option.get("optionId"))
        .and_then(Value::as_str)
        .map(str::to_owned);
    let request = shared.next_id.fetch_add(1, Ordering::Relaxed);
    if let Ok(mut asking) = shared.asking.lock() {
        asking.insert(
            request,
            Asking {
                session: id.clone(),
                ask: AskId(ask_id.to_owned()),
                reject,
            },
        );
    }
    let title = data
        .get("title")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let agent = data
        .get("agentName")
        .and_then(Value::as_str)
        .unwrap_or_default();
    shared.send(&json!({
        "jsonrpc": "2.0",
        "id": request,
        "method": "session/request_permission",
        "params": {
            "sessionId": id.0,
            "toolCall": {
                "toolCallId": ask_id,
                "title": format!("{agent}: {title}"),
                "rawInput": data.get("body"),
            },
            "options": options,
        },
    }));
}
