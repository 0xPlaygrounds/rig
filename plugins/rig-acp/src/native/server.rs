use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use agent_client_protocol::schema::{v1, v2};
use agent_client_protocol::{
    Agent, Client, ConnectTo, ConnectionTo, Error, Result, Stdio as AcpStdio, V2ConnectionTo,
};
use rig_harness::prelude::Home;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::{Mutex, oneshot};

#[derive(Clone)]
enum Peer {
    V1(ConnectionTo<Client>),
    V2(V2ConnectionTo<Client>),
}
impl Peer {
    fn v2(&self) -> bool {
        matches!(self, Self::V2(_))
    }
    fn update(&self, session: &str, update: Value) -> Result<()> {
        let value = json!({"sessionId":session,"update":update});
        match self {
            Self::V1(cx) => cx.send_notification(decode::<v1::SessionNotification>(value)?),
            Self::V2(cx) => cx.send_notification(decode::<v2::UpdateSessionNotification>(value)?),
        }
    }
    async fn permission(&self, session: &str, event: &Value) -> Result<bool> {
        let options = json!([{"optionId":"deny","name":"Deny","kind":"reject_once"},{"optionId":"allow","name":"Allow","kind":"allow_once"}]);
        let tool =
            json!({"toolCallId":event.get("tool"),"title":event.get("title"),"status":"pending"});
        let response=match self {
            Self::V1(cx)=>serde_json::to_value(cx.send_request(decode::<v1::RequestPermissionRequest>(json!({"sessionId":session,"toolCall":tool,"options":options}))?).block_task().await?).map_err(internal)?,
            Self::V2(cx)=>serde_json::to_value(cx.send_request(decode::<v2::RequestPermissionRequest>(json!({"sessionId":session,"title":event.get("title"),"description":event.get("body"),"subject":{"type":"tool_call","toolCall":tool},"options":options}))?).block_task().await?).map_err(internal)?,
        };
        Ok(response
            .pointer("/outcome/optionId")
            .and_then(Value::as_str)
            == Some("allow"))
    }
}

fn config_options(mut response: Value, v2: bool) -> Value {
    if v2 {
        for option in response
            .get_mut("configOptions")
            .and_then(Value::as_array_mut)
            .into_iter()
            .flatten()
        {
            if let Some(object) = option.as_object_mut()
                && let Some(id) = object.shift_remove("id")
            {
                object.insert("configId".into(), id);
            }
        }
    }
    response
}

fn decode<T: serde::de::DeserializeOwned>(value: Value) -> Result<T> {
    serde_json::from_value(value).map_err(internal)
}
fn internal(error: impl std::fmt::Display) -> Error {
    Error::internal_error().data(error.to_string())
}
fn invalid(error: impl std::fmt::Display) -> Error {
    Error::invalid_params().data(error.to_string())
}

#[derive(Serialize, Deserialize)]
struct Saved {
    cwd: PathBuf,
    rig_id: String,
    version: u8,
}
struct Worker {
    input: Mutex<Option<tokio::process::ChildStdin>>,
    child: Mutex<tokio::process::Child>,
    pending: Mutex<HashMap<u64, oneshot::Sender<Result<Value>>>>,
    next: AtomicU64,
    completion: Mutex<Option<oneshot::Sender<Result<Value>>>>,
    epoch: String,
    history: PathBuf,
    history_write: Mutex<()>,
    lock: Mutex<Option<std::fs::File>>,
    tools: Mutex<std::collections::HashSet<String>>,
    interactions: Mutex<Vec<tokio::task::JoinHandle<()>>>,
    reader_finished: tokio::sync::Notify,
}
impl Worker {
    async fn shutdown(&self) {
        self.input.lock().await.take();
        let mut child = self.child.lock().await;
        if tokio::time::timeout(std::time::Duration::from_secs(5), child.wait())
            .await
            .is_err()
        {
            let _ = child.kill().await;
        }
    }

    async fn request(&self, mut value: Value) -> Result<Value> {
        let id = self.next.fetch_add(1, Ordering::Relaxed);
        let (sender, receiver) = oneshot::channel();
        self.pending.lock().await.insert(id, sender);
        value
            .as_object_mut()
            .ok_or_else(|| invalid("invalid worker request"))?
            .insert("id".into(), id.into());
        let mut bytes = serde_json::to_vec(&value).map_err(internal)?;
        bytes.push(b'\n');
        let sent = {
            let mut input = self.input.lock().await;
            match input.as_mut() {
                Some(input) => input.write_all(&bytes).await.map_err(internal),
                None => Err(internal("session is closed")),
            }
        };
        if let Err(error) = sent {
            self.pending.lock().await.remove(&id);
            return Err(error);
        }
        receiver.await.map_err(internal)?
    }
}
struct Backend {
    home: Home,
    workers: Arc<Mutex<HashMap<String, Arc<Worker>>>>,
}
impl Backend {
    fn path(&self, id: &str) -> Result<PathBuf> {
        uuid::Uuid::parse_str(id).map_err(invalid)?;
        Ok(self.home.root().join("acp").join(id))
    }
    async fn worker(&self, id: &str) -> Result<Arc<Worker>> {
        self.workers
            .lock()
            .await
            .get(id)
            .cloned()
            .ok_or_else(|| invalid("session is not open"))
    }
    async fn open(&self, params: &Value, peer: Peer, resume: bool) -> Result<Value> {
        let cwd = PathBuf::from(
            params
                .get("cwd")
                .and_then(Value::as_str)
                .ok_or_else(|| invalid("cwd is required"))?,
        );
        if !cwd.is_absolute() || !cwd.is_dir() {
            return Err(invalid("cwd must be an existing absolute directory"));
        }
        if params
            .get("additionalDirectories")
            .and_then(Value::as_array)
            .is_some_and(|v| !v.is_empty())
        {
            return Err(invalid("additional directories are unsupported"));
        }
        let id = if resume {
            params
                .get("sessionId")
                .and_then(Value::as_str)
                .ok_or_else(|| invalid("sessionId is required"))?
                .to_owned()
        } else {
            uuid::Uuid::new_v4().to_string()
        };
        let path = self.path(&id)?;
        if !resume {
            tokio::fs::create_dir_all(&path).await.map_err(internal)?;
        }
        let lock = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(path.join("lock"))
            .map_err(invalid)?;
        lock.try_lock()
            .map_err(|_| invalid("session is open in another connection"))?;
        if self.workers.lock().await.contains_key(&id) {
            return Err(invalid("session is already open"));
        }
        let saved = if resume {
            let saved: Saved = decode(
                serde_json::from_slice(
                    &tokio::fs::read(path.join("session.json"))
                        .await
                        .map_err(invalid)?,
                )
                .map_err(invalid)?,
            )?;
            if saved.version != if peer.v2() { 2 } else { 1 } {
                return Err(invalid("resume using the original protocol version"));
            }
            if saved.cwd != cwd {
                return Err(invalid("session working directory differs"));
            }
            saved
        } else {
            let (rig_id, _) = self.home.new_session().map_err(internal)?;
            tokio::fs::create_dir_all(&path).await.map_err(internal)?;
            let saved = Saved {
                cwd: cwd.clone(),
                rig_id: rig_id.as_str().to_owned(),
                version: if peer.v2() { 2 } else { 1 },
            };
            tokio::fs::write(
                path.join("session.json"),
                serde_json::to_vec(&saved).map_err(internal)?,
            )
            .await
            .map_err(internal)?;
            saved
        };
        let mut command = tokio::process::Command::new(std::env::current_exe().map_err(internal)?);
        command
            .arg("acp-worker")
            .current_dir(&cwd)
            .env("RIG_SESSION", &saved.rig_id)
            .env("RIG_HOME", self.home.root())
            .env_remove("RIG_RUN")
            .env_remove("RIG_START")
            .env_remove("RIG_NOTICE")
            .env_remove("RIG_BUILD_FAILURE")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .kill_on_drop(true);
        let mut child = command.spawn().map_err(internal)?;
        let input = child
            .stdin
            .take()
            .ok_or_else(|| internal("missing worker stdin"))?;
        let output = child
            .stdout
            .take()
            .ok_or_else(|| internal("missing worker stdout"))?;
        let worker = Arc::new(Worker {
            input: Mutex::new(Some(input)),
            child: Mutex::new(child),
            pending: Mutex::default(),
            next: AtomicU64::new(1),
            completion: Mutex::default(),
            epoch: uuid::Uuid::new_v4().to_string(),
            history: path.join("events.jsonl"),
            history_write: Mutex::new(()),
            lock: Mutex::new(Some(lock)),
            tools: Mutex::default(),
            interactions: Mutex::default(),
            reader_finished: tokio::sync::Notify::new(),
        });
        let mut setup = serde_json::to_vec(
            &json!({"servers":params.get("mcpServers").cloned().unwrap_or(json!([]))}),
        )
        .map_err(internal)?;
        setup.push(b'\n');
        worker
            .input
            .lock()
            .await
            .as_mut()
            .ok_or_else(|| internal("session is closed"))?
            .write_all(&setup)
            .await
            .map_err(internal)?;
        let mut lines = BufReader::new(output).lines();
        let mut restored = std::collections::VecDeque::new();
        let ready = tokio::time::timeout(std::time::Duration::from_secs(60), async {
            loop {
                let line = lines
                    .next_line()
                    .await
                    .map_err(internal)?
                    .ok_or_else(|| internal("worker exited during setup"))?;
                let event: Value = serde_json::from_str(&line).map_err(internal)?;
                if event.get("event").is_some() {
                    restored.push_back(event);
                } else {
                    break Ok::<_, Error>(event);
                }
            }
        })
        .await
        .map_err(internal)??;
        let ready = config_options(ready, peer.v2());
        if ready.get("ready") != Some(&Value::Bool(true)) {
            return Err(internal(ready.get("error").unwrap_or(&ready)));
        }
        if resume && (params.get("replayFrom").is_some_and(|v| !v.is_null()) || !peer.v2()) {
            if let Some(replay) = params.get("replayFrom").filter(|v| !v.is_null())
                && replay.get("type").and_then(Value::as_str) != Some("start")
            {
                return Err(invalid("only replay from start is supported"));
            }
            if let Ok(history) = tokio::fs::read_to_string(&worker.history).await {
                for line in history.lines() {
                    let event: Value = serde_json::from_str(line).map_err(internal)?;
                    if event.get("version").and_then(Value::as_u64)
                        != Some(if peer.v2() { 2 } else { 1 })
                    {
                        return Err(invalid(
                            "resume using the session's original protocol version",
                        ));
                    }
                    peer.update(
                        &id,
                        event
                            .get("update")
                            .cloned()
                            .ok_or_else(|| internal("invalid history"))?,
                    )?;
                }
            }
        }
        if peer.v2() {
            peer.update(&id, json!({"sessionUpdate":"state_update","state":if ready.get("busy")==Some(&Value::Bool(true)){"running"}else{"idle"}}))?;
        }
        self.workers.lock().await.insert(id.clone(), worker.clone());
        tokio::spawn(read_worker(
            worker,
            lines,
            peer,
            id.clone(),
            self.workers.clone(),
            restored,
        ));
        Ok(
            json!({"sessionId":id,"configOptions":ready.get("configOptions").cloned().unwrap_or(json!([]))}),
        )
    }
    async fn handle(&self, method: &str, params: Value, peer: Peer) -> Result<Value> {
        if method == "new" {
            return self.open(&params, peer, false).await;
        }
        if method == "resume" {
            return self.open(&params, peer, true).await;
        }
        if method == "list" {
            let mut sessions = Vec::new();
            let directory = self.home.root().join("acp");
            if directory.exists() {
                let mut entries = tokio::fs::read_dir(directory).await.map_err(internal)?;
                while let Some(entry) = entries.next_entry().await.map_err(internal)? {
                    if let Ok(bytes) = tokio::fs::read(entry.path().join("session.json")).await
                        && let Ok(saved) = serde_json::from_slice::<Saved>(&bytes)
                        && params
                            .get("cwd")
                            .and_then(Value::as_str)
                            .is_none_or(|cwd| Path::new(cwd) == saved.cwd)
                    {
                        sessions.push(json!({"sessionId":entry.file_name().to_string_lossy(),"cwd":saved.cwd}));
                    }
                }
            }
            return Ok(json!({"sessions":sessions}));
        }
        let id = params
            .get("sessionId")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid("missing sessionId"))?;
        if method == "close" {
            let worker = self.workers.lock().await.remove(id);
            if let Some(worker) = worker {
                worker.shutdown().await;
                worker.reader_finished.notified().await;
                return Ok(json!({}));
            }
            if self.path(id)?.join("session.json").is_file() {
                return Ok(json!({}));
            }
            return Err(invalid("unknown session"));
        }
        let worker = self.worker(id).await?;
        match method {
            "prompt" => {
                let blocks = params
                    .get("prompt")
                    .and_then(Value::as_array)
                    .ok_or_else(|| invalid("missing prompt"))?;
                let mut text = String::new();
                for block in blocks {
                    match block.get("type").and_then(Value::as_str) {
                        Some("text") => text.push_str(
                            block
                                .get("text")
                                .and_then(Value::as_str)
                                .ok_or_else(|| invalid("missing text"))?,
                        ),
                        Some("resource_link") => {
                            text.push_str("Referenced resource: ");
                            text.push_str(
                                block
                                    .get("uri")
                                    .and_then(Value::as_str)
                                    .ok_or_else(|| invalid("missing resource URI"))?,
                            );
                        }
                        _ => return Err(invalid("only text and resource links are supported")),
                    }
                    text.push('\n');
                }
                if text.trim().is_empty() {
                    return Err(invalid("prompt must contain non-whitespace text"));
                }
                let (sender, receiver) = oneshot::channel();
                {
                    let mut completion = worker.completion.lock().await;
                    if completion.is_some() {
                        return Err(invalid("session is busy"));
                    }
                    *completion = Some(sender);
                }
                if let Err(error) = worker.request(json!({"method":"prompt","text":text})).await {
                    worker.completion.lock().await.take();
                    return Err(error);
                }
                let message = uuid::Uuid::new_v4().to_string();
                let update = if peer.v2() {
                    json!({
                        "sessionUpdate": "user_message",
                        "messageId": message,
                        "content": blocks,
                    })
                } else {
                    json!({
                        "sessionUpdate": "user_message_chunk",
                        "content": {"type": "text", "text": text},
                    })
                };
                let prepared = record(&worker, &peer, id, update).await.and_then(|()| {
                    if peer.v2() {
                        peer.update(
                            id,
                            json!({"sessionUpdate":"state_update","state":"running"}),
                        )
                    } else {
                        Ok(())
                    }
                });
                if let Err(error) = prepared {
                    // The cancellation event releases both pending work and its completion slot.
                    if worker.request(json!({"method":"cancel"})).await.is_err() {
                        worker.shutdown().await;
                    }
                    return Err(error);
                }
                if peer.v2() {
                    Ok(json!({"messageId":message}))
                } else {
                    worker.request(json!({"method":"run"})).await?;
                    receiver.await.map_err(internal)?
                }
            }
            "cancel" => worker.request(json!({"method":"cancel"})).await,
            "config" => {
                if peer.v2() && params.get("type").and_then(Value::as_str) != Some("id") {
                    return Err(invalid("configuration value must have type id"));
                }
                let result=worker.request(json!({"method":"config","configId":params.get("configId"),"value":params.get("value")})).await?;
                let result = config_options(result, peer.v2());
                peer.update(id,json!({"sessionUpdate":"config_option_update","configOptions":result.get("configOptions")}))?;
                Ok(result)
            }
            _ => Err(invalid("unsupported operation")),
        }
    }
}

async fn record(worker: &Worker, peer: &Peer, id: &str, update: Value) -> Result<()> {
    let _writing = worker.history_write.lock().await;
    let mut file = tokio::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&worker.history)
        .await
        .map_err(internal)?;
    let mut bytes = serde_json::to_vec(&json!({"version":if peer.v2(){2}else{1},"update":update}))
        .map_err(internal)?;
    bytes.push(b'\n');
    file.write_all(&bytes).await.map_err(internal)?;
    peer.update(id, update)
}

async fn read_worker(
    worker: Arc<Worker>,
    mut lines: tokio::io::Lines<BufReader<tokio::process::ChildStdout>>,
    peer: Peer,
    id: String,
    workers: Arc<Mutex<HashMap<String, Arc<Worker>>>>,
    mut restored: std::collections::VecDeque<Value>,
) {
    loop {
        let event = if let Some(event) = restored.pop_front() {
            event
        } else {
            let Ok(Some(line)) = lines.next_line().await else {
                break;
            };
            let Ok(event) = serde_json::from_str::<Value>(&line) else {
                break;
            };
            event
        };
        if let Some(request) = event.get("id").and_then(Value::as_u64) {
            if let Some(sender) = worker.pending.lock().await.remove(&request) {
                let result = event.get("error").map_or_else(
                    || Ok(event.get("result").cloned().unwrap_or(json!({}))),
                    |e| Err(invalid(e)),
                );
                let _ = sender.send(result);
            }
            continue;
        }
        let result = event_update(&worker, &peer, &id, &event).await;
        if let Err(error) = result {
            eprintln!("ACP session event failed: {error}");
            break;
        }
    }
    worker.input.lock().await.take();
    worker.shutdown().await;
    worker.lock.lock().await.take();
    let mut workers = workers.lock().await;
    if workers
        .get(&id)
        .is_some_and(|current| Arc::ptr_eq(current, &worker))
    {
        workers.remove(&id);
    }
    drop(workers);
    for (_, sender) in worker.pending.lock().await.drain() {
        let _ = sender.send(Err(internal("session worker exited")));
    }
    if let Some(sender) = worker.completion.lock().await.take() {
        if peer.v2() {
            let _ = peer.update(
                &id,
                json!({"sessionUpdate":"state_update","state":"idle","stopReason":"error","error":{"code":-32603,"message":"session worker exited"}}),
            );
        }
        let _ = sender.send(Err(internal("session worker exited")));
    }
    for task in worker.interactions.lock().await.drain(..) {
        task.abort();
    }
    worker.reader_finished.notify_one();
}
async fn event_update(worker: &Arc<Worker>, peer: &Peer, id: &str, event: &Value) -> Result<()> {
    match event.get("event").and_then(Value::as_str) {
        Some("state") => {
            if peer.v2() {
                peer.update(
                    id,
                    json!({"sessionUpdate":"state_update","state":event.get("state")}),
                )?;
            }
            Ok(())
        }
        Some("text" | "thought") => {
            let mut update = json!({"sessionUpdate":if event.get("event").and_then(Value::as_str)==Some("text"){"agent_message_chunk"}else{"agent_thought_chunk"},"content":{"type":"text","text":event.get("text")}});
            if peer.v2() {
                update
                    .as_object_mut()
                    .ok_or_else(|| internal("invalid update"))?
                    .insert(
                        "messageId".into(),
                        format!(
                            "{}-{}",
                            worker.epoch,
                            event
                                .get("message")
                                .and_then(Value::as_str)
                                .unwrap_or_default()
                        )
                        .into(),
                    );
            }
            record(worker, peer, id, update).await
        }
        Some("tool") => {
            let first = worker.tools.lock().await.insert(
                event
                    .get("id")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
            );
            let mut update = json!({"sessionUpdate":if !peer.v2() && first {"tool_call"}else{"tool_call_update"},"toolCallId":format!("{}-{}",worker.epoch,event.get("id").and_then(Value::as_str).unwrap_or_default()),"title":event.get("title"),"status":event.get("status")});
            if let Some(input) = event.get("input") {
                update
                    .as_object_mut()
                    .ok_or_else(|| internal("invalid update"))?
                    .insert("rawInput".into(), input.clone());
            }
            if let Some(output) = event.get("output") {
                update
                    .as_object_mut()
                    .ok_or_else(|| internal("invalid update"))?
                    .insert(
                        "content".into(),
                        json!([{"type":"content","content":{"type":"text","text":output}}]),
                    );
            }
            record(worker, peer, id, update).await
        }
        Some("ask") => {
            let owner = worker.clone();
            let worker = worker.clone();
            let peer = peer.clone();
            let id = id.to_owned();
            let mut event = event.clone();
            if let Some(tool) = event.get("tool").and_then(Value::as_str).map(str::to_owned)
                && let Some(object) = event.as_object_mut()
            {
                object.insert("tool".into(), format!("{}-{tool}", worker.epoch).into());
            }
            let task = tokio::spawn(async move {
                if event.get("tool").is_none_or(Value::is_null) {
                    // General questions have no ACP v1 permission equivalent.
                    let mut update = json!({"sessionUpdate":"agent_message_chunk","content":{"type":"text","text":format!("This ACP front cannot answer the interactive question: {}. The turn was stopped.",event.get("title").and_then(Value::as_str).unwrap_or_default())}});
                    if peer.v2()
                        && let Some(object) = update.as_object_mut()
                    {
                        object.insert("messageId".into(), uuid::Uuid::new_v4().to_string().into());
                    }
                    let _ = record(&worker, &peer, &id, update).await;
                    let _ = worker.request(json!({"method":"cancel"})).await;
                    return;
                }
                let allowed = peer.permission(&id, &event).await.unwrap_or(false);
                let _=worker.request(json!({"method":"answer","ask":event.get("ask"),"choice":if allowed{"allow"}else{"deny"}})).await;
            });
            owner.interactions.lock().await.push(task);
            Ok(())
        }
        Some("end") => {
            for task in worker.interactions.lock().await.drain(..) {
                task.abort();
            }
            let reason = event
                .get("reason")
                .and_then(Value::as_str)
                .unwrap_or("error");
            if peer.v2() {
                peer.update(
                    id,
                    json!({"sessionUpdate":"state_update","state":"idle","stopReason":reason,"error": if reason == "error" {Some(json!({"code":-32603,"message":event.get("error").and_then(Value::as_str).unwrap_or("session failed")}))} else {None}}),
                )?;
            }
            if let Some(sender) = worker.completion.lock().await.take() {
                let _ = sender.send(if reason == "error" {
                    Err(internal(event.get("error").unwrap_or(&Value::Null)))
                } else {
                    Ok(json!({"stopReason":reason}))
                });
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

pub(super) async fn serve(home: Home, stopped: oneshot::Receiver<()>) -> Result<()> {
    let backend = Arc::new(Backend {
        home,
        workers: Arc::default(),
    });
    let result = tokio::select! {
        result = router(backend.clone()).connect_to(AcpStdio::new()) => result,
        _ = stopped => Ok(()),
    };
    let workers: Vec<_> = backend
        .workers
        .lock()
        .await
        .drain()
        .map(|(_, worker)| worker)
        .collect();
    let mut shutdowns = tokio::task::JoinSet::new();
    for worker in workers {
        shutdowns.spawn(async move {
            worker.shutdown().await;
        });
    }
    while shutdowns.join_next().await.is_some() {}
    result
}

fn router(backend: Arc<Backend>) -> impl ConnectTo<Client> {
    let v1 = Agent.builder().on_receive_request(
        async |request: v1::InitializeRequest, responder, _cx| {
            responder.respond(
                v1::InitializeResponse::new(request.protocol_version)
                    .agent_info(v1::Implementation::new("rig", env!("CARGO_PKG_VERSION")))
                    .agent_capabilities(
                        v1::AgentCapabilities::new()
                            .load_session(true)
                            .mcp_capabilities(v1::McpCapabilities::new().http(true)),
                    ),
            )
        },
        agent_client_protocol::on_receive_request!(),
    );
    let v2 = Agent.v2().on_receive_request(
        async |request: v2::InitializeRequest, responder, _cx| {
            responder.respond(
                v2::InitializeResponse::new(
                    request.protocol_version,
                    v2::Implementation::new("rig", env!("CARGO_PKG_VERSION")),
                )
                .capabilities(
                    v2::AgentCapabilities::new().session(
                        v2::SessionCapabilities::new().mcp(
                            v2::McpCapabilities::new()
                                .http(v2::McpHttpCapabilities::new())
                                .stdio(v2::McpStdioCapabilities::new()),
                        ),
                    ),
                ),
            )
        },
        agent_client_protocol::on_receive_request!(),
    );
    macro_rules! handler {
        ($builder:ident,$version:ident,$request:ident,$response:ident,$method:literal,$peer:ident) => {
            let $builder = $builder.on_receive_request(
                {
                    let backend = backend.clone();
                    async move |request: $version::$request,
                                responder: agent_client_protocol::Responder<$version::$response>,
                                cx| {
                        let backend = backend.clone();
                        let peer = Peer::$peer(cx.clone());
                        cx.spawn(async move {
                            let result = backend
                                .handle(
                                    $method,
                                    serde_json::to_value(request).map_err(internal)?,
                                    peer,
                                )
                                .await
                                .and_then(decode::<$version::$response>);
                            match result {
                                Ok(value) => responder.respond(value),
                                Err(error) => responder.respond_with_error(error),
                            }
                        })
                    }
                },
                agent_client_protocol::on_receive_request!(),
            );
        };
    }
    handler!(v1, v1, NewSessionRequest, NewSessionResponse, "new", V1);
    handler!(
        v1,
        v1,
        LoadSessionRequest,
        LoadSessionResponse,
        "resume",
        V1
    );
    handler!(v1, v1, PromptRequest, PromptResponse, "prompt", V1);
    handler!(
        v1,
        v1,
        SetSessionConfigOptionRequest,
        SetSessionConfigOptionResponse,
        "config",
        V1
    );
    handler!(v2, v2, NewSessionRequest, NewSessionResponse, "new", V2);
    handler!(
        v2,
        v2,
        ResumeSessionRequest,
        ResumeSessionResponse,
        "resume",
        V2
    );
    handler!(
        v2,
        v2,
        ListSessionsRequest,
        ListSessionsResponse,
        "list",
        V2
    );
    handler!(
        v2,
        v2,
        CloseSessionRequest,
        CloseSessionResponse,
        "close",
        V2
    );
    let v2 = v2.on_receive_request(
        {
            let backend = backend.clone();
            async move |request: v2::PromptRequest,
                        responder: agent_client_protocol::Responder<v2::PromptResponse>,
                        cx: V2ConnectionTo<Client>| {
                let backend = backend.clone();
                let peer = Peer::V2(cx.clone());
                cx.spawn(async move {
                    let id = request.session_id.to_string();
                    let result = backend
                        .handle(
                            "prompt",
                            serde_json::to_value(request).map_err(internal)?,
                            peer.clone(),
                        )
                        .await
                        .and_then(decode::<v2::PromptResponse>);
                    match result {
                        Ok(value) => {
                            responder.respond(value)?;
                            let worker=backend.worker(&id).await?;
                            if let Err(error)=worker.request(json!({"method":"run"})).await
                                && worker.completion.lock().await.take().is_some() {
                                peer.update(&id,json!({"sessionUpdate":"state_update","state":"idle","stopReason":"error","error":{"code":-32603,"message":error.to_string()}}))?;
                            }
                            Ok(())
                        }
                        Err(error) => responder.respond_with_error(error),
                    }
                })
            }
        },
        agent_client_protocol::on_receive_request!(),
    );
    handler!(
        v2,
        v2,
        SetSessionConfigOptionRequest,
        SetSessionConfigOptionResponse,
        "config",
        V2
    );
    let v1 = v1.on_receive_notification(
        {
            let backend = backend.clone();
            async move |request: v1::CancelNotification, cx| {
                let backend = backend.clone();
                let peer = Peer::V1(cx.clone());
                cx.spawn(async move {
                    backend
                        .handle(
                            "cancel",
                            serde_json::to_value(request).map_err(internal)?,
                            peer,
                        )
                        .await
                        .map(|_| ())
                })
            }
        },
        agent_client_protocol::on_receive_notification!(),
    );
    let v2 = v2.on_receive_notification(
        {
            let backend = backend.clone();
            async move |request: v2::CancelSessionNotification, cx| {
                let backend = backend.clone();
                let peer = Peer::V2(cx.clone());
                cx.spawn(async move {
                    backend
                        .handle(
                            "cancel",
                            serde_json::to_value(request).map_err(internal)?,
                            peer,
                        )
                        .await
                        .map(|_| ())
                })
            }
        },
        agent_client_protocol::on_receive_notification!(),
    );
    Agent.protocol_router().with_v1(v1).with_v2(v2)
}

#[cfg(test)]
mod tests;
