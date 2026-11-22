//! The loopback MCP server a harness reaches Rig's tools through. One
//! listener serves every session: a session's bearer token names its
//! [`Door`], whose tools the server lists and whose calls it hands to the
//! session and holds open until Rig answers them.
//!
//! The server speaks stateless streamable HTTP: each `POST /mcp` carries one
//! JSON-RPC message and is answered with one JSON body.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex, PoisonError};

use axum::extract::State;
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use rig_harness::rig_core::completion::ToolDefinition;
use serde_json::{Value, json};
use tokio::sync::{mpsc, oneshot, watch};

#[cfg(test)]
pub(crate) mod tests;

/// The MCP revision answered when a client names none.
const PROTOCOL_VERSION: &str = "2025-06-18";

/// What Rig answered a tool call with.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Outcome {
    /// MCP content blocks.
    pub content: Vec<Value>,
    /// Whether the call failed.
    pub is_error: bool,
}

impl Outcome {
    /// A failed call that says `why`.
    pub(crate) fn error(why: impl Into<String>) -> Self {
        Self {
            content: vec![json!({ "type": "text", "text": why.into() })],
            is_error: true,
        }
    }
}

/// A tool call a harness made, open until `answer` is sent or dropped.
#[derive(Debug)]
pub(crate) struct Call {
    /// The tool's name, one of the door's.
    pub name: String,
    /// The arguments object.
    pub arguments: serde_json::Map<String, Value>,
    /// Where Rig's result goes.
    pub answer: oneshot::Sender<Outcome>,
}

/// One session's side of the server.
struct Door {
    tools: Arc<[Arc<ToolDefinition>]>,
    calls: mpsc::UnboundedSender<Call>,
    /// The streams the harness holds open for the server's notifications.
    streams: Vec<mpsc::UnboundedSender<Value>>,
    /// How many times the harness listed the tools.
    listed: watch::Sender<u64>,
}

type Doors = Arc<Mutex<HashMap<String, Door>>>;

/// The running server.
#[derive(Clone)]
pub(crate) struct Server {
    addr: SocketAddr,
    doors: Doors,
}

/// A session's registration, removed from the server when dropped.
pub(crate) struct Entry {
    token: String,
    url: String,
    doors: Doors,
}

impl Entry {
    /// The URL a harness reaches the server at.
    pub(crate) fn url(&self) -> &str {
        &self.url
    }

    /// The bearer token that names this session.
    pub(crate) fn token(&self) -> &str {
        &self.token
    }

    /// Offers `tools` from now on and tells the harness the list changed.
    /// The count of the harness's listings, to wait for its next one on;
    /// `None` when the harness holds no stream open to be told on.
    pub(crate) fn retool(&self, tools: Arc<[Arc<ToolDefinition>]>) -> Option<watch::Receiver<u64>> {
        let mut doors = self.doors.lock().unwrap_or_else(PoisonError::into_inner);
        let door = doors.get_mut(&self.token)?;
        door.tools = tools;
        let listed = door.listed.subscribe();
        let changed = json!({ "jsonrpc": "2.0", "method": "notifications/tools/list_changed" });
        door.streams
            .retain(|stream| stream.send(changed.clone()).is_ok());
        (!door.streams.is_empty()).then_some(listed)
    }
}

impl Drop for Entry {
    fn drop(&mut self) {
        self.doors
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .remove(&self.token);
    }
}

impl Server {
    /// Binds a loopback port and serves it on the current tokio runtime.
    pub(crate) async fn start() -> std::io::Result<Self> {
        let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0)).await?;
        let addr = listener.local_addr()?;
        let doors = Doors::default();
        let router = axum::Router::new()
            .route("/mcp", axum::routing::post(post).get(stream))
            .with_state(doors.clone());
        tokio::spawn(async move {
            let _ = axum::serve(listener, router).await;
        });
        Ok(Self { addr, doors })
    }

    /// Opens a door offering `tools`: the entry a harness is pointed at and
    /// the calls it makes.
    pub(crate) fn open(
        &self,
        tools: Arc<[Arc<ToolDefinition>]>,
    ) -> (Entry, mpsc::UnboundedReceiver<Call>) {
        let (calls, receiver) = mpsc::unbounded_channel();
        let token = uuid::Uuid::new_v4().simple().to_string();
        self.doors
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(
                token.clone(),
                Door {
                    tools,
                    calls,
                    streams: Vec::new(),
                    listed: watch::Sender::new(0),
                },
            );
        let entry = Entry {
            token,
            url: format!("http://{}/mcp", self.addr),
            doors: self.doors.clone(),
        };
        (entry, receiver)
    }
}

/// The door the request's bearer token names.
fn named<'a>(doors: &'a mut HashMap<String, Door>, headers: &HeaderMap) -> Option<&'a mut Door> {
    let token = headers
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "))?;
    doors.get_mut(token)
}

/// Opens a stream of the server's notifications to the harness, kept open
/// until the harness closes it or the session ends.
async fn stream(State(doors): State<Doors>, headers: HeaderMap) -> Response {
    let (sender, receiver) = mpsc::unbounded_channel::<Value>();
    {
        let mut doors = doors.lock().unwrap_or_else(PoisonError::into_inner);
        let Some(door) = named(&mut doors, &headers) else {
            return StatusCode::UNAUTHORIZED.into_response();
        };
        door.streams.push(sender);
    }
    let events = futures::stream::unfold(receiver, |mut receiver| async {
        let message = receiver.recv().await?;
        let event = Event::default().event("message").data(message.to_string());
        Some((Ok::<_, std::convert::Infallible>(event), receiver))
    });
    Sse::new(events)
        .keep_alive(KeepAlive::default())
        .into_response()
}

/// Answers one JSON-RPC message for the door the bearer token names.
async fn post(State(doors): State<Doors>, headers: HeaderMap, body: String) -> Response {
    let door = {
        let mut doors = doors.lock().unwrap_or_else(PoisonError::into_inner);
        named(&mut doors, &headers)
            .map(|door| (door.tools.clone(), door.calls.clone(), door.listed.clone()))
    };
    let Some((tools, calls, listings)) = door else {
        return StatusCode::UNAUTHORIZED.into_response();
    };
    let Ok(message) = serde_json::from_str::<Value>(&body) else {
        return failure(Value::Null, -32700, "the body is not JSON");
    };
    let Some(id) = message.get("id").filter(|id| !id.is_null()).cloned() else {
        // A notification or a response: nothing to answer.
        return StatusCode::ACCEPTED.into_response();
    };
    let params = message.get("params");
    let result = match message.get("method").and_then(Value::as_str) {
        Some("initialize") => {
            let version = params
                .and_then(|params| params.get("protocolVersion"))
                .and_then(Value::as_str)
                .unwrap_or(PROTOCOL_VERSION);
            json!({
                "protocolVersion": version,
                "capabilities": { "tools": { "listChanged": true } },
                "serverInfo": { "name": "rig", "version": env!("CARGO_PKG_VERSION") },
            })
        }
        Some("ping") => json!({}),
        Some("tools/list") => {
            listings.send_modify(|count| *count += 1);
            json!({ "tools": tools.iter().map(|tool| listed(tool)).collect::<Vec<_>>() })
        }
        Some("tools/call") => call(&tools, &calls, params).await,
        Some(other) => return failure(id, -32601, &format!("no method `{other}`")),
        None => return failure(id, -32600, "the message names no method"),
    };
    axum::Json(json!({ "jsonrpc": "2.0", "id": id, "result": result })).into_response()
}

/// A tool as `tools/list` shows it.
fn listed(tool: &ToolDefinition) -> Value {
    let mut listed = json!({
        "name": tool.name.as_str(),
        "description": tool.description,
        "inputSchema": tool.parameters,
    });
    let hints = &tool.annotations;
    let mut annotations = serde_json::Map::new();
    if let Some(read_only) = hints.read_only_hint {
        annotations.insert("readOnlyHint".to_owned(), read_only.into());
    }
    if let Some(destructive) = hints.destructive_hint {
        annotations.insert("destructiveHint".to_owned(), destructive.into());
    }
    if !annotations.is_empty()
        && let Some(listed) = listed.as_object_mut()
    {
        listed.insert("annotations".to_owned(), annotations.into());
    }
    listed
}

/// Hands a `tools/call` to the session and waits for Rig's result. A call
/// the session never answers, because it ended, fails.
async fn call(
    tools: &[Arc<ToolDefinition>],
    calls: &mpsc::UnboundedSender<Call>,
    params: Option<&Value>,
) -> Value {
    let name = params
        .and_then(|params| params.get("name"))
        .and_then(Value::as_str)
        .unwrap_or_default();
    let outcome = if tools.iter().any(|tool| tool.name.as_str() == name) {
        let arguments = params
            .and_then(|params| params.get("arguments"))
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        let (answer, answered) = oneshot::channel();
        let sent = calls.send(Call {
            name: name.to_owned(),
            arguments,
            answer,
        });
        match sent {
            Ok(()) => answered.await.ok(),
            Err(_) => None,
        }
        .unwrap_or_else(|| Outcome::error("The call was cancelled."))
    } else {
        Outcome::error(format!("No tool named `{name}`."))
    };
    json!({ "content": outcome.content, "isError": outcome.is_error })
}

/// A JSON-RPC error response.
fn failure(id: Value, code: i32, message: &str) -> Response {
    axum::Json(json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": { "code": code, "message": message },
    }))
    .into_response()
}
