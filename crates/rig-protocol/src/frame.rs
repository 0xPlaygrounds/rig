//! JSON-RPC 2.0 frames and their NDJSON framing.

use std::io::{BufRead, Read, Write};

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::{MAX_FRAME, PROTOCOL, SUPPORTED, codes};

/// One frame on the wire.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Frame {
    Request(Request),
    Response(Response),
    Notification(Notification),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Request {
    pub jsonrpc: JsonRpc,
    pub rig: u32,
    pub id: RequestId,
    pub method: String,
    #[serde(default)]
    pub params: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Response {
    pub jsonrpc: JsonRpc,
    pub rig: u32,
    pub id: RequestId,
    #[serde(flatten)]
    pub outcome: Outcome,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Outcome {
    Result(serde_json::Value),
    Error(RpcError),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Notification {
    pub jsonrpc: JsonRpc,
    pub rig: u32,
    pub method: String,
    pub params: serde_json::Value,
}

/// A request id, unique per connection.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RequestId {
    Num(u64),
    Str(String),
}

/// Serializes as "2.0" and refuses anything else.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct JsonRpc;

impl Serialize for JsonRpc {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str("2.0")
    }
}

impl<'de> Deserialize<'de> for JsonRpc {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let version = String::deserialize(d)?;
        if version == "2.0" {
            Ok(JsonRpc)
        } else {
            Err(serde::de::Error::custom(format!(
                "jsonrpc must be \"2.0\", not {version:?}"
            )))
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RpcError {
    pub code: i32,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<ErrorData>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ErrorData {
    /// The command or action that fixes it, e.g. "rig answer s-1 a-3 --allow".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub next: Option<String>,
    /// Version refusals: which side must update.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub update: Option<Side>,
    /// Version refusals: what the refusing side speaks.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub supported: Vec<u32>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Side {
    Client,
    Server,
    Worker,
}

impl Request {
    /// A request at this build's [`PROTOCOL`].
    pub fn new(id: RequestId, method: impl Into<String>, params: serde_json::Value) -> Request {
        Request {
            jsonrpc: JsonRpc,
            rig: PROTOCOL,
            id,
            method: method.into(),
            params,
        }
    }
}

impl Response {
    /// The answer to request `id` at this build's [`PROTOCOL`].
    pub fn new(id: RequestId, outcome: Result<serde_json::Value, RpcError>) -> Response {
        Response {
            jsonrpc: JsonRpc,
            rig: PROTOCOL,
            id,
            outcome: match outcome {
                Ok(value) => Outcome::Result(value),
                Err(error) => Outcome::Error(error),
            },
        }
    }
}

impl Notification {
    /// A notification at this build's [`PROTOCOL`].
    pub fn new(method: impl Into<String>, params: serde_json::Value) -> Notification {
        Notification {
            jsonrpc: JsonRpc,
            rig: PROTOCOL,
            method: method.into(),
            params,
        }
    }
}

impl RpcError {
    pub fn new(code: i32, message: impl Into<String>) -> RpcError {
        RpcError {
            code,
            message: message.into(),
            data: None,
        }
    }

    /// This error with `data.next` set to the command that fixes it.
    pub fn with_next(mut self, next: impl Into<String>) -> RpcError {
        self.data.get_or_insert_with(ErrorData::default).next = Some(next.into());
        self
    }

    /// The `-32010` refusal a server makes of a peer speaking `got` (`None`:
    /// no `rig` field): [`version_read_by`](Self::version_read_by) the server.
    pub fn version(got: Option<u32>) -> RpcError {
        RpcError::version_read_by(got, Side::Server)
    }

    /// The `-32010` refusal `reader` makes of a frame speaking `got`. The side
    /// to update is the older one: `reader` itself when the frame is newer
    /// than anything it speaks, else the peer that wrote the frame (a server
    /// is read by clients and workers; a client or worker by the server).
    pub fn version_read_by(got: Option<u32>, reader: Side) -> RpcError {
        let newer = got.is_some_and(|v| v > *SUPPORTED.end());
        let update = match (newer, reader) {
            (true, reader) => reader,
            (false, Side::Server) => Side::Client,
            (false, Side::Client | Side::Worker) => Side::Server,
        };
        let message = match got {
            Some(v) => format!("rig wire version {v} is not supported"),
            None => "the frame has no \"rig\" version".to_owned(),
        };
        let next = match update {
            Side::Server => {
                "install a newer rig where the server runs, then `rig server stop` so the new one starts"
            }
            Side::Client => {
                "install a newer rig on the client side, or the newest clients/python/rig_client.py"
            }
            Side::Worker => "rebuild the agent with `rig build`",
        };
        RpcError {
            code: codes::VERSION,
            message,
            data: Some(ErrorData {
                next: Some(next.to_owned()),
                update: Some(update),
                supported: SUPPORTED.collect(),
                ..ErrorData::default()
            }),
        }
    }
}

impl std::fmt::Display for RpcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ({})", self.message, self.code)
    }
}

impl std::error::Error for RpcError {}

/// Why a line is not a frame. Every case but `Io` leaves the reader at the
/// next line, so the connection can answer and go on.
#[derive(Debug, thiserror::Error)]
pub enum FrameError {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("a frame is longer than {MAX_FRAME} bytes")]
    TooLarge {
        /// The request's id, when the part read holds it at the top level.
        id: Option<RequestId>,
    },
    #[error("a frame is not UTF-8")]
    Utf8,
    #[error("JSON-RPC batches are not supported")]
    Batch,
    /// The frame has no `rig` field, or one outside [`SUPPORTED`].
    #[error("unsupported rig wire version {got:?}")]
    Version {
        got: Option<u32>,
        id: Option<RequestId>,
    },
    #[error("not a JSON-RPC frame: {0}")]
    Json(#[from] serde_json::Error),
}

impl FrameError {
    /// The id of the request the bad line was, when it could be read: the
    /// id to answer the refusal with (`null` otherwise).
    pub fn id(&self) -> Option<&RequestId> {
        match self {
            FrameError::TooLarge { id } | FrameError::Version { id, .. } => id.as_ref(),
            _ => None,
        }
    }

    /// The error a server answers the bad line with, if the connection
    /// stays open: [`refusal`](Self::refusal) by the server.
    pub fn rpc(&self) -> Option<RpcError> {
        self.refusal(Side::Server)
    }

    /// Why `reader` refuses the line, as an error: `None` for I/O, which ends
    /// the connection. JSON that isn't a frame is `-32600`, text that isn't
    /// JSON `-32700`.
    pub fn refusal(&self, reader: Side) -> Option<RpcError> {
        Some(match self {
            FrameError::Io(_) => return None,
            FrameError::TooLarge { .. } => RpcError::new(codes::FRAME_TOO_LARGE, self.to_string()),
            FrameError::Json(e) if e.is_data() => {
                RpcError::new(codes::INVALID_REQUEST, self.to_string())
            }
            FrameError::Utf8 | FrameError::Json(_) => RpcError::new(codes::PARSE, self.to_string()),
            FrameError::Batch => RpcError::new(codes::INVALID_REQUEST, self.to_string()),
            FrameError::Version { got, .. } => RpcError::version_read_by(*got, reader),
        })
    }
}

/// One NDJSON line. `Ok(None)` at EOF. Skips blank lines, accepts CRLF, and
/// rejects lines over [`MAX_FRAME`], invalid UTF-8, batches and frames whose
/// `rig` version is outside [`SUPPORTED`]. `line` is the reused buffer.
pub fn read_frame(r: &mut impl BufRead, line: &mut Vec<u8>) -> Result<Option<Frame>, FrameError> {
    loop {
        line.clear();
        // Two more bytes than a frame: its CRLF.
        let limit = MAX_FRAME as u64 + 2;
        let read = r.by_ref().take(limit).read_until(b'\n', line)?;
        if read == 0 {
            return Ok(None);
        }
        let ended = line.last() == Some(&b'\n');
        if ended {
            line.pop();
            if line.last() == Some(&b'\r') {
                line.pop();
            }
        }
        if line.len() > MAX_FRAME {
            if !ended {
                r.skip_until(b'\n')?;
            }
            let id = leading_id(line);
            return Err(FrameError::TooLarge { id });
        }
        let text = std::str::from_utf8(line).map_err(|_| FrameError::Utf8)?;
        let text = text.trim();
        if text.is_empty() {
            continue;
        }
        if text.starts_with('[') {
            return Err(FrameError::Batch);
        }
        let value: serde_json::Value = serde_json::from_str(text)?;
        let got = value
            .get("rig")
            .and_then(serde_json::Value::as_u64)
            .and_then(|v| u32::try_from(v).ok());
        if !got.is_some_and(|v| SUPPORTED.contains(&v)) {
            let id = value
                .get("id")
                .and_then(|id| serde_json::from_value(id.clone()).ok());
            return Err(FrameError::Version { got, id });
        }
        return Ok(Some(serde_json::from_value(value)?));
    }
}

/// Writes `frame` as one line and flushes.
pub fn write_frame(w: &mut impl Write, frame: &Frame) -> std::io::Result<()> {
    let mut bytes = serde_json::to_vec(frame)?;
    bytes.push(b'\n');
    w.write_all(&bytes)?;
    w.flush()
}

/// The top-level `"id"` of the JSON object `bytes` begins, read from its
/// prefix alone: a number or a string without escapes. Members nested in
/// other values are skipped, so a `params.id` is never taken for it.
fn leading_id(bytes: &[u8]) -> Option<RequestId> {
    let mut depth = 0usize;
    let mut i = 0;
    while let Some(&byte) = bytes.get(i) {
        match byte {
            b'{' | b'[' => depth += 1,
            b'}' | b']' => depth = depth.checked_sub(1)?,
            b'"' => {
                let end = string_end(bytes, i)?;
                let key = bytes.get(i + 1..end)?;
                i = end + 1;
                if depth == 1 && key == b"id" {
                    let rest = bytes.get(i..)?;
                    let rest = rest.trim_ascii_start().strip_prefix(b":")?;
                    return id_value(rest.trim_ascii_start());
                }
                continue;
            }
            _ => {}
        }
        i += 1;
    }
    None
}

/// The index of the quote that closes the string opening at `start`.
fn string_end(bytes: &[u8], start: usize) -> Option<usize> {
    let mut i = start + 1;
    loop {
        match bytes.get(i)? {
            b'\\' => i += 2,
            b'"' => return Some(i),
            _ => i += 1,
        }
    }
}

fn id_value(value: &[u8]) -> Option<RequestId> {
    if let Some(text) = value.strip_prefix(b"\"") {
        let end = text.iter().position(|&b| b == b'"' || b == b'\\')?;
        let id = text.get(..end)?;
        return (text.get(end) == Some(&b'"'))
            .then(|| {
                std::str::from_utf8(id)
                    .ok()
                    .map(|id| RequestId::Str(id.to_owned()))
            })
            .flatten();
    }
    let digits = value.iter().take_while(|b| b.is_ascii_digit()).count();
    std::str::from_utf8(value.get(..digits)?)
        .ok()?
        .parse()
        .ok()
        .map(RequestId::Num)
}
