//! The reply bank: one real recorded reply per provider, completion encoder,
//! reply shape and called tools, extracted from the cassette corpus by
//! `cargo xtask cassette bank` into `crates/rig-cassette/fixtures/bank/`.
//!
//! A runtime scenario asks the bank for the replies it needs in the order it
//! needs them and serves them over a [`BankHttpClient`], so each reply
//! goes through the provider's real decoder while the scenario runs once, not
//! once per provider's cassette. It names them by the scenario whose reply
//! shapes it runs against ([`script`], read off `scripts.tsv`) or by what
//! each reply must be ([`Step`]). The transport does not read
//! what Rig sends; request encoding is pinned by the request snapshots and
//! the acceptance index.

use std::collections::{HashMap, VecDeque};
use std::future::Future;
use std::sync::{Arc, LazyLock, Mutex};

use base64::Engine as _;
use futures::StreamExt as _;
use http::{HeaderMap, HeaderName, HeaderValue, StatusCode};
use rig_agent::test_utils::MockHttpResponse;
use rig_core::http_client::{
    DynHttpClient, HttpClientExt, LazyBody, MultipartForm, Request, Response, StreamingResponse,
};
use rig_core::wasm_compat::WasmCompatSend;
use serde::Deserialize;

/// The bank's directory.
pub fn bank_root() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../crates/rig-cassette/fixtures/bank")
}

/// One recorded header.
#[derive(Clone, Debug, Deserialize)]
pub struct Header {
    pub name: String,
    pub value: String,
}

/// One recorded reply, as the cassette held it.
#[derive(Clone, Debug, Deserialize)]
pub struct Reply {
    pub status: u16,
    #[serde(default)]
    pub header: Vec<Header>,
    #[serde(default)]
    pub body: Option<String>,
    #[serde(default)]
    pub body_encoding: Option<String>,
}

/// One bank entry.
#[derive(Clone, Debug, Deserialize)]
pub struct Entry {
    pub provider: String,
    pub encoder: String,
    pub shape: String,
    /// The distinct names of the tools the reply calls, sorted.
    pub calls: Vec<String>,
    /// The distinct values of the reply's ending keys, sorted.
    pub ends: Vec<String>,
    /// The interaction the reply was taken from.
    pub source: String,
    pub then: Reply,
}

impl Entry {
    /// The reply's content type, lowercased.
    pub fn content_type(&self) -> String {
        self.then
            .header
            .iter()
            .find(|header| header.name.eq_ignore_ascii_case("content-type"))
            .map_or_else(String::new, |header| header.value.to_ascii_lowercase())
    }

    /// Whether the reply is a stream of server-sent events, by its content
    /// type or, recorded without one, by its first bytes.
    pub fn is_sse(&self) -> bool {
        let body = self.then.body.as_deref().unwrap_or_default();
        self.content_type().starts_with("text/event-stream")
            || body.starts_with("event:")
            || body.starts_with("data:")
    }

    /// Whether the reply is a stream.
    pub fn streamed(&self) -> bool {
        let content_type = self.content_type();
        self.is_sse()
            || content_type.starts_with("application/vnd.amazon.eventstream")
            || content_type.starts_with("application/x-ndjson")
    }

    /// The body's bytes.
    pub fn body(&self) -> bytes::Bytes {
        let body = self.then.body.clone().unwrap_or_default();
        if self.then.body_encoding.as_deref() == Some("base64") {
            let decoded = base64::engine::general_purpose::STANDARD
                .decode(body.as_bytes())
                .unwrap_or_else(|error| panic!("{}: base64 body: {error}", self.source));
            bytes::Bytes::from(decoded)
        } else {
            bytes::Bytes::from(body)
        }
    }

    /// The reply's status, body and headers.
    pub fn parts(&self) -> (StatusCode, bytes::Bytes, HeaderMap) {
        let mut headers = HeaderMap::new();
        for header in &self.then.header {
            if let (Ok(name), Ok(value)) = (
                HeaderName::from_bytes(header.name.as_bytes()),
                HeaderValue::from_str(&header.value),
            ) {
                headers.append(name, value);
            }
        }
        let status = StatusCode::from_u16(self.then.status)
            .unwrap_or_else(|error| panic!("{}: status: {error}", self.source));
        (status, self.body(), headers)
    }

    /// The reply as the bundled transport delivers it: a success with its
    /// headers, or a status error carrying the body and headers.
    pub fn response(&self) -> MockHttpResponse {
        let (status, body, headers) = self.parts();
        if status.is_success() {
            MockHttpResponse::SuccessWithHeaders(body, headers)
        } else {
            MockHttpResponse::ErrorWithHeaders(
                status,
                String::from_utf8_lossy(&body).into_owned(),
                headers,
            )
        }
    }

    /// The scenario directory of the source, `<provider>/<dir>`.
    pub fn source_dir(&self) -> &str {
        self.source
            .rsplit_once('/')
            .map_or(self.source.as_str(), |(dir, _)| dir)
    }

    /// The portable class of the reply's ending.
    pub fn ending(&self) -> Ending {
        Ending::of(&self.ends)
    }

    /// The entry's key in a script: `encoder;shape;calls`.
    pub fn script_key(&self) -> String {
        format!("{};{};{}", self.encoder, self.shape, self.calls.join(","))
    }
}

static SCRIPTS: LazyLock<HashMap<String, Vec<Option<String>>>> = LazyLock::new(|| {
    let path = bank_root().join("scripts.tsv");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    text.lines()
        .skip(1)
        .filter_map(|line| {
            let mut columns = line.split('\t');
            let fixture = columns.next()?.to_owned();
            let keys = columns
                .map(|key| (key != "-").then(|| key.to_owned()))
                .collect();
            Some((fixture, keys))
        })
        .collect()
});

static PINNED: LazyLock<Vec<Entry>> = LazyLock::new(|| {
    let path = bank_root().join("pinned.yaml");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    serde_yaml::Deserializer::from_str(&text)
        .map(|document| {
            Entry::deserialize(document)
                .unwrap_or_else(|error| panic!("{}: {error}", path.display()))
        })
        .collect()
});

/// The replies `provider`'s `scenario` recorded, verbatim, in its order:
/// for a runtime scenario whose assertions read what a reply says. The
/// fixture must be listed in the bank's `pinned.txt`.
pub fn recorded(provider: &str, scenario: &str) -> Vec<Entry> {
    let fixture = format!("{provider}/{scenario}.yaml#");
    let replies: Vec<Entry> = PINNED
        .iter()
        .filter(|entry| entry.source.starts_with(&fixture))
        .cloned()
        .collect();
    assert!(
        !replies.is_empty(),
        "{provider}/{scenario}.yaml is not pinned: list it in the bank's pinned.txt with its reason"
    );
    replies
}

/// The bank's replies of the shapes `provider`'s `scenario` recorded, in its
/// order: what a runtime scenario that recorded `scenario` is served once its
/// own cassette is not read. Panics when the scenario recorded a reply the
/// bank does not hold (a non-completion exchange).
pub fn script(provider: &str, scenario: &str) -> Vec<Entry> {
    let fixture = format!("{provider}/{scenario}.yaml");
    let keys = SCRIPTS
        .get(&fixture)
        .unwrap_or_else(|| panic!("the bank has no script for {fixture}"));
    let bank = entries(provider);
    keys.iter()
        .enumerate()
        .map(|(index, key)| {
            let key = key
                .as_deref()
                .unwrap_or_else(|| panic!("{fixture}#{index} is not a completion reply"));
            bank.iter()
                .find(|entry| entry.script_key() == key)
                .cloned()
                .unwrap_or_else(|| panic!("{fixture}#{index}: the bank holds no {key}"))
        })
        .collect()
}

/// How a reply's turn ended, read off its ending values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ending {
    /// The model finished its turn (`stop`, `end_turn`, `STOP`, `completed`).
    Stop,
    /// The model asked for tools (`tool_calls`, `tool_use`).
    Tool,
    /// The output cap cut the turn (`length`, `max_tokens`, ...).
    Length,
    /// A content filter or refusal ended it.
    Filtered,
    /// No ending value, or several classes at once.
    Other,
}

impl Ending {
    fn of(ends: &[String]) -> Self {
        let class = |value: &str| match value {
            "stop" | "end_turn" | "STOP" | "completed" | "COMPLETE" | "stop_sequence" => Self::Stop,
            "tool_calls" | "tool_use" | "function_call" => Self::Tool,
            "length" | "max_tokens" | "MAX_TOKENS" | "max_output_tokens" | "incomplete" => {
                Self::Length
            }
            "content_filter" | "SAFETY" | "refusal" | "RECITATION" | "PROHIBITED_CONTENT" => {
                Self::Filtered
            }
            _ => Self::Other,
        };
        // A Responses stream also names the states it passed through.
        let mut classes: Vec<Self> = ends
            .iter()
            .filter(|value| !matches!(value.as_str(), "in_progress" | "queued"))
            .map(|value| class(value))
            .collect();
        classes.sort_by_key(|class| *class as u8);
        classes.dedup();
        match classes.as_slice() {
            [one] => *one,
            // Gemini and Responses end a tool turn with `STOP`/`completed`.
            [Self::Stop, Self::Tool] => Self::Tool,
            _ => Self::Other,
        }
    }
}

static BANK: LazyLock<Mutex<HashMap<String, Arc<Vec<Entry>>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Every entry of `provider`'s bank file, in file order.
pub fn entries(provider: &str) -> Arc<Vec<Entry>> {
    let mut bank = BANK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    bank.entry(provider.to_owned())
        .or_insert_with(|| {
            let path = bank_root().join(format!("{provider}.yaml"));
            let text = std::fs::read_to_string(&path)
                .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
            let entries: Vec<Entry> = serde_yaml::Deserializer::from_str(&text)
                .map(|document| {
                    Entry::deserialize(document)
                        .unwrap_or_else(|error| panic!("{}: {error}", path.display()))
                })
                .collect();
            Arc::new(entries)
        })
        .clone()
}

/// Every provider the bank holds replies of, in name order.
pub fn providers() -> Vec<String> {
    let dir = bank_root();
    let mut providers: Vec<String> = std::fs::read_dir(&dir)
        .unwrap_or_else(|error| panic!("{}: {error}", dir.display()))
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            let stem = path.file_stem()?.to_str()?.to_owned();
            (path.extension()? == "yaml" && stem != "pinned").then_some(stem)
        })
        .collect();
    providers.sort();
    providers
}

/// One reply a scenario needs.
#[derive(Clone, Copy, Debug)]
pub enum Step {
    /// A finished text answer that calls no tool.
    Text,
    /// A turn that calls exactly these tools (by distinct name).
    Calls(&'static [&'static str]),
    /// A turn whose output cap cut it, calling no tool.
    Truncated,
    /// A reply with this non-success status.
    Status(u16),
}

impl Step {
    /// Whether `entry` is a reply this step takes.
    pub fn takes(self, entry: &Entry) -> bool {
        let status = entry.then.status;
        match self {
            Self::Text => status == 200 && entry.calls.is_empty() && entry.ending() == Ending::Stop,
            Self::Calls(names) => {
                status == 200
                    && entry.calls.len() == names.len()
                    && names
                        .iter()
                        .all(|name| entry.calls.iter().any(|call| call == name))
                    && matches!(entry.ending(), Ending::Tool | Ending::Stop)
            }
            Self::Truncated => {
                status == 200 && entry.calls.is_empty() && entry.ending() == Ending::Length
            }
            Self::Status(wanted) => status == wanted,
        }
    }
}

/// Where a scenario's replies come from: a provider's completion encoder and
/// the reply mode.
#[derive(Clone, Copy, Debug)]
pub struct Source {
    pub provider: &'static str,
    /// The encoder, as the bank names it (`POST /chat/completions`).
    pub encoder: &'static str,
    pub streamed: bool,
}

impl Source {
    /// Every entry of this source that `step` takes, in bank order.
    pub fn candidates(&self, step: Step) -> Vec<Entry> {
        entries(self.provider)
            .iter()
            .filter(|entry| {
                entry.encoder == self.encoder
                    && (matches!(step, Step::Status(_)) || entry.streamed() == self.streamed)
                    && step.takes(entry)
            })
            .cloned()
            .collect()
    }

    /// The reply `step` takes: among the candidates, the one whose source
    /// directory comes first in `prefer`, then the smallest, then the first in
    /// bank order. Panics when the bank holds none.
    pub fn pick(&self, step: Step, prefer: &[&str]) -> Entry {
        let rank = |entry: &Entry| {
            prefer
                .iter()
                .position(|dir| entry.source_dir().ends_with(dir))
                .unwrap_or(prefer.len())
        };
        self.candidates(step)
            .into_iter()
            .enumerate()
            .min_by_key(|(index, entry)| (rank(entry), entry.body().len(), *index))
            .map(|(_, entry)| entry)
            .unwrap_or_else(|| panic!("the bank holds no {step:?} reply for {self:?}"))
    }

    /// The replies of `steps`, in order.
    pub fn script(&self, steps: &[Step], prefer: &[&str]) -> Vec<Entry> {
        steps.iter().map(|step| self.pick(*step, prefer)).collect()
    }
}

/// A transport that answers each request with the next of its replies, in
/// order, and ignores what was sent. A streamed success arrives one
/// server-sent event per chunk, with a yield between chunks, as a socket
/// delivers a long stream: a consumer that stops at a delta drops the stream
/// before its end, as it does live. A status reply is the status error the
/// bundled transport reports.
#[derive(Clone, Debug, Default)]
pub struct BankHttpClient {
    replies: Arc<Mutex<VecDeque<Entry>>>,
}

impl BankHttpClient {
    /// A transport serving `replies` in order.
    pub fn new(replies: &[Entry]) -> Self {
        Self {
            replies: Arc::new(Mutex::new(replies.iter().cloned().collect())),
        }
    }

    /// The replies not served yet.
    pub fn remaining(&self) -> usize {
        self.replies
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len()
    }

    fn next(&self) -> Option<Entry> {
        self.replies
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .pop_front()
    }
}

/// The transport's answer when it has no reply left.
fn exhausted() -> rig_core::http_client::Error {
    rig_core::http_client::Error::InvalidStatusCodeWithDetails {
        status: StatusCode::NOT_IMPLEMENTED,
        body: "the bank transport has no reply left".to_owned(),
        headers: HeaderMap::new(),
    }
}

/// `body` cut after every blank line: one server-sent event per chunk.
pub fn frames(body: &[u8]) -> Vec<bytes::Bytes> {
    let mut chunks = Vec::new();
    let mut start = 0;
    let mut line = 0;
    for (index, byte) in body.iter().enumerate() {
        if *byte != b'\n' {
            continue;
        }
        let text = body.get(line..=index).unwrap_or_default();
        if text == b"\n" || text == b"\r\n" {
            chunks.push(bytes::Bytes::copy_from_slice(
                body.get(start..=index).unwrap_or_default(),
            ));
            start = index + 1;
        }
        line = index + 1;
    }
    if start < body.len() {
        chunks.push(bytes::Bytes::copy_from_slice(
            body.get(start..).unwrap_or_default(),
        ));
    }
    chunks
}

/// `entry`'s status, body and headers; a non-success status is the status
/// error the bundled transport reports.
fn delivered(
    entry: Option<Entry>,
) -> rig_core::http_client::Result<(Entry, bytes::Bytes, HeaderMap)> {
    let entry = entry.ok_or_else(exhausted)?;
    let (status, body, headers) = entry.parts();
    if !status.is_success() {
        return Err(rig_core::http_client::Error::InvalidStatusCodeWithDetails {
            status,
            body: String::from_utf8_lossy(&body).into_owned(),
            headers,
        });
    }
    Ok((entry, body, headers))
}

/// `entry` as the bundled transport delivers a unary reply.
fn unary<U>(entry: Option<Entry>) -> rig_core::http_client::Result<Response<LazyBody<U>>>
where
    U: From<bytes::Bytes> + WasmCompatSend + 'static,
{
    let (_, body, headers) = delivered(entry)?;
    let body: LazyBody<U> = Box::pin(async move { Ok(U::from(body)) });
    let mut response = Response::builder()
        .status(StatusCode::OK)
        .body(body)
        .map_err(rig_core::http_client::Error::Protocol)?;
    *response.headers_mut() = headers;
    Ok(response)
}

/// `entry` as the bundled transport delivers a streamed reply: a success
/// arrives one server-sent event per chunk.
fn streamed(entry: Option<Entry>) -> rig_core::http_client::Result<StreamingResponse> {
    let (entry, body, headers) = delivered(entry)?;
    let chunks = if entry.is_sse() {
        frames(&body)
    } else {
        vec![body]
    };
    let chunks: rig_core::http_client::BoxedStream =
        Box::pin(futures::stream::iter(chunks).then(|chunk| async move {
            tokio::task::yield_now().await;
            Ok::<_, rig_core::http_client::Error>(chunk)
        }));
    let mut response = Response::builder()
        .status(StatusCode::OK)
        .body(chunks)
        .map_err(rig_core::http_client::Error::Protocol)?;
    *response.headers_mut() = headers;
    Ok(response)
}

impl HttpClientExt for BankHttpClient {
    fn send<T, U>(
        &self,
        _req: Request<T>,
    ) -> impl Future<Output = rig_core::http_client::Result<Response<LazyBody<U>>>>
    + WasmCompatSend
    + 'static
    where
        T: Into<bytes::Bytes> + WasmCompatSend,
        U: From<bytes::Bytes> + WasmCompatSend + 'static,
    {
        std::future::ready(unary(self.next()))
    }

    fn send_multipart<U>(
        &self,
        _req: Request<MultipartForm>,
    ) -> impl Future<Output = rig_core::http_client::Result<Response<LazyBody<U>>>>
    + WasmCompatSend
    + 'static
    where
        U: From<bytes::Bytes> + WasmCompatSend + 'static,
    {
        std::future::ready(unary(self.next()))
    }

    fn send_streaming<T>(
        &self,
        _req: Request<T>,
    ) -> impl Future<Output = rig_core::http_client::Result<StreamingResponse>> + WasmCompatSend
    where
        T: Into<bytes::Bytes> + WasmCompatSend,
    {
        std::future::ready(streamed(self.next()))
    }
}

/// A transport serving `replies` in order, erased.
pub fn client(replies: &[Entry]) -> DynHttpClient {
    DynHttpClient::new(BankHttpClient::new(replies))
}
