//! Gemini test doubles on Google's own JSON. [`Scripted`] answers a model's
//! requests with scripted or recorded `generateContent` replies, [`reply`]
//! builds them, and [`Exchanges`] reads a recorded cassette back through
//! Google's schema so a test can state what was sent.
//!
//! ```
//! use rig_cassette::gemini::{Scripted, reply};
//! use serde_json::json;
//!
//! let scripted = Scripted::new([
//!     reply::function_call("lookup_order", json!({"order_id": "A-17"}), Some("c2ln")),
//!     reply::text("Order A-17 was refunded."),
//! ]);
//! assert!(scripted.requests().is_empty());
//! ```

use std::collections::VecDeque;
use std::future::{self, Future};
use std::path::Path;
use std::sync::{Arc, Mutex, MutexGuard};

use bytes::Bytes;
use rig_core::completion::Usage;
use rig_core::http_client::{
    self, BoxedStream, HttpClientExt, LazyBody, MultipartForm, Request, Response, StatusCode,
    StreamingResponse,
};
use rig_core::providers::gemini::api::{self, Mirrored};
use rig_core::providers::gemini::edge;
use rig_core::wasm_compat::WasmCompatSend;
use serde::Deserialize;
use serde_json::value::RawValue;

/// Builders of `generateContent` replies, as Google sends them.
pub mod reply {
    use rig_core::providers::gemini::api;
    use serde_json::Value;

    fn reply(part: api::Part) -> api::GenerateContentResponse {
        api::GenerateContentResponse {
            candidates: vec![api::Candidate {
                content: Some(api::Content {
                    role: Some("model".to_owned()),
                    parts: vec![part],
                    ..Default::default()
                }),
                finish_reason: Some(api::FinishReason::Stop),
                index: Some(0),
                ..Default::default()
            }],
            usage_metadata: Some(api::UsageMetadata {
                prompt_token_count: Some(16),
                candidates_token_count: Some(8),
                total_token_count: Some(24),
                ..Default::default()
            }),
            model_version: Some(rig_core::providers::gemini::GEMINI_3_8_FLASH.to_owned()),
            ..Default::default()
        }
    }

    /// A reply of one text part.
    pub fn text(text: &str) -> api::GenerateContentResponse {
        reply(api::Part {
            text: Some(text.to_owned()),
            ..Default::default()
        })
    }

    /// A reply calling `name` with `args`, signed with `signature` when set.
    pub fn function_call(
        name: &str,
        args: Value,
        signature: Option<&str>,
    ) -> api::GenerateContentResponse {
        let args = match args {
            Value::Object(args) => args,
            other => serde_json::Map::from_iter([("value".to_owned(), other)]),
        };
        reply(api::Part {
            function_call: Some(api::FunctionCall {
                id: Some(format!("call-{name}")),
                name: Some(name.to_owned()),
                args: Some(args),
                ..Default::default()
            }),
            thought_signature: signature.map(str::to_owned),
            ..Default::default()
        })
    }
}

/// A transport that answers each request with the next reply, in order, and
/// keeps what was sent. A unary call gets the reply as one document; a
/// streamed call gets it as server-sent events.
#[derive(Clone, Debug, Default)]
pub struct Scripted {
    replies: Arc<Mutex<VecDeque<String>>>,
    requests: Arc<Mutex<Vec<Bytes>>>,
}

impl Scripted {
    /// A transport answering with `replies`, in order.
    pub fn new(replies: impl IntoIterator<Item = api::GenerateContentResponse>) -> Self {
        let replies = replies
            .into_iter()
            .map(|reply| serde_json::to_string(&reply).unwrap_or_default())
            .collect();
        Self {
            replies: Arc::new(Mutex::new(replies)),
            requests: Arc::default(),
        }
    }

    /// A transport answering with the `generateContent` replies recorded in
    /// the cassette at `path`, in order.
    pub fn from_cassette(path: &Path) -> Result<Self, ExchangesError> {
        let replies = recorded(path)?
            .into_iter()
            .filter(|interaction| is_generate(&interaction.path))
            .filter_map(|interaction| interaction.response)
            .collect();
        Ok(Self {
            replies: Arc::new(Mutex::new(replies)),
            requests: Arc::default(),
        })
    }

    /// Every `generateContent` request sent so far, as Google reads it.
    pub fn requests(&self) -> Vec<api::GenerateContentRequest> {
        lock(&self.requests)
            .iter()
            .filter_map(|body| serde_json::from_slice(body).ok())
            .collect()
    }

    fn next(&self, body: Bytes) -> Option<String> {
        lock(&self.requests).push(body);
        lock(&self.replies).pop_front()
    }
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn exhausted() -> http_client::Error {
    http_client::Error::InvalidStatusCodeWithDetails {
        status: StatusCode::NOT_IMPLEMENTED,
        body: "the script has no reply left".to_owned(),
        headers: http_client::HeaderMap::new(),
    }
}

/// A recorded or scripted reply as one server-sent event stream.
fn as_events(reply: String) -> String {
    if reply.trim_start().starts_with("data:") {
        reply
    } else {
        format!("data: {reply}\r\n\r\n")
    }
}

impl HttpClientExt for Scripted {
    fn send<T, U>(
        &self,
        req: Request<T>,
    ) -> impl Future<Output = http_client::Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        T: Into<Bytes> + WasmCompatSend,
        U: From<Bytes> + WasmCompatSend + 'static,
    {
        let reply = self.next(req.into_body().into());
        async move {
            let reply = reply.ok_or_else(exhausted)?;
            let body: LazyBody<U> = Box::pin(async move { Ok(U::from(Bytes::from(reply))) });
            Response::builder()
                .status(StatusCode::OK)
                .body(body)
                .map_err(http_client::Error::Protocol)
        }
    }

    fn send_multipart<U>(
        &self,
        _req: Request<MultipartForm>,
    ) -> impl Future<Output = http_client::Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        U: From<Bytes> + WasmCompatSend + 'static,
    {
        future::ready(Err(exhausted()))
    }

    fn send_streaming<T>(
        &self,
        req: Request<T>,
    ) -> impl Future<Output = http_client::Result<StreamingResponse>> + WasmCompatSend
    where
        T: Into<Bytes> + WasmCompatSend,
    {
        let reply = self.next(req.into_body().into());
        future::ready(reply.ok_or_else(exhausted).and_then(|reply| {
            let chunks: BoxedStream =
                Box::pin(futures::stream::iter([Ok::<Bytes, http_client::Error>(
                    Bytes::from(as_events(reply)),
                )]));
            Response::builder()
                .status(StatusCode::OK)
                .header("content-type", "text/event-stream")
                .body(chunks)
                .map_err(http_client::Error::Protocol)
        }))
    }
}

/// Why a cassette could not be read as Gemini exchanges.
#[derive(Debug)]
pub enum ExchangesError {
    /// The cassette could not be read.
    Io(std::io::Error),
    /// A YAML document is malformed.
    Yaml(serde_yaml::Error),
    /// A body does not parse as Google's schema.
    Schema {
        /// Where the body is: the request path and which side.
        origin: String,
        /// The parse error.
        error: serde_json::Error,
    },
    /// A request does not re-send what the reply before it returned.
    NotReplayed {
        /// The index of the request.
        request: usize,
        /// The signature or native part JSON it lacks.
        missing: String,
    },
}

impl std::fmt::Display for ExchangesError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(f, "reading the cassette: {error}"),
            Self::Yaml(error) => write!(f, "a cassette document is malformed: {error}"),
            Self::Schema { origin, error } => {
                write!(f, "{origin} does not parse as Google's schema: {error}")
            }
            Self::NotReplayed { request, missing } => {
                write!(f, "request {request} does not re-send {missing}")
            }
        }
    }
}

impl std::error::Error for ExchangesError {}

/// One recorded interaction: its path and bodies.
struct Recorded {
    path: String,
    request: Option<String>,
    response: Option<String>,
}

fn recorded(path: &Path) -> Result<Vec<Recorded>, ExchangesError> {
    let text = std::fs::read_to_string(path).map_err(ExchangesError::Io)?;
    let mut out = Vec::new();
    for document in serde_yaml::Deserializer::from_str(&text) {
        let document = serde_yaml::Value::deserialize(document).map_err(ExchangesError::Yaml)?;
        let field = |side: &str, key: &str| {
            document
                .get(side)
                .and_then(|side| side.get(key))
                .and_then(serde_yaml::Value::as_str)
                .map(str::to_owned)
        };
        let Some(path) = field("when", "path") else {
            continue;
        };
        out.push(Recorded {
            path,
            request: field("when", "body").filter(|body| !body.is_empty()),
            response: field("then", "body").filter(|body| !body.is_empty()),
        });
    }
    Ok(out)
}

fn is_generate(path: &str) -> bool {
    path.contains(":generateContent") || path.contains(":streamGenerateContent")
}

/// The `data:` payloads of a server-sent event body, or the body itself when
/// it is one document.
fn documents(body: &str) -> Vec<&str> {
    let events: Vec<&str> = body
        .lines()
        .filter_map(|line| line.strip_prefix("data:"))
        .map(str::trim)
        .filter(|event| !event.is_empty() && *event != "[DONE]")
        .collect();
    if events.is_empty() {
        vec![body]
    } else {
        events
    }
}

/// One recorded `generateContent` exchange.
#[derive(Debug)]
pub struct Exchange {
    /// The request path.
    pub path: String,
    /// What rig sent.
    pub request: api::GenerateContentRequest,
    /// What Google answered: one document, or each streamed chunk.
    pub responses: Vec<api::GenerateContentResponse>,
    request_text: String,
    response_parts: Vec<Box<RawValue>>,
}

impl Exchange {
    /// The reply's usage by rig's rule, from the last chunk that reported it.
    pub fn usage(&self) -> Option<Usage> {
        self.responses
            .iter()
            .rev()
            .find_map(|response| response.usage_metadata.as_ref())
            .map(|usage| {
                edge::usage(edge::Counts {
                    prompt: edge::count(usage.prompt_token_count),
                    tool_use_prompt: edge::count(usage.tool_use_prompt_token_count),
                    cached: edge::count(usage.cached_content_token_count),
                    candidates: edge::count(usage.candidates_token_count),
                    thoughts: edge::count(usage.thoughts_token_count),
                })
            })
    }

    /// The request body exactly as it was sent.
    pub fn request_text(&self) -> &str {
        &self.request_text
    }
}

/// The `generateContent` exchanges of one recorded cassette, in order.
#[derive(Debug)]
pub struct Exchanges {
    turns: Vec<Exchange>,
}

impl Exchanges {
    /// The exchanges recorded at `path`. Every request and response body must
    /// parse as Google's schema.
    pub fn from_cassette(path: &Path) -> Result<Self, ExchangesError> {
        let mut turns = Vec::new();
        for interaction in recorded(path)? {
            if !is_generate(&interaction.path) {
                continue;
            }
            let schema = |side: &str, error| ExchangesError::Schema {
                origin: format!("{} {side}", interaction.path),
                error,
            };
            let request_text = interaction.request.unwrap_or_default();
            let request =
                serde_json::from_str(&request_text).map_err(|error| schema("request", error))?;
            let mut responses = Vec::new();
            let mut response_parts = Vec::new();
            if let Some(body) = &interaction.response {
                for document in documents(body) {
                    responses.push(
                        serde_json::from_str(document)
                            .map_err(|error| schema("response", error))?,
                    );
                    response_parts
                        .extend(raw_parts(document).map_err(|error| schema("response", error))?);
                }
            }
            turns.push(Exchange {
                path: interaction.path,
                request,
                responses,
                request_text,
                response_parts,
            });
        }
        Ok(Self { turns })
    }

    /// How many exchanges were recorded.
    pub fn len(&self) -> usize {
        self.turns.len()
    }

    /// Whether none was.
    pub fn is_empty(&self) -> bool {
        self.turns.is_empty()
    }

    /// The exchanges, in order.
    pub fn turns(&self) -> &[Exchange] {
        &self.turns
    }

    /// Every field of every body the mirror does not type.
    pub fn unmodeled(&self) -> Vec<String> {
        let mut fields = Vec::new();
        for (index, exchange) in self.turns.iter().enumerate() {
            exchange
                .request
                .unmodeled_fields(&format!("[{index}].request"), &mut fields);
            for (chunk, response) in exchange.responses.iter().enumerate() {
                response.unmodeled_fields(&format!("[{index}].response[{chunk}]"), &mut fields);
            }
        }
        fields
    }

    /// Check that each reply's signatures and native parts are re-sent
    /// unchanged by the request after it: every `thoughtSignature` string and
    /// the exact JSON of every part rig keeps native.
    pub fn assert_replayed_verbatim(&self) -> Result<(), ExchangesError> {
        for (index, pair) in self.turns.windows(2).enumerate() {
            let [reply, next] = pair else { continue };
            for part in &reply.response_parts {
                let typed: api::Part =
                    serde_json::from_str(part.get()).map_err(|error| ExchangesError::Schema {
                        origin: format!("{} response part", reply.path),
                        error,
                    })?;
                let not_replayed = |missing: String| ExchangesError::NotReplayed {
                    request: index + 1,
                    missing,
                };
                if let Some(signature) =
                    typed.thought_signature.as_deref().filter(|s| !s.is_empty())
                {
                    let spelled = format!("\"thoughtSignature\":\"{signature}\"");
                    if !next.request_text.contains(&spelled) {
                        return Err(not_replayed(spelled));
                    }
                }
                if is_native(&typed) && !next.request_text.contains(part.get()) {
                    return Err(not_replayed(part.get().to_owned()));
                }
            }
        }
        Ok(())
    }
}

/// The raw parts of a response document's first candidate.
fn raw_parts(document: &str) -> Result<Vec<Box<RawValue>>, serde_json::Error> {
    #[derive(Deserialize)]
    struct Document {
        #[serde(default)]
        candidates: Vec<Candidate>,
    }
    #[derive(Deserialize)]
    struct Candidate {
        #[serde(default)]
        content: Option<Content>,
    }
    #[derive(Deserialize)]
    struct Content {
        #[serde(default)]
        parts: Vec<Box<RawValue>>,
    }
    let document: Document = serde_json::from_str(document)?;
    Ok(document
        .candidates
        .into_iter()
        .next()
        .and_then(|candidate| candidate.content)
        .map(|content| content.parts)
        .unwrap_or_default())
}

/// Whether rig keeps `part` native: anything but text, a thought or a whole
/// function call.
fn is_native(part: &api::Part) -> bool {
    let bare = api::Part {
        text: None,
        thought: None,
        thought_signature: None,
        function_call: None,
        ..part.clone()
    };
    if bare != api::Part::default() {
        return true;
    }
    match (&part.text, part.thought, &part.function_call) {
        (Some(_), None | Some(true), None) => false,
        (None, None, Some(call)) => !call.unmodeled.is_empty() || call.name.is_none(),
        _ => true,
    }
}

#[cfg(test)]
mod tests;
