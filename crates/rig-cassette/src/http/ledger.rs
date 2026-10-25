//! The created-resource ledger: provider-side state a recording creates, and
//! the cleanup pass that deletes it.
//!
//! A recording session appends one JSON line per created resource (a stored
//! response, an uploaded file, a context cache, a stored interaction, a
//! conversation) the moment its reply arrives, before the test sees the reply
//! and so before anything can panic. Each line carries the provider's delete
//! URL. [`clean_up`] later deletes every resource the ledger has not already
//! seen deleted, outside any cassette, and appends each outcome; a 404 (or
//! Gemini's 403 "not found") means the test already deleted it.
//!
//! ```
//! use rig_cassette::http::ledger::created_resources;
//! let created = created_resources(
//!     "openai",
//!     "https://api.openai.com",
//!     "POST",
//!     "/v1/responses",
//!     br#"{"model":"gpt-4.1-nano","input":"hi"}"#,
//!     br#"{"id":"resp_1","object":"response"}"#,
//! );
//! assert_eq!(created[0].delete_url, "https://api.openai.com/v1/responses/resp_1");
//! ```

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use rig_core::http_client::Bytes;
use rig_core::http_client::framing::{Framing, SseEvent, SseFramer};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The kinds of provider-side state a recording can leave behind.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourceKind {
    /// A stored Responses API response (OpenAI, xAI).
    Response,
    /// An uploaded file.
    File,
    /// A Gemini context cache.
    CachedContent,
    /// A stored Gemini interaction.
    Interaction,
    /// An OpenAI conversation.
    Conversation,
    /// An OpenAI vector store.
    VectorStore,
}

/// One resource a recording created.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CreatedResource {
    /// The cassette provider (`openai`, `gemini`, …), which selects the credential.
    pub provider: String,
    /// The scenario whose session created it.
    pub scenario: String,
    /// What it is.
    pub kind: ResourceKind,
    /// The provider's id or resource name.
    pub id: String,
    /// The URL a `DELETE` removes it at.
    pub delete_url: String,
}

/// One ledger line: a creation or a cleanup outcome.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum LedgerEntry {
    /// A recording created the resource.
    Created(CreatedResource),
    /// A cleanup `DELETE` answered.
    Cleanup {
        /// The URL deleted.
        delete_url: String,
        /// The outcome.
        outcome: CleanupOutcome,
        /// The HTTP status, when the request completed.
        status: Option<u16>,
    },
}

/// What a cleanup `DELETE` achieved.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CleanupOutcome {
    /// The provider deleted it.
    Deleted,
    /// It was already gone (404, or Gemini's 403 "not found"): the test
    /// deleted it, or it expired.
    Gone,
    /// The request failed or the provider refused it; a later pass retries.
    Failed,
    /// No credential for the provider was available; a later pass retries.
    NoCredential,
}

/// The ledger's file name under an attempt root.
pub const LEDGER_FILE: &str = "ledger.jsonl";

/// The ledger file: [`LEDGER_FILE`] under the recorder's attempt root.
pub fn ledger_path() -> PathBuf {
    super::attempt_root().join(LEDGER_FILE)
}

/// Append `entries` to the ledger at `path`, one JSON line each, flushed
/// before returning. Errors are reported on stderr, never raised: a ledger
/// write must not change what the recording does.
pub fn append(path: &Path, entries: &[LedgerEntry]) {
    if entries.is_empty() {
        return;
    }
    let result = (|| -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        let mut lines = String::new();
        for entry in entries {
            lines.push_str(&serde_json::to_string(entry).map_err(std::io::Error::other)?);
            lines.push('\n');
        }
        file.write_all(lines.as_bytes())?;
        file.sync_data()
    })();
    if let Err(error) = result {
        eprintln!("cassette ledger {} not written: {error}", path.display());
    }
}

/// Where a recording session's ledger lines go and whose resources they are.
#[derive(Clone, Debug)]
pub(crate) struct LedgerTarget {
    pub(crate) path: PathBuf,
    pub(crate) provider: String,
    pub(crate) scenario: String,
    /// The real provider's scheme and host, for delete URLs.
    pub(crate) origin: String,
}

/// A request as the ledger reads it: its method, URI, body as sent, and the
/// body's `content-encoding`.
pub(crate) struct Sent<'a> {
    pub(crate) method: &'a str,
    pub(crate) uri: &'a str,
    pub(crate) body: &'a [u8],
    pub(crate) encoding: Option<&'a str>,
}

/// Append a line to `target` for each resource a whole reply with `status`
/// and `body`, encoded by `encoding`, to `request` created; nothing for a
/// reply that is not a success.
pub(crate) fn record_created(
    target: &Arc<LedgerTarget>,
    request: Sent<'_>,
    status: u16,
    body: &[u8],
    encoding: Option<&str>,
) {
    if let Some(ledger) = ReplyLedger::start(target.clone(), request, status, false, encoding) {
        ledger.finish(body);
    }
}

/// The ledger of one successful reply to a request that can create a
/// resource. An event stream is read event by event as its chunks arrive
/// ([`push`](Self::push)), so a resource is on disk before the caller can
/// read the event that names it, even when the stream later fails; any other
/// body, and a compressed event stream, is read when it ends
/// ([`finish`](Self::finish)), decoded from its `content-encoding`. Each
/// resource is written once.
pub(crate) struct ReplyLedger {
    target: Arc<LedgerTarget>,
    kind: ResourceKind,
    path: String,
    /// The event stream's framer; `None` for a body read whole.
    events: Option<SseFramer>,
    /// The reply's `content-encoding`, which a body read whole is decoded
    /// from.
    encoding: Option<String>,
    written: Vec<String>,
}

impl ReplyLedger {
    /// The ledger of a reply with `status`, encoded by `encoding`, to
    /// `request`; `None` when the reply is not a success or the request
    /// creates nothing.
    pub(crate) fn start(
        target: Arc<LedgerTarget>,
        request: Sent<'_>,
        status: u16,
        event_stream: bool,
        encoding: Option<&str>,
    ) -> Option<Self> {
        if !(200..300).contains(&status) {
            return None;
        }
        let path = super::uri_path(request.uri);
        // What the request asked for, such as `store: false`, is read decoded;
        // a body that does not decode is read as it is, so it reads as
        // storing and its resource is cleaned up.
        let request_body = crate::scrub::decoded(request.body, request.encoding).map_or_else(
            |_| Bytes::copy_from_slice(request.body),
            |decoded| decoded.bytes,
        );
        let kind = creates(&target.provider, request.method, &path, &request_body)?;
        // A compressed event stream cannot be read event by event, so it is
        // read whole when it ends, decoded.
        let compressed =
            encoding.is_some_and(|encoding| !encoding.trim().eq_ignore_ascii_case("identity"));
        Some(Self {
            target,
            kind,
            path: request_path(&path).to_owned(),
            events: (event_stream && !compressed).then(SseFramer::new),
            encoding: encoding.map(str::to_owned),
            written: Vec::new(),
        })
    }

    /// Write what the events `chunk` completes created.
    pub(crate) fn push(&mut self, chunk: &[u8]) {
        let Some(framer) = &mut self.events else {
            return;
        };
        let documents = event_documents(framer.push(chunk));
        self.write(&documents);
    }

    /// The reply ended with `body`: write what a body read whole created, or
    /// what an event stream's last event created when the stream ended
    /// before that event's end; its other events were written as they
    /// completed.
    pub(crate) fn finish(mut self, body: &[u8]) {
        let documents = match &mut self.events {
            Some(framer) => event_documents(
                framer
                    .finish_recording()
                    .events
                    .into_iter()
                    .map(|recorded| recorded.event),
            ),
            // A body that does not decode names nothing the ledger can read;
            // its recording is refused.
            None => crate::scrub::decoded(body, self.encoding.as_deref())
                .map(|decoded| response_documents(&decoded.bytes))
                .unwrap_or_default(),
        };
        self.write(&documents);
    }

    fn write(&mut self, documents: &[Value]) {
        let target = &self.target;
        let entries: Vec<LedgerEntry> = new_resources(
            &target.provider,
            &target.origin,
            self.kind,
            &self.path,
            documents,
            &mut self.written,
        )
        .into_iter()
        .map(|resource| {
            LedgerEntry::Created(CreatedResource {
                scenario: target.scenario.clone(),
                ..resource
            })
        })
        .collect();
        append(&target.path, &entries);
    }
}

/// The JSON documents `events` hold; a heartbeat, with only whitespace for
/// data, holds none.
fn event_documents(events: impl IntoIterator<Item = SseEvent>) -> Vec<Value> {
    events
        .into_iter()
        .filter(|event| !event.data.trim().is_empty())
        .filter_map(|event| serde_json::from_str(&event.data).ok())
        .collect()
}

/// The resources a successful reply created. `origin` is the provider's
/// scheme and host; `path` the request path the recording saw. A Responses
/// or Interactions request that sets `store: false` creates nothing, and an
/// error reply creates nothing.
pub fn created_resources(
    provider: &str,
    origin: &str,
    method: &str,
    path: &str,
    request_body: &[u8],
    response_body: &[u8],
) -> Vec<CreatedResource> {
    let Some(kind) = creates(provider, method, path, request_body) else {
        return Vec::new();
    };
    new_resources(
        provider,
        origin,
        kind,
        request_path(path),
        &response_documents(response_body),
        &mut Vec::new(),
    )
}

/// The `kind` resources `documents` name on `provider` at `origin` for a
/// request to `path`, each once and none whose id `seen` holds; `seen` gains
/// their ids, so a reader of a stream keeps it across chunks. Their
/// `scenario` is the caller's to set.
fn new_resources(
    provider: &str,
    origin: &str,
    kind: ResourceKind,
    path: &str,
    documents: &[Value],
    seen: &mut Vec<String>,
) -> Vec<CreatedResource> {
    let mut created = Vec::new();
    for id in documents
        .iter()
        .filter_map(|document| created_id(kind, document))
    {
        if seen.contains(&id) {
            continue;
        }
        seen.push(id.clone());
        created.push(CreatedResource {
            provider: provider.to_owned(),
            scenario: String::new(),
            kind,
            delete_url: delete_url(kind, origin, path, &id),
            id,
        });
    }
    created
}

/// `path` without its query or a trailing slash.
fn request_path(path: &str) -> &str {
    path.split('?').next().unwrap_or(path).trim_end_matches('/')
}

/// The kind of resource a `method path` request with `request_body` creates
/// on `provider`: a `POST` to a creating route, unless a Responses or
/// Interactions request sets `store: false`.
fn creates(provider: &str, method: &str, path: &str, request_body: &[u8]) -> Option<ResourceKind> {
    if !method.eq_ignore_ascii_case("POST") {
        return None;
    }
    let kind = resource_kind(provider, request_path(path))?;
    let stores = serde_json::from_slice::<Value>(request_body)
        .ok()
        .and_then(|request| request.get("store").and_then(Value::as_bool))
        .unwrap_or(true);
    let opted_out = matches!(kind, ResourceKind::Response | ResourceKind::Interaction) && !stores;
    (!opted_out).then_some(kind)
}

fn resource_kind(provider: &str, path: &str) -> Option<ResourceKind> {
    let last = path.rsplit('/').next().unwrap_or_default();
    match last {
        // Only OpenAI and xAI store responses; other Responses routes
        // (OpenRouter, local servers) keep nothing.
        "responses" if super::STORING_RESPONSES_PROVIDERS.contains(&provider) => {
            Some(ResourceKind::Response)
        }
        "files" => Some(ResourceKind::File),
        "cachedContents" => Some(ResourceKind::CachedContent),
        "interactions" => Some(ResourceKind::Interaction),
        "conversations" => Some(ResourceKind::Conversation),
        "vector_stores" => Some(ResourceKind::VectorStore),
        _ => None,
    }
}

/// The id a creation reply names: the object's `id`, or a Gemini resource
/// `name`. A stream names it on its creation event.
fn created_id(kind: ResourceKind, document: &Value) -> Option<String> {
    let owner = match kind {
        ResourceKind::Response => document.get("response").unwrap_or(document),
        ResourceKind::Interaction => document.get("interaction").unwrap_or(document),
        ResourceKind::File => document.get("file").unwrap_or(document),
        _ => document,
    };
    let field = match kind {
        ResourceKind::CachedContent => "name",
        ResourceKind::File if owner.get("name").is_some() && owner.get("id").is_none() => "name",
        _ => "id",
    };
    owner
        .get(field)
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty())
        .map(str::to_owned)
}

fn delete_url(kind: ResourceKind, origin: &str, path: &str, id: &str) -> String {
    match kind {
        // Gemini names resources `files/<id>` and `cachedContents/<id>` and
        // deletes them under the API version, not the upload route.
        ResourceKind::CachedContent | ResourceKind::File if id.contains('/') => {
            let version = path
                .split('/')
                .find(|segment| {
                    segment.starts_with('v') && segment[1..].starts_with(char::is_numeric)
                })
                .unwrap_or("v1beta");
            format!("{origin}/{version}/{id}")
        }
        _ => format!("{origin}{path}/{id}"),
    }
}

/// The JSON documents of a recorded reply body: the body itself, each line
/// of an NDJSON stream, or each SSE event's data.
pub fn response_documents(body: &[u8]) -> Vec<Value> {
    if let Ok(value) = serde_json::from_slice(body) {
        return vec![value];
    }
    let framing = match body.trim_ascii_start().first() {
        Some(b'{' | b'[') => Framing::Ndjson,
        _ => Framing::Sse,
    };
    let frames = framing.split(body);
    frames
        .iter()
        .filter_map(|frame| serde_json::from_slice(frame.as_bytes()).ok())
        .collect()
}

/// Headers that authorize a cleanup `DELETE` for one provider.
pub type Credential = Vec<(String, String)>;

/// The credential for `provider` from its usual environment variable, with
/// the headers its delete endpoints need.
pub fn credential_from_env(provider: &str) -> Option<Credential> {
    let key = |name: &str| std::env::var(name).ok().filter(|key| !key.is_empty());
    match provider {
        "openai" => {
            key("OPENAI_API_KEY").map(|key| vec![("authorization".into(), format!("Bearer {key}"))])
        }
        "xai" => {
            key("XAI_API_KEY").map(|key| vec![("authorization".into(), format!("Bearer {key}"))])
        }
        "anthropic" => key("ANTHROPIC_API_KEY").map(|key| {
            vec![
                ("x-api-key".into(), key),
                ("anthropic-version".into(), "2023-06-01".into()),
                ("anthropic-beta".into(), "files-api-2025-04-14".into()),
            ]
        }),
        "gemini" => key("GEMINI_API_KEY").map(|key| vec![("x-goog-api-key".into(), key)]),
        _ => None,
    }
}

/// What one cleanup pass did.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CleanupReport {
    /// Resources deleted now.
    pub deleted: usize,
    /// Resources already gone.
    pub gone: usize,
    /// Resources a `DELETE` failed for; retried next pass.
    pub remaining: usize,
    /// Resources of a provider with no credential available; retried next pass.
    pub no_credential: usize,
}

/// The resources the ledger at `path` holds that no earlier pass deleted or
/// found gone, in creation order.
pub fn outstanding(path: &Path) -> Vec<CreatedResource> {
    let Ok(contents) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    let entries: Vec<LedgerEntry> = contents
        .lines()
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect();
    let settled: std::collections::BTreeSet<&str> = entries
        .iter()
        .filter_map(|entry| match entry {
            LedgerEntry::Cleanup {
                delete_url,
                outcome: CleanupOutcome::Deleted | CleanupOutcome::Gone,
                ..
            } => Some(delete_url.as_str()),
            _ => None,
        })
        .collect();
    let mut seen = std::collections::BTreeSet::new();
    entries
        .iter()
        .filter_map(|entry| match entry {
            LedgerEntry::Created(resource)
                if !settled.contains(resource.delete_url.as_str())
                    && seen.insert(resource.delete_url.clone()) =>
            {
                Some(resource.clone())
            }
            _ => None,
        })
        .collect()
}

/// Delete every outstanding resource in the ledger at `path`, appending each
/// outcome to it. `credential` supplies the headers for a provider; a
/// provider without one is left for a later pass.
pub async fn clean_up(
    path: &Path,
    credential: impl Fn(&str) -> Option<Credential>,
) -> CleanupReport {
    let client = rig_reqwest::reqwest::Client::builder()
        .no_proxy()
        .build()
        .unwrap_or_default();
    let mut report = CleanupReport::default();
    for resource in outstanding(path) {
        let (outcome, status) = match credential(&resource.provider) {
            None => (CleanupOutcome::NoCredential, None),
            Some(headers) => {
                let mut request = client.delete(&resource.delete_url);
                for (name, value) in headers {
                    request = request.header(name, value);
                }
                match request.send().await {
                    Ok(response) => {
                        let status = response.status().as_u16();
                        let outcome = match status {
                            200..=299 => CleanupOutcome::Deleted,
                            404 => CleanupOutcome::Gone,
                            // Gemini answers a deleted cache or file with
                            // "CachedContent not found (or permission denied)".
                            403 if resource.provider == "gemini"
                                && response.text().await.is_ok_and(|body| {
                                    body.to_ascii_lowercase().contains("not found")
                                }) =>
                            {
                                CleanupOutcome::Gone
                            }
                            _ => CleanupOutcome::Failed,
                        };
                        (outcome, Some(status))
                    }
                    Err(_) => (CleanupOutcome::Failed, None),
                }
            }
        };
        match outcome {
            CleanupOutcome::Deleted => report.deleted += 1,
            CleanupOutcome::Gone => report.gone += 1,
            CleanupOutcome::Failed => report.remaining += 1,
            CleanupOutcome::NoCredential => report.no_credential += 1,
        }
        append(
            path,
            &[LedgerEntry::Cleanup {
                delete_url: resource.delete_url,
                outcome,
                status,
            }],
        );
    }
    report
}

#[cfg(test)]
mod tests;
