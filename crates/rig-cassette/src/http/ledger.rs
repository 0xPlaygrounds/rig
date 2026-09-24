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
    if !method.eq_ignore_ascii_case("POST") {
        return Vec::new();
    }
    let path = path.split('?').next().unwrap_or(path).trim_end_matches('/');
    let Some(kind) = resource_kind(provider, path) else {
        return Vec::new();
    };
    let stores = serde_json::from_slice::<Value>(request_body)
        .ok()
        .and_then(|request| request.get("store").and_then(Value::as_bool))
        .unwrap_or(true);
    if matches!(kind, ResourceKind::Response | ResourceKind::Interaction) && !stores {
        return Vec::new();
    }
    let mut ids = Vec::new();
    for document in response_documents(response_body) {
        if let Some(id) = created_id(kind, &document)
            && !ids.contains(&id)
        {
            ids.push(id);
        }
    }
    ids.into_iter()
        .map(|id| CreatedResource {
            provider: provider.to_owned(),
            scenario: String::new(),
            kind,
            delete_url: delete_url(kind, origin, path, &id),
            id,
        })
        .collect()
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

/// The JSON documents of a reply body: the body itself, or each SSE event's
/// data.
pub(crate) fn response_documents(body: &[u8]) -> Vec<Value> {
    let text = String::from_utf8_lossy(body);
    let trimmed = text.trim_start();
    if trimmed.starts_with('{') || trimmed.starts_with('[') {
        return serde_json::from_str::<Value>(trimmed).into_iter().collect();
    }
    text.lines()
        .filter_map(|line| line.strip_prefix("data:"))
        .filter_map(|data| serde_json::from_str::<Value>(data.trim()).ok())
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
