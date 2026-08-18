//! Gemini explicit context caching — the `cachedContents` resource.
//!
//! Gemini has two caching features and they are not interchangeable.
//!
//! **Implicit caching** is automatic and best-effort: send a long prefix twice
//! and the provider *may* serve the second one from cache. There is no API
//! surface and no guarantee, and the warm-up is real — measured on
//! `gemini-2.5-flash` against an 18.5k-token corpus, five consecutive turns
//! reusing that corpus read **zero** cached tokens, and only a sixth request saw
//! 99.6%. Those five turns were billed at full price for ~92k prompt tokens.
//!
//! **Explicit caching** — this module — uploads the content once, gives you a
//! handle, and bills the handle for storage. The same corpus, measured the same
//! day through this API:
//!
//! | turn | prompt | cached | ratio |
//! |---|---:|---:|---:|
//! | 1 | 36,978 | 36,970 | **100.0%** |
//! | 5 | 37,026 | 36,970 | 99.8% |
//! | fresh conversation | 36,976 | 36,970 | **100.0%** |
//!
//! It hits on the *first* request, and it keeps hitting across conversations
//! that share nothing but the handle — which is the thing implicit caching
//! structurally cannot do, because implicit keys on a prefix that a new
//! conversation does not have yet.
//!
//! # When it pays
//!
//! Explicit caching bills storage per token-hour on top of the (reduced) cached
//! input rate, so an idle cache is not free. It pays when one large fixed
//! payload — a document corpus, a long system prompt, a video transcript — is
//! reused across enough calls to beat the storage cost, and it pays immediately
//! rather than after a warm-up. For a single short conversation, implicit
//! caching costs nothing and is the better default.
//!
//! # Constraints the API imposes
//!
//! A cached content owns the `systemInstruction`, `tools` and `toolConfig` for
//! every request that uses it. Sending any of them *alongside* `cachedContent`
//! is rejected:
//!
//! ```text
//! CachedContent can not be used with GenerateContent request setting
//! system_instruction, tools or tool_config.
//! ```
//!
//! rig checks that before the request leaves the process — see
//! [`super::completion::GenerateContentRequest::with_cached_content`] — so the
//! failure names the conflict instead of surfacing a provider 400.
//!
//! # Example
//!
//! ```no_run
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! use rig_core::providers::gemini;
//! use rig_core::providers::gemini::cached_content::{CacheExpiry, NewCachedContent};
//! use std::time::Duration;
//!
//! let client = gemini::Client::new("YOUR_API_KEY")?;
//! let caches = client.cached_contents();
//!
//! let cache = caches
//!     .create(
//!         NewCachedContent::new(gemini::completion::GEMINI_2_5_FLASH)
//!             .system_instruction("You answer questions about the attached corpus.")
//!             .content(std::fs::read_to_string("corpus.txt")?)
//!             .expiry(CacheExpiry::ttl(Duration::from_secs(600)))
//!             .display_name("corpus-v1"),
//!     )
//!     .await?;
//!
//! let model = client
//!     .completion_model(gemini::completion::GEMINI_2_5_FLASH)
//!     .with_cached_content(cache.name.clone());
//!
//! // ... use `model` normally; every request reads the cache ...
//!
//! caches.delete(&cache.name).await?; // storage bills until you do this
//! # Ok(())
//! # }
//! ```

use std::time::Duration;

use serde::{Deserialize, Serialize};

use super::client::Client;
use super::completion::gemini_api_types::{Content, Part, Role, Tool, ToolConfig};
use crate::http_client::{self, HttpClientExt};
use crate::wasm_compat::{WasmCompatSend, WasmCompatSync};
use serde_json::Value;

/// The `cachedContents` collection path.
const CACHED_CONTENTS_PATH: &str = "/v1beta/cachedContents";

/// Gemini caps a page of `cachedContents` at 1000.
const MAX_PAGE_SIZE: usize = 1000;

/// Something went wrong talking to the `cachedContents` API.
#[derive(Debug, thiserror::Error)]
pub enum CachedContentError {
    /// The cache handle no longer exists — almost always because its TTL
    /// elapsed.
    ///
    /// Separated from the other failures because it is the one a caller is
    /// expected to *handle* rather than propagate: a cache that expired mid-run
    /// is recreated, not reported. Gemini answers an expired handle with 403 or
    /// 404 depending on how long ago it lapsed, which is why matching on a
    /// status code is not something callers should have to do.
    #[error("gemini cached content `{name}` is expired or was deleted")]
    Expired { name: String },

    /// The API rejected the request.
    #[error("gemini cached content request failed with status {status}: {message}")]
    Api { status: u16, message: String },

    /// A caller-side mistake caught before the request went out.
    #[error("invalid gemini cached content request: {0}")]
    Invalid(String),

    #[error("http error: {0}")]
    Http(#[from] http_client::Error),

    #[error("could not build the request: {0}")]
    Request(#[from] http::Error),

    #[error("serialization error: {0}")]
    Serde(#[from] serde_json::Error),
}

/// How a cached content expires.
///
/// An enum rather than two `Option` fields because Gemini accepts exactly one of
/// `ttl` and `expireTime` and rejects a body carrying both. Making that
/// unrepresentable is cheaper than validating it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CacheExpiry {
    /// Expire this long after creation. Serialized as Gemini's duration string
    /// (`"600s"`).
    Ttl(Duration),
    /// Expire at an absolute RFC 3339 timestamp.
    ExpireTime(String),
}

impl CacheExpiry {
    pub fn ttl(ttl: Duration) -> Self {
        Self::Ttl(ttl)
    }

    pub fn expire_time(timestamp: impl Into<String>) -> Self {
        Self::ExpireTime(timestamp.into())
    }

    /// Gemini's duration encoding: fractional seconds with an `s` suffix.
    fn ttl_string(ttl: Duration) -> String {
        format!("{}.{:09}s", ttl.as_secs(), ttl.subsec_nanos())
    }
}

/// A cached content to create.
#[derive(Clone, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NewCachedContent {
    /// Fully qualified model name (`models/gemini-2.5-flash`). A request that
    /// uses the cache must name the same model.
    pub model: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub contents: Vec<Content>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_instruction: Option<Content>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Serialized `Tool` values, matching how
    /// [`super::completion::gemini_api_types::GenerateContentRequest`] carries
    /// them — `Tool` is not `Clone`, and a builder that cannot be cloned is
    /// awkward for the "one cache, many callers" shape this exists to serve.
    pub tools: Option<Vec<Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Serialized `ToolConfig`, carried as a value for the same reason as
    /// [`Self::tools`].
    pub tool_config: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub display_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttl: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expire_time: Option<String>,
}

impl NewCachedContent {
    /// Start a cached content for `model`.
    ///
    /// Accepts either the bare id (`gemini-2.5-flash`) or the qualified name
    /// (`models/gemini-2.5-flash`) and normalizes to the latter, which is what
    /// the API returns and what a `generateContent` request must match.
    pub fn new(model: impl AsRef<str>) -> Self {
        Self {
            model: qualify_model(model.as_ref()),
            ..Default::default()
        }
    }

    /// Append a user-role text content block.
    pub fn content(mut self, text: impl Into<String>) -> Self {
        self.contents.push(Content {
            parts: vec![Part::from(text.into())],
            role: Some(Role::User),
        });
        self
    }

    /// Append an already-built content block (multimodal payloads).
    pub fn content_block(mut self, content: Content) -> Self {
        self.contents.push(content);
        self
    }

    pub fn system_instruction(mut self, text: impl Into<String>) -> Self {
        self.system_instruction = Some(Content {
            parts: vec![Part::from(text.into())],
            role: Some(Role::Model),
        });
        self
    }

    /// Attach the tool set this cache owns.
    ///
    /// Every request using the handle inherits these; a request may not send its
    /// own (Gemini rejects that, and so does rig).
    pub fn tools(mut self, tools: Vec<Tool>) -> Result<Self, CachedContentError> {
        self.tools = Some(
            tools
                .into_iter()
                .map(serde_json::to_value)
                .collect::<Result<Vec<_>, _>>()?,
        );
        Ok(self)
    }

    /// Attach the tool choice this cache owns.
    pub fn tool_config(mut self, tool_config: ToolConfig) -> Result<Self, CachedContentError> {
        self.tool_config = Some(serde_json::to_value(tool_config)?);
        Ok(self)
    }

    pub fn display_name(mut self, name: impl Into<String>) -> Self {
        self.display_name = Some(name.into());
        self
    }

    /// Set the expiry. Setting it twice replaces the previous value rather than
    /// sending both, which the API rejects.
    pub fn expiry(mut self, expiry: CacheExpiry) -> Self {
        match expiry {
            CacheExpiry::Ttl(ttl) => {
                self.ttl = Some(CacheExpiry::ttl_string(ttl));
                self.expire_time = None;
            }
            CacheExpiry::ExpireTime(at) => {
                self.expire_time = Some(at);
                self.ttl = None;
            }
        }
        self
    }

    fn validate(&self) -> Result<(), CachedContentError> {
        if self.contents.is_empty() && self.system_instruction.is_none() {
            return Err(CachedContentError::Invalid(
                "a cached content needs contents or a system instruction; an empty cache would \
                 bill for storage and cache nothing"
                    .to_owned(),
            ));
        }
        Ok(())
    }
}

/// Storage accounting Gemini reports for a cached content.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CachedContentUsage {
    /// Tokens held by this cache. This is what storage is billed on, and it is
    /// also the ceiling on what a request against the handle can read back.
    #[serde(default)]
    pub total_token_count: u64,
}

/// A cached content resource as Gemini reports it.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CachedContent {
    /// Server-assigned handle, `cachedContents/<id>`. This is what
    /// [`super::completion::CompletionModel::with_cached_content`] takes.
    pub name: String,
    /// Qualified model this cache is bound to.
    #[serde(default)]
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub display_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub create_time: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub update_time: Option<String>,
    /// When this cache lapses. After it does, using the handle fails with
    /// [`CachedContentError::Expired`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expire_time: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage_metadata: Option<CachedContentUsage>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ListCachedContentsResponse {
    #[serde(default)]
    cached_contents: Vec<CachedContent>,
    #[serde(default)]
    next_page_token: Option<String>,
}

/// Client for Gemini's `cachedContents` resource.
///
/// Obtained from [`Client::cached_contents`].
#[derive(Clone, Debug)]
pub struct CachedContentClient<H = reqwest::Client> {
    client: Client<H>,
}

impl<H> CachedContentClient<H> {
    pub(crate) fn new(client: Client<H>) -> Self {
        Self { client }
    }
}

impl<H> CachedContentClient<H>
where
    H: HttpClientExt + Clone + WasmCompatSend + WasmCompatSync + 'static,
{
    /// Upload content and get a handle back.
    ///
    /// The returned [`CachedContent::usage_metadata`] reports how many tokens
    /// are now being stored — and therefore billed — so log it if cost matters.
    pub async fn create(
        &self,
        request: NewCachedContent,
    ) -> Result<CachedContent, CachedContentError> {
        request.validate()?;
        let body = serde_json::to_vec(&request)?;
        let http = self.client.post(CACHED_CONTENTS_PATH)?.body(body)?;
        self.send(http, None).await
    }

    /// Fetch one cached content by handle.
    pub async fn get(&self, name: &str) -> Result<CachedContent, CachedContentError> {
        let http = self.client.get(resource_path(name))?.body(Vec::new())?;
        self.send(http, Some(name)).await
    }

    /// Every cached content this API key can see, following pagination.
    pub async fn list(&self) -> Result<Vec<CachedContent>, CachedContentError> {
        let mut all = Vec::new();
        let mut page_token: Option<String> = None;

        loop {
            let mut path = format!("{CACHED_CONTENTS_PATH}?pageSize={MAX_PAGE_SIZE}");
            if let Some(token) = &page_token {
                path.push_str("&pageToken=");
                path.push_str(token);
            }
            let http = self.client.get(&path)?.body(Vec::new())?;
            let page: ListCachedContentsResponse = self.send_json(http, None).await?;
            all.extend(page.cached_contents);

            // An empty cursor counts as absent, matching how every other
            // provider-reported cursor in rig is read: re-sending an empty
            // `pageToken` returns the same page forever.
            let next = page.next_page_token.filter(|token| !token.is_empty());
            match next {
                // A cursor that does not advance is a server bug that would
                // otherwise spin here until the process dies.
                Some(token) if Some(&token) != page_token.as_ref() => page_token = Some(token),
                _ => break,
            }
        }

        Ok(all)
    }

    /// Extend (or shorten) a cache's life.
    ///
    /// Expiry is the only mutable part of the resource — the content itself is
    /// immutable, so refreshing a corpus means creating a new cache and deleting
    /// the old one.
    pub async fn update_expiry(
        &self,
        name: &str,
        expiry: CacheExpiry,
    ) -> Result<CachedContent, CachedContentError> {
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct Patch {
            #[serde(skip_serializing_if = "Option::is_none")]
            ttl: Option<String>,
            #[serde(skip_serializing_if = "Option::is_none")]
            expire_time: Option<String>,
        }

        let (patch, mask) = match expiry {
            CacheExpiry::Ttl(ttl) => (
                Patch {
                    ttl: Some(CacheExpiry::ttl_string(ttl)),
                    expire_time: None,
                },
                "ttl",
            ),
            CacheExpiry::ExpireTime(at) => (
                Patch {
                    ttl: None,
                    expire_time: Some(at),
                },
                "expireTime",
            ),
        };

        let path = format!("{}?updateMask={mask}", resource_path(name));
        let http = self
            .client
            .patch(&path)?
            .body(serde_json::to_vec(&patch)?)?;
        self.send(http, Some(name)).await
    }

    /// Delete a cached content.
    ///
    /// Storage bills until this is called, so a cache created for the duration
    /// of a task should be deleted on the failure path too.
    pub async fn delete(&self, name: &str) -> Result<(), CachedContentError> {
        let http = self.client.delete(resource_path(name))?.body(Vec::new())?;
        let _: serde_json::Value = self.send_json(http, Some(name)).await?;
        Ok(())
    }

    async fn send(
        &self,
        request: http_client::Request<Vec<u8>>,
        name: Option<&str>,
    ) -> Result<CachedContent, CachedContentError> {
        self.send_json(request, name).await
    }

    async fn send_json<T>(
        &self,
        request: http_client::Request<Vec<u8>>,
        name: Option<&str>,
    ) -> Result<T, CachedContentError>
    where
        T: serde::de::DeserializeOwned,
    {
        let response = HttpClientExt::send::<_, Vec<u8>>(&self.client, request).await;

        let bytes = match response {
            Ok(response) => http_client::text(response)
                .await
                .map_err(CachedContentError::Http)?,
            Err(http_client::Error::InvalidStatusCodeWithDetails { status, body, .. }) => {
                let message = body.to_string();
                // 403 and 404 both mean "this handle is gone" depending on how
                // long ago it lapsed; collapsing them spares callers from
                // matching on a status code to answer one question.
                if matches!(status.as_u16(), 403 | 404)
                    && let Some(name) = name
                {
                    return Err(CachedContentError::Expired {
                        name: name.to_owned(),
                    });
                }
                return Err(CachedContentError::Api {
                    status: status.as_u16(),
                    message,
                });
            }
            Err(error) => return Err(CachedContentError::Http(error)),
        };

        // DELETE answers `{}`; `serde_json::Value` absorbs that, and a typed
        // caller never asks for one.
        if bytes.trim().is_empty() {
            return Ok(serde_json::from_str("null")?);
        }
        Ok(serde_json::from_str(&bytes)?)
    }
}

/// `models/x` from `x`, idempotently.
fn qualify_model(model: &str) -> String {
    if model.starts_with("models/") {
        model.to_owned()
    } else {
        format!("models/{model}")
    }
}

/// `/v1beta/cachedContents/<id>` from either a bare id or a full handle.
fn resource_path(name: &str) -> String {
    let id = name.strip_prefix("cachedContents/").unwrap_or(name);
    format!("{CACHED_CONTENTS_PATH}/{id}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_is_qualified_idempotently() {
        assert_eq!(qualify_model("gemini-2.5-flash"), "models/gemini-2.5-flash");
        assert_eq!(
            qualify_model("models/gemini-2.5-flash"),
            "models/gemini-2.5-flash"
        );
    }

    #[test]
    fn resource_path_accepts_a_bare_id_or_a_full_handle() {
        assert_eq!(resource_path("abc123"), "/v1beta/cachedContents/abc123");
        assert_eq!(
            resource_path("cachedContents/abc123"),
            "/v1beta/cachedContents/abc123"
        );
    }

    #[test]
    fn ttl_serializes_in_geminis_duration_form() {
        assert_eq!(
            CacheExpiry::ttl_string(Duration::from_secs(600)),
            "600.000000000s"
        );
    }

    /// Expiry is one field on the wire, never two — Gemini rejects a body that
    /// carries both, so the builder must replace rather than accumulate.
    #[test]
    fn setting_expiry_twice_replaces_rather_than_sending_both() {
        let request = NewCachedContent::new("gemini-2.5-flash")
            .content("corpus")
            .expiry(CacheExpiry::ttl(Duration::from_secs(60)))
            .expiry(CacheExpiry::expire_time("2030-01-01T00:00:00Z"));
        assert!(request.ttl.is_none());
        assert_eq!(request.expire_time.as_deref(), Some("2030-01-01T00:00:00Z"));

        let request = request.expiry(CacheExpiry::ttl(Duration::from_secs(60)));
        assert!(request.expire_time.is_none());
        assert!(request.ttl.is_some());
    }

    #[test]
    fn an_empty_cache_is_rejected_before_it_bills_for_storage() {
        let error = NewCachedContent::new("gemini-2.5-flash")
            .display_name("empty")
            .validate()
            .expect_err("an empty cached content should be refused");
        assert!(matches!(error, CachedContentError::Invalid(_)), "{error:?}");
    }

    #[test]
    fn create_body_omits_unset_fields() {
        let body = serde_json::to_value(
            NewCachedContent::new("gemini-2.5-flash")
                .content("corpus")
                .expiry(CacheExpiry::ttl(Duration::from_secs(600))),
        )
        .expect("serialize");
        let object = body.as_object().expect("object");
        assert!(!object.contains_key("expireTime"));
        assert!(!object.contains_key("tools"));
        assert!(!object.contains_key("systemInstruction"));
        assert_eq!(
            object.get("model").and_then(|m| m.as_str()),
            Some("models/gemini-2.5-flash")
        );
    }
}
