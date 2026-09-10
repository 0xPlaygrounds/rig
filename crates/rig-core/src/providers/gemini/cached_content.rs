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
//! [`super::completion::gemini_api_types::GenerateContentRequest::with_cached_content`] — so the
//! failure names the conflict instead of surfacing a provider 400.
//!
//! ## What that means for an `Agent`
//!
//! An agent mostly does not choose which of the three it sends — it sends what
//! it holds. A preamble becomes `systemInstruction`; every always-exposed tool
//! is advertised on every turn (one registered through `retrieved_tools` is
//! advertised on the turns retrieval selects it, so such an agent is refused
//! intermittently rather than not at all); and a configured tool choice becomes
//! `toolConfig` whether or not the agent has any tools.
//!
//! The one lever that does exist is a per-turn `RequestPatch::active_tools`
//! allow-list: an empty one empties the tool snapshot, so a tool-holding agent
//! builds a request with no `tools` and the handle is accepted. That is a
//! supported configuration, not a loophole — but it buys only the *request*,
//! never the dispatch. The tools it suppressed are still the agent's, and the
//! ones in the cache are still unreachable, so an agent that has to empty its
//! allow-list to use a cache is an agent whose tools do nothing on that turn.
//!
//! The agent derives the declarations it sends and the handles it dispatches
//! through from a single registry snapshot, so it can only ever dispatch a tool
//! it advertised — a call to a tool it never advertised is an invalid tool call,
//! not a dispatch. The converse is representable and rig uses it:
//! `OutputMode::Tool` advertises a synthetic output tool that is deliberately
//! not executable. But that only ever adds declarations, never dispatch reach,
//! which is why a tool set that lives in the cache stays out of an agent's
//! hands.
//!
//! Leaving the allow-list aside, then, an agent reads from a cache when it has
//! no preamble, no tools and no tool choice. Native structured output is fine — the schema
//! rides in `generationConfig`, and the default `OutputMode::Auto` resolves
//! there for a tool-less agent. `OutputMode::Tool` is not, because it advertises
//! that synthetic tool *and* extends the preamble; `Extractor` pins that mode,
//! so extractors cannot use a cache. `OutputMode::Prompted` is not either: it
//! writes the schema into the preamble. Context documents are fine: they are
//! appended to the chat history as user content.
//!
//! A cache carrying *function declarations* or `toolConfig` is consequently for
//! the caller who drives [`super::completion::CompletionModel`] directly and
//! runs the tool loop themselves. A provider-hosted tool is the exception:
//! `codeExecution` runs on Gemini's side and needs no loop, so a cache carrying
//! one is usable from an agent that declares nothing itself.
//!
//! # Example
//!
//! ```ignore
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! use rig_core::client::CompletionClient;
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
use crate::providers::internal::model_listing::MAX_LISTING_PAGES;
use crate::wasm_compat::{WasmCompatSend, WasmCompatSync};

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
    #[error("gemini cached content `{name}` is expired or was deleted: {message}")]
    Expired { name: String, message: String },

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
///
/// Not `Clone`, deliberately. An earlier revision held `serde_json::Value` for
/// `tools`/`tool_config` specifically to stay cloneable, on the theory that a
/// "one cache, many callers" shape would want it. It does not: what gets shared
/// is the [`CachedContent`] *handle* the create returns, which is cheap and
/// `Clone`, not the request that made it. Holding the typed values instead buys
/// a builder whose methods all return `Self` rather than two of nine returning
/// `Result` for an implementation detail.
///
/// Every field is private and reachable only through the builder. That is what
/// makes [`CacheExpiry`]'s guarantee real: with public `ttl` and `expire_time`,
/// `NewCachedContent { ttl: Some(..), expire_time: Some(..), ..Default::default() }`
/// compiles and the API rejects it — exactly the state the enum exists to make
/// unrepresentable. Keeping them private also avoids freezing untyped JSON into
/// the public API for `tools`/`tool_config`.
#[derive(Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NewCachedContent {
    /// Fully qualified model name (`models/gemini-2.5-flash`). A request that
    /// uses the cache must name the same model.
    model: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    contents: Vec<Content>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system_instruction: Option<Content>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_config: Option<ToolConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    display_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    ttl: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    expire_time: Option<String>,
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
    /// own (Gemini rejects that, and so does rig — see
    /// [`super::completion::gemini_api_types::GenerateContentRequest::with_cached_content`]).
    ///
    /// Function declarations here are *declarations*, not implementations, which
    /// is what puts them out of reach of rig's `Agent` — see the module docs
    /// above for why. A cached function tool set is usable only when you drive
    /// [`super::completion::CompletionModel`] yourself and run the tool loop by
    /// hand: read the `functionCall` parts off the response and append the
    /// matching `functionResponse` parts to the next request. A provider-hosted
    /// tool such as `codeExecution` is different — Gemini runs it, so a cache
    /// carrying one needs no loop and works from an agent.
    pub fn tools(mut self, tools: Vec<Tool>) -> Self {
        self.tools = Some(tools);
        self
    }

    /// Attach the tool choice this cache owns.
    ///
    /// Same reachability caveat as [`Self::tools`]: a request carrying its own
    /// tool choice alongside the handle is refused, and rig's `Agent` sends one
    /// whenever it is configured with one — even a tool-less agent — so this is
    /// for callers driving [`super::completion::CompletionModel`] directly. A
    /// tool-less agent does at least lose nothing by dropping its tool choice,
    /// which is not true of a tool set. Gemini accepts a
    /// `toolConfig` with no `tools` (measured; see the create matrix), which is
    /// why the two are separate builders rather than one.
    pub fn tool_config(mut self, tool_config: ToolConfig) -> Self {
        self.tool_config = Some(tool_config);
        self
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
pub struct CachedContentClient<H = crate::http_client::BoxedHttpClient> {
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
        let http = self.client.get(resource_path(name)?)?.body(Vec::new())?;
        self.send(http, Some(name)).await
    }

    /// Every cached content this API key can see, following pagination.
    pub async fn list(&self) -> Result<Vec<CachedContent>, CachedContentError> {
        self.list_with_page_size(MAX_PAGE_SIZE).await
    }

    /// [`Self::list`] with an explicit page size.
    ///
    /// Exists for two reasons. A caller holding thousands of caches may want
    /// smaller responses, and — less obviously but more importantly — the
    /// cursor-following loop below is otherwise unreachable in a test: Gemini
    /// returns up to 1,000 entries per page, so proving the loop works would
    /// mean creating a thousand billed caches. With a page size of 1 and three
    /// caches it is three pages and the loop is exercised for real.
    pub async fn list_with_page_size(
        &self,
        page_size: usize,
    ) -> Result<Vec<CachedContent>, CachedContentError> {
        let mut all = Vec::new();
        let mut page_token: Option<String> = None;
        // Only the loop running out of iterations is a ceiling. Every `break`
        // below is Gemini ending the listing, which is the normal path and must
        // stay silent — inferring the ceiling from `page_token` instead would
        // report one on any listing that fetched more than a single page, since
        // the cursor of the *previous* page is still held when the loop breaks.
        let mut exhausted_page_budget = true;

        for _ in 0..MAX_LISTING_PAGES {
            // Percent-encoded through the same helper `list_models_path` uses
            // (`internal::model_listing::with_query_pairs`), which has a test
            // pinning `pageToken=weird+token%26x%3D1`. Concatenating the cursor
            // raw would let a `+`, `&`, `=` or `/` in it truncate the cursor or
            // inject a query parameter, silently dropping pages.
            let page_size = page_size.to_string();
            let mut pairs: Vec<(&str, &str)> = vec![("pageSize", page_size.as_str())];
            if let Some(token) = &page_token {
                pairs.push(("pageToken", token.as_str()));
            }
            let path = crate::providers::internal::model_listing::with_query_pairs(
                CACHED_CONTENTS_PATH,
                &pairs,
            );
            let http = self.client.get(&path)?.body(Vec::new())?;
            let page: ListCachedContentsResponse = self.send_json(http, None).await?;
            all.extend(page.cached_contents);

            // An empty cursor counts as absent, matching how every other
            // provider-reported cursor in rig is read: re-sending an empty
            // `pageToken` returns the same page forever.
            let Some(token) = page.next_page_token.filter(|token| !token.is_empty()) else {
                exhausted_page_budget = false;
                break;
            };
            // A cursor that does not advance is a server bug: the next request
            // would be byte-identical to the one just answered, so the same
            // page would come back forever.
            if page_token.as_deref() == Some(token.as_str()) {
                tracing::warn!(
                    provider = "Gemini",
                    cached_contents = all.len(),
                    "cachedContents listing repeated its pagination cursor; returning the \
                     pages fetched so far"
                );
                exhausted_page_budget = false;
                break;
            }
            page_token = Some(token);
        }

        if exhausted_page_budget {
            tracing::warn!(
                provider = "Gemini",
                cached_contents = all.len(),
                pages = MAX_LISTING_PAGES,
                "cachedContents listing hit its page ceiling with a cursor still advancing; \
                 returning the pages fetched so far"
            );
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

        // The `?` below is only ours because `resource_path` refuses an id that
        // carries one: an unvalidated handle would put `updateMask` inside the
        // caller's query string on a resource we did not mean to patch.
        let path = format!("{}?updateMask={mask}", resource_path(name)?);
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
    ///
    /// A handle that is not a plain `cachedContents/<id>` (or a bare `<id>`) is
    /// refused with [`CachedContentError::Invalid`] before anything is sent —
    /// spliced into the path, a `?` or `#` would aim this delete at a different
    /// cache and succeed.
    pub async fn delete(&self, name: &str) -> Result<(), CachedContentError> {
        let http = self.client.delete(resource_path(name)?)?.body(Vec::new())?;
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
            // A transport is free to hand the non-success status back as an
            // `Ok` response rather than an error, and rig's own test double
            // does exactly that. Without this arm the *error* body fell through
            // to the `serde_json::from_str` below and surfaced as "missing
            // field `name`" — a deserialization failure standing in for a 404,
            // with `Expired` unreachable. Every other status triage in the
            // crate checks this on the `Ok` path too (`client::Client::verify`,
            // `internal::model_listing::decode_json_response`).
            Ok(response) if !response.status().is_success() => {
                let status = response.status().as_u16();
                // A failed body read must not cancel the triage. The status is
                // already in hand, and the `Err` arm below classifies even when
                // the error carries no body at all — dropping to `Http` here
                // would throw away the one thing that says the handle is gone.
                let message = http_client::text(response)
                    .await
                    .unwrap_or_else(|error| format!("failed to read error response body: {error}"));
                return Err(classify_failure(status, message, name));
            }
            Ok(response) => http_client::text(response)
                .await
                .map_err(CachedContentError::Http)?,
            // Triage on the *status*, not on one error variant. The bundled
            // reqwest transports always report a non-success status as
            // `InvalidStatusCodeWithDetails`, but `H` is a public extension
            // point (`ClientBuilder::http_client`) and a custom `HttpClientExt`
            // may report `InvalidStatusCode` or `InvalidStatusCodeWithMessage`
            // instead. Matching the one variant dropped those into `Http`
            // below, so the recovery this module documents — recreate the cache
            // on `Expired` — never fired outside the bundled clients.
            Err(error) => {
                let Some(status) = error.non_success_status() else {
                    // No status at all: a genuine transport failure (DNS, TLS,
                    // a dropped connection), which recreating a cache does not
                    // answer.
                    return Err(CachedContentError::Http(error));
                };
                let message = error.non_success_body().unwrap_or_default().to_owned();
                return Err(classify_failure(status.as_u16(), message, name));
            }
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

/// Stand-in for the provider's message when a failure carried no text.
///
/// [`http_client::Error::InvalidStatusCode`] carries a status and nothing else,
/// and a non-success response can have an empty body; either way there is
/// nothing to quote. An empty `message` would leave both [`CachedContentError`]
/// Displays ending in a bare colon, which reads as a truncated error rather
/// than as a silent provider.
const NO_RESPONSE_BODY: &str = "no response body";

/// Turn a non-success status and its body into the error a caller matches on.
///
/// Shared by both failure paths in [`CachedContentClient::send_json`] — the
/// transport that reports the status as an error and the one that hands back
/// the non-success response — so the two cannot drift apart.
///
/// `name` is `Some` only for calls that address an existing handle. `create`
/// passes `None` deliberately: a 403 there is a disabled key, a project without
/// the API enabled, or quota denial, and reporting it as `Expired` for a cache
/// that was never made would send a caller into a recreate loop.
fn classify_failure(status: u16, message: String, name: Option<&str>) -> CachedContentError {
    let message = if message.trim().is_empty() {
        NO_RESPONSE_BODY.to_owned()
    } else {
        message
    };

    // 403 and 404 both mean "this handle is gone" depending on how long ago it
    // lapsed; collapsing them spares callers from matching on a status code to
    // answer one question.
    if matches!(status, 403 | 404)
        && let Some(name) = name
    {
        // Carry the provider's own message. A 403 also covers a disabled key, a
        // project without the API enabled, and quota denial — collapsing those
        // into "expired" without the message would throw away the only text
        // that says which.
        return CachedContentError::Expired {
            name: name.to_owned(),
            message,
        };
    }

    CachedContentError::Api { status, message }
}

/// The characters a Gemini `cachedContents` id is made of.
///
/// The ids Gemini hands back are twelve lowercase alphanumerics
/// (`cachedContents/n3v1qk0nqz9k`). `-` and `_` are admitted on top of that
/// because the cassette scrubber rewrites every recorded id to
/// `cached-REDACTED_1` (`tests/common/cassettes.rs`), and a replayed test
/// hands that placeholder straight back to `delete`. `.` is deliberately left
/// out: no observed id carries one, and a `..` segment is path traversal.
fn is_cache_id_char(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_')
}

/// `/v1beta/cachedContents/<id>` from either a bare id or a full handle.
///
/// Validates rather than interpolating, because this is the path `get`,
/// `update_expiry` and — the one that matters — `delete` send. A handle
/// carrying a `?` does not produce a malformed URL the provider rejects:
/// `Gemini::build_uri` switches its key separator to `&` the moment it sees
/// a `?` in the path, so `delete("abc?stale")` would issue a perfectly
/// well-formed `DELETE /v1beta/cachedContents/abc?stale&key=…` and destroy the
/// cache named `abc`. A `#` truncates the path the same way, a `/` retargets it
/// at another resource, and an empty id aims the request at the *collection*.
///
/// Refusing beats percent-encoding here. The id is server-assigned and opaque,
/// so a caller holding one that needs escaping is holding a bug; and encoding
/// would have to escape the id while leaving the optional `cachedContents/`
/// prefix intact — two rules for one string, in service of quietly rewriting
/// input that is always wrong.
///
/// The prefix stays optional here, unlike
/// [`super::completion::gemini_api_types::GenerateContentRequest::with_cached_content`],
/// which requires it. That is not an inconsistency: there the handle is a wire
/// value the API compares verbatim, here it is a path segment this function
/// writes itself.
fn resource_path(name: &str) -> Result<String, CachedContentError> {
    let id = name.strip_prefix("cachedContents/").unwrap_or(name);
    if id.is_empty() || !id.chars().all(is_cache_id_char) {
        return Err(CachedContentError::Invalid(format!(
            "`{name}` is not a cached content handle; expected `cachedContents/<id>` or a bare \
             `<id>` of letters, digits, `-` and `_`. The id is spliced into the request path, \
             where a `?`, `#` or `/` silently retargets the call at a different resource — and \
             this is the path that deletes"
        )));
    }
    Ok(format!("{CACHED_CONTENTS_PATH}/{id}"))
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod exhaustive_validation_tests;

#[cfg(test)]
mod status_triage_tests;
