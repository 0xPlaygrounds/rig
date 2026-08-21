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
//! use rig_core::client::{CompletionClient, ProviderClient};
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
pub struct CachedContentClient<H> {
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
/// `GeminiExt::build_uri` switches its key separator to `&` the moment it sees
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
mod tests {
    use super::*;
    use crate::test_utils::{MockHttpResponse, SequencedHttpClient};

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
        assert_eq!(
            resource_path("abc123").expect("a bare id is a handle"),
            "/v1beta/cachedContents/abc123"
        );
        assert_eq!(
            resource_path("cachedContents/abc123").expect("a full handle is a handle"),
            "/v1beta/cachedContents/abc123"
        );
    }

    /// The destructive path, end to end: a handle that would mis-target must
    /// not reach the socket at all.
    ///
    /// `resource_path`'s unit tests prove the string is refused; this proves the
    /// refusal happens *before* the request is built. It matters because the
    /// URL these handles produce is not malformed — `GeminiExt::build_uri`
    /// appends the API key with `&` once the path contains a `?`, so
    /// `DELETE /v1beta/cachedContents/abc?stale&key=…` is a well-formed request
    /// that deletes cache `abc` and returns 200.
    #[tokio::test]
    async fn a_mis_targeting_handle_never_reaches_the_socket() {
        for smuggled in ["abc?stale", "abc#frag", "abc/def", ""] {
            // No scripted responses: anything that does escape fails twice, once
            // on the error variant and once on the captured request.
            let http_client = SequencedHttpClient::default();
            let client = Client::builder()
                .api_key("test-key")
                .http_client(http_client.clone())
                .build()
                .expect("client should build");
            let caches = client.cached_contents();

            let outcomes = [
                ("get", caches.get(smuggled).await.err()),
                ("delete", caches.delete(smuggled).await.err()),
                (
                    "update_expiry",
                    caches
                        .update_expiry(smuggled, CacheExpiry::ttl(Duration::from_secs(60)))
                        .await
                        .err(),
                ),
            ];
            for (label, error) in outcomes {
                let error = error
                    .unwrap_or_else(|| panic!("{label} should refuse the handle {smuggled:?}"));
                assert!(
                    matches!(error, CachedContentError::Invalid(_)),
                    "{label} on {smuggled:?}: {error:?}"
                );
            }

            assert!(
                http_client.requests().is_empty(),
                "handle {smuggled:?} escaped the process: {:?}",
                http_client.requests()
            );
        }
    }

    /// The exact URI `update_expiry` builds, so the ordering of its three
    /// query-string writers is pinned in one place.
    ///
    /// `resource_path` writes the path, the `format!` appends `?updateMask=`,
    /// and `build_uri` follows with `&key=` because it now sees a `?`. That
    /// layout is only stable while a handle cannot carry its own `?` — which is
    /// what `resource_path` refuses, and what the cells above cover. This cell
    /// pins the well-formed side: it passed before the validation existed and
    /// exists to catch the mask being concatenated ahead of it, or the path
    /// being escaped. The recorded PATCH in
    /// `cached_content_matrix/edge_update_expiry_absolute` pins the same layout
    /// against the live API; this one names it locally.
    #[tokio::test]
    async fn update_expiry_puts_its_update_mask_after_the_validated_path() {
        let http_client = SequencedHttpClient::new([MockHttpResponse::success(
            serde_json::json!({
                "name": "cachedContents/n3v1qk0nqz9k",
                "model": "models/gemini-2.5-flash"
            })
            .to_string(),
        )]);
        let client = Client::builder()
            .api_key("test-key")
            .http_client(http_client.clone())
            .build()
            .expect("client should build");

        client
            .cached_contents()
            .update_expiry(
                "cachedContents/n3v1qk0nqz9k",
                CacheExpiry::ttl(Duration::from_secs(600)),
            )
            .await
            .expect("a well-formed handle should be patched");

        let requests = http_client.requests();
        let [request] = requests.as_slice() else {
            panic!("exactly one request should have been sent: {requests:?}");
        };
        assert!(
            request
                .uri
                .ends_with("/v1beta/cachedContents/n3v1qk0nqz9k?updateMask=ttl&key=test-key"),
            "{}",
            request.uri
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

    // The pagination loop, and every way its cursor can fail to advance:
    // absent, empty, repeated, and alternating — the last of which only the
    // page ceiling catches. `paginate_models` carries the same three rules for
    // model listings, but this resource cannot call it (it is typed on
    // `Model`/`ModelListingError` and fetches through `get_bytes`, which
    // collapses the 403/404 triage `CachedContentError::Expired` exists for),
    // so the rules are restated in `list_with_page_size` and pinned here.
    //
    // Only the malformed-cursor cells are unrecordable: no live response
    // carries an empty, repeated or alternating cursor, and no live cursor
    // carries URL-significant characters. Ordinary and multi-page listings are
    // recorded — `prompt_caching/explicit_cache_lifecycle` for a single page,
    // `cached_content_matrix/edge_list_pagination` for three pages at
    // `pageSize=1`. These cells exist to pin the three termination guards,
    // which a recording cannot exercise.

    /// One page of Gemini's `cachedContents` list envelope.
    fn cached_page(names: &[&str], next_page_token: Option<&str>) -> MockHttpResponse {
        let cached_contents: Vec<_> = names
            .iter()
            .map(|name| serde_json::json!({ "name": format!("cachedContents/{name}") }))
            .collect();
        MockHttpResponse::success(
            serde_json::json!({
                "cachedContents": cached_contents,
                "nextPageToken": next_page_token,
            })
            .to_string(),
        )
    }

    /// A `cachedContents` client whose transport answers the scripted pages in
    /// order and `NOT_IMPLEMENTED` once they run out — so a loop that fails to
    /// terminate ends its test with an error rather than hanging the suite.
    fn caches(
        pages: Vec<MockHttpResponse>,
    ) -> (
        CachedContentClient<SequencedHttpClient>,
        SequencedHttpClient,
    ) {
        let http_client = SequencedHttpClient::new(pages);
        let client = Client::builder()
            .api_key("test-key")
            .http_client(http_client.clone())
            .build()
            .expect("client should build");
        (client.cached_contents(), http_client)
    }

    /// The ordinary single-page listing — what `list()`'s default page size
    /// returns for any realistic collection — is unchanged by the termination
    /// guards. Recorded live in `prompt_caching/explicit_cache_lifecycle`;
    /// repeated here so the guards have a no-cursor baseline on the same mock
    /// transport as the cells below.
    #[tokio::test]
    async fn single_page_listing_is_unchanged() {
        let (caches, http_client) = caches(vec![cached_page(&["a", "b"], None)]);

        let listed = caches.list().await.expect("listing should succeed");

        let names: Vec<_> = listed.iter().map(|entry| entry.name.as_str()).collect();
        assert_eq!(names, ["cachedContents/a", "cachedContents/b"]);
        assert_eq!(http_client.remaining_responses(), 0);
    }

    /// An empty `nextPageToken` is as unusable as an absent one: re-sending an
    /// empty `pageToken` returns the same page forever.
    #[tokio::test]
    async fn pagination_stops_on_an_empty_cursor() {
        let (caches, http_client) = caches(vec![
            cached_page(&["a"], Some("")),
            cached_page(&["b"], None),
        ]);

        let listed = caches
            .list_with_page_size(1)
            .await
            .expect("listing should terminate");

        let names: Vec<_> = listed.iter().map(|entry| entry.name.as_str()).collect();
        assert_eq!(names, ["cachedContents/a"]);
        assert_eq!(http_client.remaining_responses(), 1);
    }

    /// A server that keeps echoing the same cursor cannot advance the listing
    /// either — the next request would be byte-identical to the one just
    /// answered, so the same page would come back forever.
    #[tokio::test]
    async fn pagination_stops_on_a_cursor_that_does_not_advance() {
        let (caches, http_client) = caches(vec![
            cached_page(&["a"], Some("stuck")),
            cached_page(&["b"], Some("stuck")),
            cached_page(&["c"], Some("stuck")),
        ]);

        let listed = caches
            .list_with_page_size(1)
            .await
            .expect("listing should terminate");

        let names: Vec<_> = listed.iter().map(|entry| entry.name.as_str()).collect();
        assert_eq!(
            names,
            ["cachedContents/a", "cachedContents/b"],
            "the repeat is only detectable on the second page, so both are kept",
        );
        assert_eq!(http_client.remaining_responses(), 1);
    }

    /// A cursor that keeps *changing* without making progress — a gateway
    /// alternating between two values, or minting a fresh one per request —
    /// defeats the repeat check, which only remembers the previous cursor. Only
    /// the page ceiling stops it, and without one `list` never returns while
    /// `all` grows without bound (rig#2334).
    #[tokio::test]
    async fn pagination_stops_at_the_page_ceiling_on_an_alternating_cursor() {
        // Two cursors that alternate forever: every request differs from the
        // one before, so no repeat is ever observed.
        let pages: Vec<_> = (0..MAX_LISTING_PAGES + 10)
            .map(|i| cached_page(&["a"], Some(if i % 2 == 0 { "ping" } else { "pong" })))
            .collect();
        let (caches, http_client) = caches(pages);

        let listed = caches
            .list_with_page_size(1)
            .await
            .expect("the ceiling ends the listing instead of looping");

        assert_eq!(
            listed.len(),
            MAX_LISTING_PAGES,
            "exactly the ceiling's worth of pages is fetched",
        );
        assert_eq!(
            http_client.remaining_responses(),
            10,
            "the loop stops at the ceiling rather than draining every page",
        );
    }

    /// A cursor carrying URL-significant characters is percent-encoded rather
    /// than interpolated, so it cannot truncate the path or inject a query
    /// parameter — Gemini appends `key=` to every URI, so a raw `&` in the
    /// cursor would sit next to the credential.
    #[tokio::test]
    async fn pagination_percent_encodes_the_cursor() {
        let (caches, http_client) = caches(vec![
            cached_page(&["a"], Some("weird token&x=1")),
            cached_page(&["b"], None),
        ]);

        caches
            .list_with_page_size(1)
            .await
            .expect("listing should succeed");

        let uris: Vec<_> = http_client
            .requests()
            .into_iter()
            .map(|request| request.uri)
            .collect();
        assert!(
            uris[1].contains("pageSize=1&pageToken=weird+token%26x%3D1&key="),
            "the cursor must be percent-encoded: {uris:?}",
        );
    }
}

#[cfg(test)]
mod exhaustive_validation_tests {
    //! The client-side validation surface, exhaustively.
    //!
    //! Every cell here is free: it never opens a socket, so the full Cartesian
    //! product is affordable where a recorded matrix would have to sample.

    use super::*;

    /// `models/x` from every spelling a caller might reach for.
    #[test]
    fn model_qualification_is_total_and_idempotent() {
        for (input, expected) in [
            ("gemini-2.5-flash", "models/gemini-2.5-flash"),
            ("models/gemini-2.5-flash", "models/gemini-2.5-flash"),
            ("", "models/"),
            ("models/", "models/"),
            ("tunedModels/x", "models/tunedModels/x"),
        ] {
            assert_eq!(qualify_model(input), expected, "input {input:?}");
        }
    }

    /// Every shape a handle can arrive in, against the path that `get`,
    /// `update_expiry` and `delete` all build from it.
    ///
    /// The refusals carry the weight. `abc?stale` is not a malformed URL the
    /// provider would reject — `build_uri` appends the API key with `&` once a
    /// `?` is present, so it is a valid `DELETE` of cache `abc`. `#` and `/`
    /// mis-target the same way, and an empty id aims the request at the
    /// collection endpoint.
    #[test]
    fn resource_path_accepts_server_assigned_ids_and_refuses_everything_else() {
        for (input, expected) in [
            // The shapes the API actually hands back, in both spellings, plus
            // the scrubbed spelling a replayed cassette feeds back to `delete`.
            ("n3v1qk0nqz9k", Some("/v1beta/cachedContents/n3v1qk0nqz9k")),
            (
                "cachedContents/n3v1qk0nqz9k",
                Some("/v1beta/cachedContents/n3v1qk0nqz9k"),
            ),
            (
                "cached-REDACTED_1",
                Some("/v1beta/cachedContents/cached-REDACTED_1"),
            ),
            ("abc?stale", None),
            ("abc#frag", None),
            ("abc/def", None),
            ("cachedContents/cachedContents/abc", None),
            ("abc%2Fdef", None),
            ("abc def", None),
            ("abc\n", None),
            ("..", None),
            ("", None),
            ("cachedContents/", None),
        ] {
            match (resource_path(input), expected) {
                (Ok(path), Some(expected)) => assert_eq!(path, expected, "input {input:?}"),
                (Err(CachedContentError::Invalid(message)), None) => assert!(
                    message.contains(input),
                    "input {input:?}: the refusal should quote the handle, got {message}"
                ),
                (outcome, _) => panic!("input {input:?}: unexpected {outcome:?}"),
            }
        }
    }

    /// Gemini's duration encoding across the range a caller might pass.
    #[test]
    fn ttl_encoding_covers_the_useful_range() {
        for (secs, nanos, expected) in [
            (0u64, 0u32, "0.000000000s"),
            (1, 0, "1.000000000s"),
            (60, 0, "60.000000000s"),
            (3_600, 0, "3600.000000000s"),
            (86_400, 0, "86400.000000000s"),
            (0, 500_000_000, "0.500000000s"),
        ] {
            assert_eq!(
                CacheExpiry::ttl_string(Duration::new(secs, nanos)),
                expected,
                "{secs}s {nanos}ns"
            );
        }
    }

    /// Expiry is one field on the wire, whichever order it is set in and
    /// however many times.
    #[test]
    fn expiry_is_exclusive_under_every_ordering() {
        let orderings: Vec<Vec<CacheExpiry>> = vec![
            vec![CacheExpiry::ttl(Duration::from_secs(60))],
            vec![CacheExpiry::expire_time("2030-01-01T00:00:00Z")],
            vec![
                CacheExpiry::ttl(Duration::from_secs(60)),
                CacheExpiry::expire_time("2030-01-01T00:00:00Z"),
            ],
            vec![
                CacheExpiry::expire_time("2030-01-01T00:00:00Z"),
                CacheExpiry::ttl(Duration::from_secs(60)),
            ],
            vec![
                CacheExpiry::ttl(Duration::from_secs(1)),
                CacheExpiry::ttl(Duration::from_secs(2)),
            ],
        ];

        for ordering in orderings {
            let mut request = NewCachedContent::new("gemini-2.5-flash").content("corpus");
            for expiry in &ordering {
                request = request.expiry(expiry.clone());
            }
            let body = serde_json::to_value(&request).expect("serialize");
            let object = body.as_object().expect("object");
            let set = usize::from(object.contains_key("ttl"))
                + usize::from(object.contains_key("expireTime"));
            assert_eq!(
                set, 1,
                "exactly one expiry field should reach the wire: {body}"
            );

            // The last one set is the one that survives.
            match ordering.last().expect("non-empty") {
                CacheExpiry::Ttl(_) => assert!(object.contains_key("ttl"), "{body}"),
                CacheExpiry::ExpireTime(_) => {
                    assert!(object.contains_key("expireTime"), "{body}")
                }
            }
        }
    }

    /// A cache with nothing in it would bill for storage and cache nothing.
    #[test]
    fn emptiness_is_rejected_but_either_payload_alone_suffices() {
        assert!(
            NewCachedContent::new("gemini-2.5-flash")
                .validate()
                .is_err(),
            "an empty cached content should be refused"
        );
        assert!(
            NewCachedContent::new("gemini-2.5-flash")
                .content("corpus")
                .validate()
                .is_ok()
        );
        assert!(
            NewCachedContent::new("gemini-2.5-flash")
                .system_instruction("be brief")
                .validate()
                .is_ok()
        );
        // Display name and expiry are not payload.
        assert!(
            NewCachedContent::new("gemini-2.5-flash")
                .display_name("x")
                .expiry(CacheExpiry::ttl(Duration::from_secs(60)))
                .validate()
                .is_err()
        );
    }

    /// Only the fields a caller actually set reach the wire.
    #[test]
    fn unset_fields_are_omitted_across_every_builder_combination() {
        let always_present = ["model"];
        for (label, request) in [
            (
                "content only",
                NewCachedContent::new("gemini-2.5-flash").content("corpus"),
            ),
            (
                "system only",
                NewCachedContent::new("gemini-2.5-flash").system_instruction("be brief"),
            ),
            (
                "both",
                NewCachedContent::new("gemini-2.5-flash")
                    .content("corpus")
                    .system_instruction("be brief"),
            ),
            (
                "named",
                NewCachedContent::new("gemini-2.5-flash")
                    .content("corpus")
                    .display_name("corpus-v1"),
            ),
        ] {
            let body = serde_json::to_value(&request).expect("serialize");
            let object = body.as_object().expect("object");
            for key in always_present {
                assert!(object.contains_key(key), "{label}: missing {key}");
            }
            for key in ["tools", "toolConfig", "ttl", "expireTime"] {
                assert!(
                    !object.contains_key(key),
                    "{label}: {key} was never set and must not be sent: {body}"
                );
            }
        }
    }

    /// Multiple content blocks accumulate in order — a corpus is usually more
    /// than one document.
    #[test]
    fn content_blocks_accumulate_in_order() {
        let request = NewCachedContent::new("gemini-2.5-flash")
            .content("first")
            .content("second")
            .content("third");
        let body = serde_json::to_value(&request).expect("serialize");
        let contents = body
            .get("contents")
            .and_then(|value| value.as_array())
            .expect("contents array");
        assert_eq!(contents.len(), 3);
        let texts: Vec<&str> = contents
            .iter()
            .filter_map(|entry| {
                entry
                    .get("parts")?
                    .as_array()?
                    .first()?
                    .get("text")?
                    .as_str()
            })
            .collect();
        assert_eq!(texts, vec!["first", "second", "third"]);
    }
}

#[cfg(test)]
mod status_triage_tests {
    //! Non-success triage across every shape a transport can report one in.
    //!
    //! The recorded cassettes only ever exercise the bundled reqwest shape,
    //! `http_client::Error::InvalidStatusCodeWithDetails`. But `H` is a public
    //! extension point (`ClientBuilder::http_client`), and a custom
    //! [`HttpClientExt`] may report the same 404 as a bare
    //! `InvalidStatusCode`, as `InvalidStatusCodeWithMessage`, or as an `Ok`
    //! response carrying the status — shapes rig's own test double produces.
    //! On those the triage used to fall through to `CachedContentError::Http`
    //! or to a bogus deserialization error, so the recovery this module
    //! documents (`Expired { .. } => recreate the cache`) silently never fired.

    use super::*;
    use crate::test_utils::{MockHttpResponse, SequencedHttpClient};

    /// A `cachedContents` client whose transport answers the next request with
    /// `response` and nothing after it.
    fn caches(response: MockHttpResponse) -> CachedContentClient<SequencedHttpClient> {
        Client::builder()
            .api_key("test-key")
            .http_client(SequencedHttpClient::new(vec![response]))
            .build()
            .expect("client should build")
            .cached_contents()
    }

    const GONE: &str =
        r#"{"error":{"code":404,"message":"CachedContent not found (or permission denied)."}}"#;

    /// A transport that reports the 404 as `InvalidStatusCodeWithMessage` —
    /// the variant every non-bundled `HttpClientExt` in rig produces — must
    /// still reach `Expired`.
    ///
    /// Before the triage moved from the variant to the status this fell into
    /// the catch-all `Err(error) => Http(error)` arm, so a caller matching
    /// `Expired` to recreate the cache saw an opaque transport error instead.
    #[tokio::test]
    async fn a_status_error_without_captured_headers_still_reports_expired() {
        let error = caches(MockHttpResponse::error(http::StatusCode::NOT_FOUND, GONE))
            .get("cachedContents/abc123")
            .await
            .expect_err("a missing handle should not resolve");

        let CachedContentError::Expired { name, message } = &error else {
            panic!("a handle that is gone should report Expired: {error:?}");
        };
        assert_eq!(name, "cachedContents/abc123");
        assert!(message.contains("permission denied"), "{message}");
    }

    /// A transport that hands back the 404 as an `Ok` response instead of an
    /// error must reach `Expired` too.
    ///
    /// This is the worse half of the same bug: the error body reached
    /// `serde_json::from_str::<CachedContent>` and failed there, so the call
    /// reported `CachedContentError::Serde` ("missing field `name`") for what
    /// is plainly a 404 — a status-shaped failure disguised as a parse bug.
    #[tokio::test]
    async fn a_non_success_response_is_triaged_rather_than_deserialized() {
        let error = caches(MockHttpResponse::ErrorResponse(
            http::StatusCode::NOT_FOUND,
            GONE.into(),
        ))
        .get("cachedContents/abc123")
        .await
        .expect_err("a missing handle should not resolve");

        let CachedContentError::Expired { name, message } = &error else {
            panic!("an Ok-wrapped 404 should report Expired, not a parse error: {error:?}");
        };
        assert_eq!(name, "cachedContents/abc123");
        assert!(message.contains("permission denied"), "{message}");
    }

    /// Gemini answers a handle that lapsed a while ago with 403 rather than
    /// 404, and both mean the same thing to a caller.
    #[tokio::test]
    async fn a_403_on_an_existing_handle_reports_expired_like_a_404() {
        let error = caches(MockHttpResponse::error(
            http::StatusCode::FORBIDDEN,
            r#"{"error":{"code":403,"message":"You do not have permission to access the CachedContent."}}"#,
        ))
        .delete("cachedContents/abc123")
        .await
        .expect_err("a lapsed handle should not delete");

        assert!(
            matches!(&error, CachedContentError::Expired { name, .. } if name == "cachedContents/abc123"),
            "{error:?}"
        );
    }

    /// A 403 on `create` is not an expiry — there is no handle yet.
    ///
    /// `create` passes `name: None` for exactly this reason: a disabled key or
    /// a project without the API enabled answers 403, and calling that
    /// `Expired` would put a caller into a recreate loop against an API that
    /// will keep refusing.
    #[tokio::test]
    async fn a_403_on_create_is_an_api_error_not_an_expiry() {
        let error = caches(MockHttpResponse::error(
            http::StatusCode::FORBIDDEN,
            r#"{"error":{"code":403,"message":"Generative Language API has not been used in project 1234 before or it is disabled."}}"#,
        ))
        .create(NewCachedContent::new("gemini-2.5-flash").content("corpus"))
        .await
        .expect_err("a refused create should not succeed");

        let CachedContentError::Api { status, message } = &error else {
            panic!("a create that never made a handle cannot be Expired: {error:?}");
        };
        assert_eq!(*status, 403);
        assert!(
            message.contains("has not been used in project"),
            "{message}"
        );
    }

    /// Everything that is not a 403/404 on a named handle is an `Api` failure,
    /// carrying the status a caller needs to decide whether to retry.
    #[tokio::test]
    async fn a_server_error_reports_the_status_rather_than_an_expiry() {
        let error = caches(MockHttpResponse::error(
            http::StatusCode::INTERNAL_SERVER_ERROR,
            r#"{"error":{"code":500,"message":"Internal error encountered."}}"#,
        ))
        .get("cachedContents/abc123")
        .await
        .expect_err("a 500 should not resolve");

        let CachedContentError::Api { status, message } = &error else {
            panic!("a 500 is not an expiry: {error:?}");
        };
        assert_eq!(*status, 500);
        assert!(message.contains("Internal error"), "{message}");
    }

    /// `InvalidStatusCode` carries no body, so there is no provider text to
    /// quote. The message must still say something: an empty one leaves the
    /// error Display ending in a bare colon, which reads as truncated output
    /// rather than as a provider that said nothing.
    #[tokio::test]
    async fn a_status_error_with_no_body_still_names_why_it_has_no_message() {
        // `SequencedHttpClient` reports `InvalidStatusCode(501)` once its
        // scripted responses run out, which is the body-less shape.
        let caches = Client::builder()
            .api_key("test-key")
            .http_client(SequencedHttpClient::new(Vec::new()))
            .build()
            .expect("client should build")
            .cached_contents();

        let error = caches
            .get("cachedContents/abc123")
            .await
            .expect_err("an unscripted request should not resolve");

        let CachedContentError::Api { status, message } = &error else {
            panic!("a 501 is not an expiry: {error:?}");
        };
        assert_eq!(*status, 501);
        assert_eq!(message, NO_RESPONSE_BODY);
        assert!(
            !error.to_string().ends_with(": "),
            "the Display must not trail off: {error}"
        );
    }
}
