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
//! the caller who drives [`super::completion::GenerateContent`] directly and
//! runs the tool loop themselves. A provider-hosted tool is the exception:
//! `codeExecution` runs on Gemini's side and needs no loop, so a cache carrying
//! one is usable from an agent that declares nothing itself.
//!
//! # Example
//!
//! ```no_run
//! use rig_core::providers::gemini;
//! use rig_core::providers::gemini::cached_content::{CacheExpiry, NewCachedContent};
//! use std::time::Duration;
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let provider = gemini::Gemini::from_env()?;
//!
//! // What to cache. `provider.bind(transport).cached_contents().create(..)`
//! // uploads it and hands back a `CachedContent` whose `name` is the handle.
//! let corpus = NewCachedContent::new(gemini::completion::GEMINI_2_5_FLASH)
//!     .system_instruction("You answer questions about the attached corpus.")
//!     .content(std::fs::read_to_string("corpus.txt")?)
//!     .expiry(CacheExpiry::ttl(Duration::from_secs(600)))
//!     .display_name("corpus-v1");
//!
//! // Every request this wire sends reads the cache. Delete the handle when
//! // you are done — storage bills until you do.
//! let wire = provider
//!     .generate_content(gemini::completion::GEMINI_2_5_FLASH)
//!     .with_cached_content("cachedContents/n3v1qk0nqz9k");
//! # let _ = (corpus, wire);
//! # Ok(())
//! # }
//! ```

use std::time::Duration;

use serde::{Deserialize, Serialize};

use super::completion::gemini_api_types::{Content, Part, Role, Tool, ToolConfig};
use crate::operation;
use crate::providers::internal::{
    wire::{classify_or, classify_untyped_line},
    with_query_pairs,
};
use crate::wire::{
    Body, Decoder, Encoded, Framing, Mode, Output, Sink, Wire, WireEvent, WireFrame,
};

/// The `cachedContents` collection path.
const CACHED_CONTENTS_PATH: &str = "/v1beta/cachedContents";

/// Gemini caps a page of `cachedContents` at 1000.
const MAX_PAGE_SIZE: usize = 1000;

crate::provider_response::provider_error_enum! {
    /// A non-success reply is preserved verbatim as [`Self::ProviderResponse`]
    /// with its status, and the status triage every `cachedContents` call
    /// shares — 403 and 404 on an existing handle are the handle being gone —
    /// is `on_handle`, derived from it once the driver has funnelled
    /// every transport shape to the same variant.
    CachedContentError, "cached content" {
        /// The cache handle no longer exists — almost always because its TTL
        /// elapsed.
        ///
        /// Separated from the other failures because it is the one a caller is
        /// expected to *handle* rather than propagate: a cache that expired
        /// mid-run is recreated, not reported. Gemini answers an expired handle
        /// with 403 or 404 depending on how long ago it lapsed, which is why
        /// matching on a status code is not something callers should have to
        /// do. `message` is the provider's own text.
        #[error("gemini cached content `{name}` is expired or was deleted: {message}")]
        Expired { name: String, message: String },

        /// A caller-side mistake caught before the request went out.
        #[error("invalid gemini cached content request: {0}")]
        Invalid(String),

        #[error("could not build the request: {0}")]
        Request(#[from] http::Error),
    }
}

impl CachedContentError {
    /// This failure as seen by a call that addressed the existing handle
    /// `name`: a 403 or 404 there is the handle being gone.
    ///
    /// Both statuses mean "this handle is gone" depending on how long ago it
    /// lapsed; collapsing them spares callers from matching on a status code
    /// to answer one question. The provider's own message rides along: a 403
    /// also covers a disabled key, a project without the API enabled, and
    /// quota denial, and the message is the only text that says which.
    ///
    /// `create` never calls this, deliberately: a 403 there is one of those
    /// other things, and reporting it as `Expired` for a cache that was never
    /// made would send a caller into a recreate loop.
    pub(crate) fn on_handle(self, name: &str) -> Self {
        match self {
            Self::ProviderResponse(response)
                if matches!(
                    response.status,
                    Some(http::StatusCode::FORBIDDEN | http::StatusCode::NOT_FOUND)
                ) =>
            {
                Self::Expired {
                    name: name.to_owned(),
                    message: response.body,
                }
            }
            other => other,
        }
    }
}

crate::error::impl_report_for_provider_error!(
    CachedContentError,
    // An expiry is the provider's verdict on the handle, with its status
    // folded into the variant, so it reports as a provider response rather
    // than as the request fault the table's default arm assumes.
    CachedContentError::Expired { .. } => ErrorKind::ProviderResponse,
);

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
    /// [`super::completion::GenerateContent`] yourself and run the tool loop by
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
    /// for callers driving [`super::completion::GenerateContent`] directly. A
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
    /// [`super::completion::GenerateContent::with_cached_content`] takes.
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

/// One `cachedContents` verb: what [`operation::ContextCache`] sends.
#[derive(Debug)]
pub enum CachedContentRequest {
    /// `POST /v1beta/cachedContents`; answers with the resource.
    Create(NewCachedContent),
    /// `GET /v1beta/cachedContents/<id>`; answers with the resource.
    Get(String),
    /// `GET /v1beta/cachedContents?pageSize=…`, followed on `nextPageToken`;
    /// answers with the pages' entries.
    List,
    /// `PATCH /v1beta/cachedContents/<id>?updateMask=…`; answers with the
    /// resource.
    UpdateExpiry { name: String, expiry: CacheExpiry },
    /// `DELETE /v1beta/cachedContents/<id>`; answers with `{}`.
    Delete(String),
}

/// One page of a `cachedContents` listing.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CachedContentPage {
    /// The entries of every page read, in arrival order — the fold
    /// concatenates them.
    ///
    /// Required, deliberately: this key is what makes a page a page, so a
    /// resource body cannot decode as an empty listing.
    pub cached_contents: Vec<CachedContent>,
    /// The cursor naming the next page, when the listing has one. The
    /// decoder takes it before the page reaches the fold, so a folded
    /// reply's is always `None`.
    #[serde(default)]
    pub next_page_token: Option<String>,
}

/// One `cachedContents` reply: exactly one of the three answers Gemini
/// gives, decided by the shape the body actually has — a page carries
/// `cachedContents`, a resource carries `name`, and an acknowledgement is
/// the empty object.
///
/// Three variants rather than one envelope holding an
/// `Option<CachedContent>` beside a `Vec` and a cursor: that shape let a
/// `delete` carry a resource and a `get` carry a page, and — because a
/// flattened `Option` swallows the deserialization error — read
/// `{"name": 5}` as a resource that was *missing*. Here each shape's
/// decode is strict and none of them accepts another's body, so a
/// malformed resource is the JSON error it is.
#[derive(Debug, Default)]
pub enum CachedContentReply {
    /// `create`, `get` and `update_expiry`: the resource.
    Resource(CachedContent),
    /// `list`: one page of the collection.
    Page(CachedContentPage),
    /// A 2xx with nothing to read — what `delete` is acknowledged with,
    /// and what an empty collection lists as. The default, so a fold that
    /// absorbed nothing holds the reply an empty 2xx already is.
    #[default]
    Acknowledged,
}

impl CachedContentReply {
    /// The resource `create`, `get` and `update_expiry` ask for.
    ///
    /// Another shape is the provider answering a different question, and
    /// says which one it answered instead. A *malformed* resource never
    /// reaches here: a body that names a resource and fails to decode is
    /// a corrupt frame in [`CachedContentsDecoder::classify`], which the
    /// driver turns into this operation's JSON error.
    pub fn resource(self) -> Result<CachedContent, CachedContentError> {
        match self {
            Self::Resource(resource) => Ok(resource),
            other => Err(other.mismatch("one cached content")),
        }
    }

    /// The entries `list` asks for, as the fold concatenated the pages. An
    /// empty collection is answered with the empty object, which is
    /// [`Self::Acknowledged`].
    pub fn entries(self) -> Result<Vec<CachedContent>, CachedContentError> {
        match self {
            Self::Page(page) => Ok(page.cached_contents),
            Self::Acknowledged => Ok(Vec::new()),
            other => Err(other.mismatch("a listing page")),
        }
    }

    /// This reply is not what the verb asked for, and both halves of that
    /// are named — the reply the provider sent is as much of the
    /// diagnosis as the one it was supposed to send.
    fn mismatch(&self, wanted: &str) -> CachedContentError {
        let carried = match self {
            Self::Resource(_) => "one cached content",
            Self::Page(_) => "a listing page",
            Self::Acknowledged => "nothing to read, only a success status",
        };
        CachedContentError::ResponseError(format!("the reply carried {carried}, not {wanted}"))
    }
}

/// Gemini's `cachedContents` resource: the wire for
/// [`operation::ContextCache`].
///
/// Built by [`Gemini::cached_contents`](super::Gemini::cached_contents), or
/// on a socket by `Bound<Gemini, H>::cached_contents()`; the calls are the
/// inherent methods of `Bound<CachedContents, H>`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CachedContents {
    /// The provider this wire speaks to.
    pub provider: super::Gemini,
    /// Entries per listing page, at most 1,000 (Gemini's cap) — the default,
    /// which makes every realistic listing one request.
    ///
    /// A caller holding thousands of caches may want smaller responses, and —
    /// less obviously but more importantly — the cursor-following loop is
    /// otherwise unreachable in a test: proving it works against the live
    /// API would mean creating a thousand billed caches. With a page size of
    /// 1 and three caches it is three pages.
    pub page_size: usize,
}

impl CachedContents {
    /// The wire over `provider`, listing a full page at a time.
    pub fn new(provider: super::Gemini) -> Self {
        Self {
            provider,
            page_size: MAX_PAGE_SIZE,
        }
    }

    /// List `page_size` entries per request.
    pub fn with_page_size(mut self, page_size: usize) -> Self {
        self.page_size = page_size;
        self
    }

    /// One listing page's request, after `page_token` when a page named one.
    ///
    /// Percent-encoded through the same helper the model listing uses:
    /// concatenating the cursor raw would let a `+`, `&`, `=` or `/` in it
    /// truncate the cursor or inject a query parameter — next to the
    /// credential `Gemini::uri` appends — silently dropping pages.
    fn list_request(&self, page_token: Option<&str>) -> Result<http::Request<Body>, http::Error> {
        let page_size = self.page_size.to_string();
        let mut pairs = vec![("pageSize", page_size.as_str())];
        if let Some(token) = page_token {
            pairs.push(("pageToken", token));
        }
        let path = with_query_pairs(CACHED_CONTENTS_PATH, &pairs);
        http::Request::get(self.provider.uri(&path)).body(Body::empty())
    }
}

impl super::Gemini {
    /// Gemini's explicit context cache (`cachedContents`).
    ///
    /// Explicit caching is a different feature from the implicit prefix
    /// caching that happens automatically: it hits on the first request and
    /// across unrelated conversations, at the cost of billing storage per
    /// token-hour. See this module's docs for when each pays.
    pub fn cached_contents(&self) -> CachedContents {
        CachedContents::new(self.clone())
    }
}

impl<H: Clone> crate::driver::Bound<super::Gemini, H> {
    /// The provider's `cached_contents` wire, on this socket.
    pub fn cached_contents(&self) -> crate::driver::Bound<CachedContents, H> {
        crate::driver::Bound::new(self.wire.cached_contents(), self.http.clone())
    }
}

impl Wire for CachedContents {
    type Op = operation::ContextCache;
    type Decoder = CachedContentsDecoder;

    fn name(&self) -> &str {
        super::PROVIDER_NAME
    }

    /// A resource call never streams, so both modes send the one request.
    /// A handle is validated by `resource_path` before anything is built,
    /// and an empty cache is refused before it bills.
    fn encode(
        &self,
        request: CachedContentRequest,
        _mode: Mode,
    ) -> Result<Encoded, CachedContentError> {
        let request = match request {
            CachedContentRequest::Create(new) => {
                new.validate()?;
                http::Request::post(self.provider.uri(CACHED_CONTENTS_PATH))
                    .body(Body::Bytes(serde_json::to_vec(&new)?))?
            }
            CachedContentRequest::Get(name) => {
                http::Request::get(self.provider.uri(&resource_path(&name)?)).body(Body::empty())?
            }
            CachedContentRequest::List => self.list_request(None)?,
            CachedContentRequest::UpdateExpiry { name, expiry } => {
                let (patch, mask) = expiry_patch(expiry)?;
                // The `?` below is only ours because `resource_path` refuses
                // an id that carries one: an unvalidated handle would put
                // `updateMask` inside the caller's query string on a resource
                // we did not mean to patch.
                let path = format!("{}?updateMask={mask}", resource_path(&name)?);
                http::Request::patch(self.provider.uri(&path)).body(Body::Bytes(patch))?
            }
            CachedContentRequest::Delete(name) => {
                http::Request::delete(self.provider.uri(&resource_path(&name)?))
                    .body(Body::empty())?
            }
        };
        Ok(Encoded::new(request, Framing::Whole))
    }

    fn decoder(&self, _mode: Mode) -> Self::Decoder {
        CachedContentsDecoder {
            wire: self.clone(),
            next: None,
        }
    }
}

/// Decodes one `cachedContents` reply and follows a listing's cursor.
pub struct CachedContentsDecoder {
    wire: CachedContents,
    /// The cursor the page just interpreted named, when it named a usable one.
    next: Option<String>,
}

impl Decoder<operation::ContextCache> for CachedContentsDecoder {
    type Event = CachedContentReply;

    /// Which of the three replies this body is, read as the shape it has:
    /// a page names `cachedContents`, an acknowledgement is the empty
    /// object, and anything else must decode as the resource. Each decode
    /// is strict and no shape accepts another's body, so `{"name": 5}`
    /// fails the resource decode and is reported as the defect it is
    /// rather than as a resource that went missing.
    ///
    /// The composition — read one classifier's verdict, try the next shape
    /// when the body was not its kind — is
    /// [`classify_or`]'s, which is where a wire with several reply shapes
    /// is allowed to state it.
    fn classify(&self, frame: WireFrame) -> WireEvent<Self::Event> {
        let body = frame.as_str();
        classify_or(&body, as_page, |data| {
            classify_or(data, as_acknowledgement, as_resource)
        })
    }

    fn interpret(&mut self, mut reply: Self::Event, out: &mut Output<operation::ContextCache>) {
        if let CachedContentReply::Page(page) = &mut reply {
            // An empty cursor counts as absent: re-sending an empty
            // `pageToken` returns the same page forever.
            self.next = page
                .next_page_token
                .take()
                .filter(|token| !token.is_empty());
        }
        out.push(Ok(reply));
    }

    fn continuation(&self) -> Option<http::Request<Body>> {
        self.wire.list_request(Some(self.next.as_deref()?)).ok()
    }
}

/// One listing page, or a body that is not one.
fn as_page(data: &str) -> WireEvent<CachedContentReply> {
    classify_untyped_line::<CachedContentPage>(data.as_bytes()).map(CachedContentReply::Page)
}

/// The empty object a `delete` is acknowledged with, or a body that is not
/// one: any key at all makes the body one of the other two shapes, which
/// is what `deny_unknown_fields` says here.
fn as_acknowledgement(data: &str) -> WireEvent<CachedContentReply> {
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Acknowledgement {}

    classify_untyped_line::<Acknowledgement>(data.as_bytes())
        .map(|_| CachedContentReply::Acknowledged)
}

/// One cached content, or a body that is not one.
fn as_resource(data: &str) -> WireEvent<CachedContentReply> {
    classify_untyped_line::<CachedContent>(data.as_bytes()).map(CachedContentReply::Resource)
}

/// The `PATCH` body and its `updateMask` for one expiry.
///
/// One `match` names the field, which is then both the body's only key and
/// the mask — so the two cannot disagree about which field is written.
fn expiry_patch(expiry: CacheExpiry) -> Result<(Vec<u8>, &'static str), CachedContentError> {
    let (field, value) = match expiry {
        CacheExpiry::Ttl(ttl) => ("ttl", CacheExpiry::ttl_string(ttl)),
        CacheExpiry::ExpireTime(at) => ("expireTime", at),
    };
    let patch = serde_json::Map::from_iter([(field.to_owned(), serde_json::Value::String(value))]);
    Ok((serde_json::to_vec(&patch)?, field))
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
///
/// The ids Gemini hands back are twelve lowercase alphanumerics
/// (`cachedContents/n3v1qk0nqz9k`). `-` and `_` are admitted on top of that
/// because the cassette scrubber rewrites every recorded id to
/// `cached-REDACTED_1` (`test-support/rig-test-support/src/cassettes.rs`), and a replayed test
/// hands that placeholder straight back to `delete`. `.` is deliberately left
/// out: no observed id carries one, and a `..` segment is path traversal.
///
/// Validates rather than interpolating, because this is the path `get`,
/// `update_expiry` and — the one that matters — `delete` send. A handle
/// carrying a `?` does not produce a malformed URL the provider rejects:
/// `Gemini::uri` switches its key separator to `&` the moment it sees
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
    let is_id_char = |ch: char| ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_');
    if id.is_empty() || !id.chars().all(is_id_char) {
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
