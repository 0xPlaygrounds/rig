//! Gemini explicit context caching through the `cachedContents` resource.
//! Upload content once and reuse its handle across requests. Storage is billed
//! until the cache expires or is deleted.
//!
//! Requests using a cache must not supply their own system instruction, tools,
//! or tool configuration. Cached function declarations require a caller-managed
//! tool loop; provider-hosted tools execute on Gemini.
//!
//! ```no_run
//! use rig_core::providers::gemini::cached_content::{CacheExpiry, NewCachedContent};
//! use rig_core::providers::gemini::completion::GEMINI_2_5_FLASH;
//! use std::time::Duration;
//!
//! let corpus = NewCachedContent::new(GEMINI_2_5_FLASH)
//!     .content("A reusable document corpus")
//!     .expiry(CacheExpiry::ttl(Duration::from_secs(600)));
//! ```

use std::time::Duration;

use serde::{Deserialize, Serialize};

use super::completion::gemini_api_types::{Content, Part, Role, Tool, ToolConfig};
use crate::error::EncodeError;
use crate::error::ProviderError;
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

/// Converts a 403 or 404 reply for the existing handle `name` to
/// [`ProviderError::CacheExpired`], keeping the reply. Other failures are
/// unchanged. Call only for existing handles, never for cache creation.
pub(crate) fn on_handle(error: ProviderError, name: &str) -> ProviderError {
    match error {
        ProviderError::ProviderResponse(response)
            if matches!(
                response.status,
                Some(http::StatusCode::FORBIDDEN | http::StatusCode::NOT_FOUND)
            ) =>
        {
            ProviderError::CacheExpired {
                name: name.to_owned(),
                response,
            }
        }
        other => other,
    }
}

/// A relative lifetime or absolute expiry time for cached content.
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

/// Content and configuration for creating a cache.
/// Supply content or a system instruction before creation. The expiry builder
/// keeps relative and absolute expiry mutually exclusive.
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

    /// Set the tools inherited by requests using this cache.
    /// Requests must not supply their own tools. Cached function declarations
    /// require a caller-managed tool loop; provider-hosted tools do not.
    pub fn tools(mut self, tools: Vec<Tool>) -> Self {
        self.tools = Some(tools);
        self
    }

    /// Set the tool configuration inherited by requests using this cache.
    /// Requests must not supply their own tool configuration. May be set
    /// without a tool set.
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

    fn validate(&self) -> Result<(), EncodeError> {
        if self.contents.is_empty() && self.system_instruction.is_none() {
            return Err(EncodeError::request(
                "a cached content needs contents or a system instruction; an empty cache would \
                 bill for storage and cache nothing",
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
    /// [`ProviderError::CacheExpired`].
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
    /// Entries in arrival order, concatenated across pages when folded.
    /// Required during deserialization to distinguish pages from resources.
    pub cached_contents: Vec<CachedContent>,
    /// The cursor naming the next page, when the listing has one. The
    /// decoder takes it before the page reaches the fold, so a folded
    /// reply's is always `None`.
    #[serde(default)]
    pub next_page_token: Option<String>,
}

/// A resource, listing page, or empty acknowledgement from `cachedContents`.
/// Malformed bodies fail decoding rather than representing absent resources.
#[derive(Debug, Default)]
pub enum CachedContentReply {
    /// `create`, `get` and `update_expiry`: the resource.
    Resource(CachedContent),
    /// `list`: one page of the collection.
    Page(CachedContentPage),
    /// An empty successful reply for deletion or an empty collection.
    /// Also the default for a fold that receives no replies.
    #[default]
    Acknowledged,
}

impl CachedContentReply {
    /// Extract the resource returned by creation, lookup, or expiry update.
    /// Return a response error for a page or acknowledgement.
    pub fn resource(self) -> Result<CachedContent, ProviderError> {
        match self {
            Self::Resource(resource) => Ok(resource),
            other => Err(other.mismatch("one cached content")),
        }
    }

    /// The entries `list` asks for, as the fold concatenated the pages. An
    /// empty collection is answered with the empty object, which is
    /// [`Self::Acknowledged`].
    pub fn entries(self) -> Result<Vec<CachedContent>, ProviderError> {
        match self {
            Self::Page(page) => Ok(page.cached_contents),
            Self::Acknowledged => Ok(Vec::new()),
            other => Err(other.mismatch("a listing page")),
        }
    }

    /// Build a response error naming the actual and expected reply shapes.
    fn mismatch(&self, wanted: &str) -> ProviderError {
        let carried = match self {
            Self::Resource(_) => "one cached content",
            Self::Page(_) => "a listing page",
            Self::Acknowledged => "nothing to read, only a success status",
        };
        ProviderError::Response(format!("the reply carried {carried}, not {wanted}"))
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
    /// Requested entries per listing page. Defaults to Gemini's cap of 1,000.
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

    /// Build a listing request, optionally continuing after `page_token`.
    /// Percent-encoding preserves the cursor and prevents query injection.
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
    /// Build a wire for Gemini's explicit context cache (`cachedContents`).
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
    fn encode(&self, request: CachedContentRequest, _mode: Mode) -> Result<Encoded, EncodeError> {
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
                // Handle validation prevents query injection and retargeting the patch.
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

    /// Classify a page, empty acknowledgement, or resource.
    /// Malformed bodies remain decoding failures.
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

/// Classify an empty object as a deletion acknowledgement; reject any fields.
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

/// Serialize the expiry patch with an update mask naming its only field.
fn expiry_patch(expiry: CacheExpiry) -> Result<(Vec<u8>, &'static str), EncodeError> {
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

/// Build `/v1beta/cachedContents/<id>` from a bare id or prefixed handle.
/// Reject empty ids and characters other than ASCII letters, digits, `-`, and
/// `_` to prevent path traversal, query injection, or resource retargeting.
fn resource_path(name: &str) -> Result<String, EncodeError> {
    let id = name.strip_prefix("cachedContents/").unwrap_or(name);
    let is_id_char = |ch: char| ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_');
    if id.is_empty() || !id.chars().all(is_id_char) {
        return Err(EncodeError::request(format!(
            "`{name}` is not a cached content handle; expected `cachedContents/<id>` or a bare \
                 `<id>` of letters, digits, `-` and `_`. The id is spliced into the request path, \
                 where a `?`, `#` or `/` silently retargets the call at a different resource — \
                 and this is the path that deletes"
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
