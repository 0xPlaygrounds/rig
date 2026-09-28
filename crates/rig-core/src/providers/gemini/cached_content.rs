//! Gemini's `cachedContents` resource: create, read, list, extend and delete
//! explicit caches. Storage is billed until a cache expires or is deleted.
//! [`NewCachedContent`](super::NewCachedContent) and
//! [`CachedPrefix`](super::CachedPrefix) describe what a cache holds.
//!
//! ```
//! use rig_core::providers::gemini::CacheExpiry;
//! use std::time::Duration;
//!
//! let hour = CacheExpiry::ttl(Duration::from_secs(3600));
//! # let _ = hour;
//! ```

use crate::wire::Flow;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use super::api::{self, Recognized};
use crate::error::EncodeError;
use crate::error::ProviderError;
use crate::operation;
use crate::providers::internal::{
    wire::{classify_or, classify_untyped_line},
    with_query_pairs,
};
use crate::wire::{
    Body, Decoder, Descriptor, Encoded, Framing, Mode, Out, Wire, WireEvent, WireFrame,
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
    /// Expire `ttl` after creation.
    pub fn ttl(ttl: Duration) -> Self {
        Self::Ttl(ttl)
    }

    /// Expire at the RFC 3339 `timestamp`.
    pub fn expire_time(timestamp: impl Into<String>) -> Self {
        Self::ExpireTime(timestamp.into())
    }

    /// Gemini's duration encoding: fractional seconds with an `s` suffix.
    pub(crate) fn ttl_string(ttl: Duration) -> String {
        format!("{}.{:09}s", ttl.as_secs(), ttl.subsec_nanos())
    }
}

/// One `cachedContents` verb: what [`operation::ContextCache`] sends.
#[derive(Debug)]
pub enum CachedContentRequest {
    /// `POST /v1beta/cachedContents` with a rendered body; answers with the
    /// resource.
    Create(Box<api::CachedContent>),
    /// `GET /v1beta/cachedContents/<id>`; answers with the resource.
    Get(String),
    /// `GET /v1beta/cachedContents?pageSize=…`, after `page_token` when
    /// continuing; answers with one page.
    List {
        /// The previous page's `nextPageToken`; `None` for the first page.
        page_token: Option<String>,
    },
    /// `PATCH /v1beta/cachedContents/<id>?updateMask=…`; answers with the
    /// resource.
    UpdateExpiry { name: String, expiry: CacheExpiry },
    /// `DELETE /v1beta/cachedContents/<id>`; answers with `{}`.
    Delete(String),
}

/// A resource, listing page, or empty acknowledgement from `cachedContents`.
/// Malformed bodies fail decoding rather than representing absent resources.
#[derive(Clone, Debug, Default)]
pub enum CachedContentReply {
    /// `create`, `get` and `update_expiry`: the resource.
    Resource(Box<api::CachedContent>),
    /// `list`: one page of the collection.
    Page(api::ListCachedContentsResponse),
    /// An empty successful reply for deletion or an empty collection.
    /// Also the default for a fold that receives no replies.
    #[default]
    Acknowledged,
}

impl CachedContentReply {
    /// Extract the resource returned by creation, lookup, or expiry update.
    /// Return a response error for a page or acknowledgement.
    pub fn resource(self) -> Result<api::CachedContent, ProviderError> {
        match self {
            Self::Resource(resource) => Ok(*resource),
            other => Err(other.mismatch("one cached content")),
        }
    }

    /// The cursor of the page after this one. An empty cursor counts as
    /// absent: re-sending an empty `pageToken` returns the same page forever.
    pub fn next_page_token(&self) -> Option<String> {
        match self {
            Self::Page(page) => page
                .next_page_token
                .clone()
                .filter(|token| !token.is_empty()),
            _ => None,
        }
    }

    /// The entries of one listing page. An empty collection is answered
    /// with the empty object, which is [`Self::Acknowledged`].
    pub fn entries(self) -> Result<Vec<api::CachedContent>, ProviderError> {
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
/// Built by [`Gemini::cached_contents`](super::Gemini::cached_contents); the
/// calls are the inherent methods of a [`Model`](crate::Model) over it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CachedContents {
    /// The provider this wire speaks to.
    pub provider: super::GeminiConfig,
    /// Requested entries per listing page. Defaults to Gemini's cap of 1,000.
    pub page_size: usize,
}

impl CachedContents {
    /// The wire over `provider`, listing a full page at a time.
    pub fn new(provider: super::GeminiConfig) -> Self {
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

impl super::GeminiConfig {
    /// Build a wire for Gemini's explicit context cache (`cachedContents`).
    pub(crate) fn cached_contents(&self) -> CachedContents {
        CachedContents::new(self.clone())
    }
}

impl Wire for CachedContents {
    type Op = operation::ContextCache;
    type Payload = crate::wire::Encoded;
    type Frame = crate::wire::WireFrame;
    type Decoder<'id> = CachedContentsDecoder;

    fn describe(&self) -> Descriptor<'_> {
        Descriptor::new(super::PROVIDER_NAME)
    }

    /// A resource call never streams, so both modes send the one request.
    /// A handle is validated by `resource_path` before anything is built,
    /// and an empty cache is refused before it bills.
    fn encode(&self, request: CachedContentRequest, _mode: Mode) -> Result<Encoded, EncodeError> {
        let request = match request {
            CachedContentRequest::Create(new) => {
                http::Request::post(self.provider.uri(CACHED_CONTENTS_PATH))
                    .header("Content-Type", "application/json")
                    .body(Body::Bytes(serde_json::to_vec(&new)?))?
            }
            CachedContentRequest::Get(name) => {
                http::Request::get(self.provider.uri(&resource_path(&name)?)).body(Body::empty())?
            }
            CachedContentRequest::List { page_token } => {
                self.list_request(page_token.as_deref())?
            }
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

    fn decoder<'id>(&self) -> Self::Decoder<'id> {
        CachedContentsDecoder
    }
}

/// Decodes one `cachedContents` reply.
pub struct CachedContentsDecoder;

impl<'id> Decoder<'id, operation::ContextCache> for CachedContentsDecoder {
    type Event = CachedContentReply;

    /// Classify a page, empty acknowledgement, or resource.
    /// Malformed bodies remain decoding failures.
    fn classify(&self, frame: WireFrame) -> WireEvent<Self::Event> {
        let body = frame.as_str();
        classify_or(&body, as_page, |data| {
            classify_or(data, as_acknowledgement, as_resource)
        })
    }

    fn decode(
        &mut self,
        reply: Self::Event,
        out: Out<'id, operation::ContextCache>,
    ) -> Result<Flow, ProviderError> {
        Ok(out.end(reply))
    }

    /// A reply with no body at all is an acknowledgement: the status
    /// already answered.
    fn eof(&mut self, out: Out<'id, operation::ContextCache>) -> Result<Flow, ProviderError> {
        Ok(out.end(CachedContentReply::Acknowledged))
    }
}

/// One listing page, or a body that is not one. A page names its entries.
fn as_page(data: &str) -> WireEvent<CachedContentReply> {
    classify_untyped_line::<Recognized<api::ListCachedContentsResponse>>(data.as_bytes())
        .map(|Recognized(page)| CachedContentReply::Page(page))
}

/// Classify an empty object as a deletion acknowledgement; reject any fields.
fn as_acknowledgement(data: &str) -> WireEvent<CachedContentReply> {
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Acknowledgement {}

    classify_untyped_line::<Acknowledgement>(data.as_bytes())
        .map(|_| CachedContentReply::Acknowledged)
}

/// One cached content, or a body that is not one. A resource names itself.
fn as_resource(data: &str) -> WireEvent<CachedContentReply> {
    classify_untyped_line::<Recognized<api::CachedContent>>(data.as_bytes())
        .map(|Recognized(resource)| CachedContentReply::Resource(Box::new(resource)))
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

/// Build `/v1beta/cachedContents/<id>` from a bare id or prefixed handle.
/// Reject empty ids and characters other than ASCII letters, digits, `-`, and
/// `_` to prevent path traversal, query injection, or resource retargeting.
fn resource_path(name: &str) -> Result<String, EncodeError> {
    let id = name.strip_prefix("cachedContents/").unwrap_or(name);
    let is_id_char = |ch: char| ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_');
    if id.is_empty() || !id.chars().all(is_id_char) {
        return Err(EncodeError::request(format!(
            "`{name}` is not a cached content handle `cachedContents/<id>`"
        )));
    }
    Ok(format!("{CACHED_CONTENTS_PATH}/{id}"))
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod status_triage_tests;
