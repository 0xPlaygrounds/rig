//! Explicit context-cache operations. Listing pages are one call each; the
//! model's `list` follows them.
//!
//! ```
//! use rig_core::operation::ContextCache;
//! use rig_core::wire::Operation;
//!
//! fn caches<Op: Operation>() {}
//! caches::<ContextCache>();
//! ```

use crate::error::ProviderError;
use crate::providers::gemini::cached_content::{CachedContentReply, CachedContentRequest};
use crate::wire::{Call, Fold, Free, Operation, Reply};

/// Creates, reads, lists, updates expiry, or deletes explicit context caches.
/// Requests use [`CachedContentRequest`]; a listing reads one page per call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContextCache;

impl Operation for ContextCache {
    type Request = CachedContentRequest;
    type Event = std::convert::Infallible;
    type End = CachedContentReply;
    type Response = CachedContentReply;
    type Fold = CachedContentFold;
    type Emit = Free;

    fn fold(_request: &Self::Request, _call: &mut Call<'_>) -> Self::Fold {
        CachedContentFold
    }
}

/// The reply's one answer: a reply with no body at all ends as
/// [`CachedContentReply::Acknowledged`]. Finishing never fails.
#[derive(Default)]
pub struct CachedContentFold;

impl Fold<ContextCache> for CachedContentFold {
    fn absorb(&mut self, event: &std::convert::Infallible) -> Result<(), ProviderError> {
        match *event {}
    }

    fn finish(
        self,
        end: CachedContentReply,
        _reply: Reply,
    ) -> Result<CachedContentReply, ProviderError> {
        Ok(end)
    }
}
