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
use crate::wire::{Call, Fold, Operation, Reply};

/// Creates, reads, lists, updates expiry, or deletes explicit context caches.
/// Requests use [`CachedContentRequest`]; a listing reads one page per call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContextCache;

impl Operation for ContextCache {
    type Request = CachedContentRequest;
    type Event = CachedContentReply;
    type Response = CachedContentReply;
    type Fold = CachedContentFold;

    fn is_terminal(_event: &Self::Event) -> bool {
        true
    }

    fn fold(_request: &Self::Request, _call: &mut Call<'_>) -> Self::Fold {
        CachedContentFold::default()
    }
}

/// Keeps the reply's one answer. A reply with no body at all is
/// [`CachedContentReply::Acknowledged`]; finishing never fails.
#[derive(Default)]
pub struct CachedContentFold {
    reply: Option<CachedContentReply>,
}

impl Fold<ContextCache> for CachedContentFold {
    fn absorb(&mut self, reply: &CachedContentReply) -> Result<(), ProviderError> {
        self.reply.get_or_insert_with(|| reply.clone());
        Ok(())
    }

    fn finish(self, _reply: Reply) -> Result<CachedContentReply, ProviderError> {
        Ok(self.reply.unwrap_or_default())
    }
}
