//! Explicit context-cache operations and listing-page aggregation.
//!
//! ```
//! use rig_core::{operation::ContextCache, wire::Operation};
//!
//! assert_eq!(ContextCache::NAME, "cached_content");
//! ```

use super::One;
use crate::error::ProviderError;
use crate::providers::gemini::cached_content::{CachedContentReply, CachedContentRequest};
use crate::wire::{Fold, Operation, Reply};

/// Creates, reads, lists, updates expiry, or deletes explicit context caches.
/// Requests use [`CachedContentRequest`]; listing pages are concatenated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContextCache;

impl Operation for ContextCache {
    type Request = CachedContentRequest;
    type Event = CachedContentReply;
    type Response = CachedContentReply;
    type Capabilities = ();
    type Output = One<Self>;
    type Fold = CachedContentFold;
    type Telemetry = ();

    const NAME: &'static str = "cached_content";

    fn is_terminal(_event: &Self::Event) -> bool {
        true
    }

    fn telemetry(_streaming: bool) -> Self::Telemetry {}
}

/// Concatenates listing pages in arrival order or retains the first resource.
/// Starts with [`CachedContentReply::Acknowledged`]; finishing never fails.
#[derive(Default)]
pub struct CachedContentFold {
    reply: CachedContentReply,
}

impl Fold<ContextCache> for CachedContentFold {
    fn absorb(&mut self, page: CachedContentReply) -> Result<(), ProviderError> {
        match (&mut self.reply, page) {
            (CachedContentReply::Page(held), CachedContentReply::Page(page)) => {
                held.cached_contents.extend(page.cached_contents);
            }
            (CachedContentReply::Acknowledged, page) => self.reply = page,
            // A second answer to a single-document verb, or a page after a
            // resource: the first one stands.
            _ => {}
        }
        Ok(())
    }

    fn finish(self, _reply: Reply) -> Result<CachedContentReply, ProviderError> {
        Ok(self.reply)
    }
}
