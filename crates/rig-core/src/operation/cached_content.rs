//! The cached-content operation: one verb against a provider's explicit
//! context cache, answered whole.

use super::One;
use crate::providers::gemini::cached_content::{
    CachedContentError, CachedContentReply, CachedContentRequest,
};
use crate::wire::{Fold, Operation, Reply};

/// Managing an explicit context cache: create, read, list, extend, delete.
///
/// A resource lifecycle rather than an assistant turn, but the same shape
/// as every other unary operation: one request, one reply document. The
/// verb is the request ([`CachedContentRequest`]), and the reply is the one
/// of [`CachedContentReply`]'s three shapes the provider answered with. A
/// listing is paged on a cursor the decoder returns from
/// [`Decoder::continuation`](crate::wire::Decoder::continuation), and the
/// fold concatenates the pages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContextCache;

impl Operation for ContextCache {
    type Request = CachedContentRequest;
    type Event = CachedContentReply;
    type Response = CachedContentReply;
    type Error = CachedContentError;
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

/// Takes the one reply a verb answers with, concatenating a listing's pages
/// in arrival order.
///
/// The starting state is [`CachedContentReply::Acknowledged`], which is
/// what a 2xx carrying nothing to read *is* — how `delete` is answered, and
/// how an empty collection lists. So there is no "nothing absorbed yet" to
/// tell apart from it: the fold holds a reply rather than an `Option`, and
/// finishing cannot fail. A second resource is a provider defect and the
/// first one latches, matching [`Take`](super::Take).
#[derive(Default)]
pub struct CachedContentFold {
    reply: CachedContentReply,
}

impl Fold<ContextCache> for CachedContentFold {
    fn absorb(&mut self, page: CachedContentReply) -> Result<(), CachedContentError> {
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

    fn finish(self, _reply: Reply) -> Result<CachedContentReply, CachedContentError> {
        Ok(self.reply)
    }
}
