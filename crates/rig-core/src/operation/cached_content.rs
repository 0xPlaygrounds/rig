//! The cached-content operation: one verb against a provider's explicit
//! context cache, answered whole.

use super::One;
use crate::providers::gemini::cached_content::{
    CachedContentError, CachedContentRequest, CachedContentResponse,
};
use crate::wire::{Fold, Operation, Reply};

/// Managing an explicit context cache: create, read, list, extend, delete.
///
/// A resource lifecycle rather than an assistant turn, but the same shape
/// as every other unary operation: one request, one reply document. The
/// verb is the request ([`CachedContentRequest`]), and the reply envelope
/// ([`CachedContentResponse`]) is read for the part the verb asked for. A
/// listing is paged on a cursor the decoder returns from
/// [`Decoder::continuation`](crate::wire::Decoder::continuation), and the
/// fold concatenates the pages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContextCache;

impl Operation for ContextCache {
    type Request = CachedContentRequest;
    type Event = CachedContentResponse;
    type Response = CachedContentResponse;
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
/// A second reply to a single-document verb is a provider defect and the
/// first one latches, matching [`Take`](super::Take).
#[derive(Default)]
pub struct CachedContentFold {
    reply: Option<CachedContentResponse>,
}

impl Fold<ContextCache> for CachedContentFold {
    fn absorb(&mut self, page: CachedContentResponse) -> Result<(), CachedContentError> {
        match &mut self.reply {
            Some(reply) => reply.cached_contents.extend(page.cached_contents),
            None => self.reply = Some(page),
        }
        Ok(())
    }

    fn finish(self, _reply: Reply) -> Result<CachedContentResponse, CachedContentError> {
        self.reply.ok_or_else(|| {
            CachedContentError::ResponseError("cached content reply carried no payload".to_owned())
        })
    }
}
