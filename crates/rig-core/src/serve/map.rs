//! A handler whose completion streams are rewritten item by item, and
//! recorded as rewritten: [`ErasedHandler::map_stream`].

use std::sync::Arc;

use futures::StreamExt;

use crate::{
    effect::{Answered, EffectKind, HandlerDescriptor, family},
    streaming::Relayed,
    tool::PublishedContext,
    wasm_compat::{WasmCompatSend, WasmCompatSync},
};

use super::{Dispatch, ErasedHandler, Reply, Serve};

/// `inner`, its replies' items passed through a fresh `make(published)`
/// each.
pub(crate) struct MapStream<M> {
    inner: ErasedHandler,
    name: String,
    make: M,
}

impl<M> MapStream<M> {
    pub(crate) fn new(inner: ErasedHandler, name: String, make: M) -> Self {
        Self { inner, name, make }
    }
}

impl<M, F> Serve for MapStream<M>
where
    M: Fn(Arc<PublishedContext>) -> F + WasmCompatSend + WasmCompatSync + 'static,
    F: FnMut(Relayed) -> Vec<Relayed> + WasmCompatSend + 'static,
{
    type Family = family::Dynamic;

    fn descriptor(&self) -> HandlerDescriptor {
        let mut descriptor = self.inner.descriptor();
        descriptor.layers.insert(0, self.name.clone());
        descriptor
    }

    fn key(&self) -> crate::effect::HandlerKey {
        self.inner.key().clone()
    }

    fn check(&self, kind: &EffectKind) -> Vec<crate::completion::UnsupportedOption> {
        self.inner.check(kind)
    }

    async fn serve(&self, kind: EffectKind, dispatch: Dispatch) -> Reply {
        // A driver that reads no facts attached none: they go nowhere.
        let published = dispatch.scope::<PublishedContext>().unwrap_or_default();
        // Recorded here, the outermost handler: the record keeps the
        // rewritten reply.
        let events = match self.inner.handle(kind, dispatch).await {
            Reply::Stream(events) => events,
            completion @ Reply::Outcome(Ok(Answered::Completion(_))) => completion.into_stream(),
            other => return other,
        };
        let mut map = (self.make)(published);
        Reply::Stream(Box::pin(events.flat_map(move |item| {
            futures::stream::iter(match item {
                Ok(relayed) => map(relayed).into_iter().map(Ok).collect(),
                Err(report) => vec![Err(report)],
            })
        })))
    }
}

#[cfg(test)]
mod tests;
