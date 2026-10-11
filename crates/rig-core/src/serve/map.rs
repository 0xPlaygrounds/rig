//! A handler whose completion streams are rewritten item by item, and
//! recorded as rewritten: [`ErasedHandler::map_stream`].

use futures::StreamExt;

use crate::{
    effect::{EffectKind, HandlerDescriptor, Outcome, family},
    streaming::Relayed,
    wasm_compat::{WasmCompatSend, WasmCompatSync},
};

use super::{Dispatch, ErasedHandler, Reply, Serve};

/// `inner`, its replies' items passed through a fresh `make()` each.
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
    M: Fn() -> F + WasmCompatSend + WasmCompatSync + 'static,
    F: FnMut(Relayed) -> Vec<Relayed> + WasmCompatSend + 'static,
{
    type Family = family::Dynamic;

    fn descriptor(&self) -> HandlerDescriptor {
        let mut descriptor = self.inner.descriptor();
        descriptor.layers.insert(0, self.name.clone());
        descriptor
    }

    async fn serve(&self, kind: EffectKind, dispatch: Dispatch) -> Reply {
        let events = match self.inner.handle(kind, dispatch.unobserved()).await {
            Reply::Stream(events) => events,
            completion @ Reply::Outcome(Ok(Outcome::Completion(_))) => completion.into_stream(),
            other => return other,
        };
        let mut map = (self.make)();
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
