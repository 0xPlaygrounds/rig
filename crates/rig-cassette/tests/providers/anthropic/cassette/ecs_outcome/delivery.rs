//! Test-owned backpressure at the first real tool or text delta. This controls the
//! cancellation boundary, not arbitrary unrestricted transport scheduling.
//! Events are never filtered: releasing the gate resumes the same stream.

use futures::StreamExt;
use rig::{
    effect::{EffectKind, HandlerDescriptor},
    serve::{Dispatch, Reply, Serve},
    streaming::{Delta, StreamEvent, StreamEvents},
};
use std::sync::Arc;
use tokio::sync::Semaphore;

#[path = "delivery/tests.rs"]
mod tests;

#[derive(Clone, Copy)]
enum DeltaBoundary {
    Tool,
    Text,
}

/// A model handler whose stream pauses after the selected first delta.
pub(in super::super) struct FirstDelta<S> {
    inner: S,
    boundary: DeltaBoundary,
}

impl<S> FirstDelta<S> {
    /// Pause after the first tool-name or tool-arguments delta.
    pub(in super::super) fn tool(inner: S) -> Self {
        Self {
            inner,
            boundary: DeltaBoundary::Tool,
        }
    }

    /// Pause after the first text delta.
    pub(in super::super) fn text(inner: S) -> Self {
        Self {
            inner,
            boundary: DeltaBoundary::Text,
        }
    }
}

impl<S: Serve + 'static> Serve for FirstDelta<S> {
    type Family = S::Family;

    fn descriptor(&self) -> HandlerDescriptor {
        self.inner.descriptor()
    }

    async fn serve(&self, kind: EffectKind, dispatch: Dispatch) -> Reply {
        match self.inner.serve(kind, dispatch).await {
            Reply::Stream(stream) => Reply::Stream(gate_events(
                stream,
                self.boundary,
                Arc::new(Semaphore::new(0)),
            )),
            outcome => outcome,
        }
    }
}

fn gate_events(
    mut events: StreamEvents,
    boundary: DeltaBoundary,
    release: Arc<Semaphore>,
) -> StreamEvents {
    Box::pin(async_stream::stream! {
        let mut crossed = false;
        while let Some(item) = events.next().await {
            let at_boundary = !crossed && item.as_ref().is_ok_and(|event| match event {
                StreamEvent::BlockDelta {
                    delta: Delta::ToolName { .. } | Delta::ToolArguments { .. }, ..
                } => matches!(boundary, DeltaBoundary::Tool),
                StreamEvent::BlockDelta { delta: Delta::Text { .. }, .. } => {
                    matches!(boundary, DeltaBoundary::Text)
                }
                _ => false,
            });
            yield item;
            if at_boundary {
                crossed = true;
                // Keep ownership of the provider stream while the ECS consumer
                // observes the published delta and despawns its dispatch.
                release.acquire().await.expect("delivery gate open").forget();
            }
        }
    })
}
