//! Test-owned backpressure at the first real tool or text delta. This controls the
//! cancellation boundary, not arbitrary unrestricted transport scheduling.
//! Events are never filtered: releasing the gate resumes the same stream.

use futures::StreamExt;
use rig::error::ProviderError;
use rig::{
    completion::{CompletionModel, CompletionRequest, CompletionResponse, ProviderCapabilities},
    streaming::{Delta, StreamEvent, StreamEvents, StreamingCompletionResponse},
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

/// A model whose stream pauses after the selected first delta.
pub(in super::super) struct FirstDelta<M> {
    model: M,
    boundary: DeltaBoundary,
}

impl<M> FirstDelta<M> {
    /// Pause after the first tool-name or tool-arguments delta.
    pub(in super::super) fn tool(model: M) -> Self {
        Self {
            model,
            boundary: DeltaBoundary::Tool,
        }
    }

    /// Pause after the first text delta.
    pub(in super::super) fn text(model: M) -> Self {
        Self {
            model,
            boundary: DeltaBoundary::Text,
        }
    }
}

impl<M: CompletionModel> CompletionModel for FirstDelta<M> {
    async fn completion(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, ProviderError> {
        self.model.completion(request).await
    }

    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<StreamingCompletionResponse, ProviderError> {
        let stream = self.model.stream(request).await?;
        let provider = stream.provider().to_owned();
        let message_id = stream.message_id.clone();
        let mut gated = StreamingCompletionResponse::from_events(
            provider,
            gate_events(Box::pin(stream), self.boundary, Arc::new(Semaphore::new(0))),
        );
        gated.message_id = message_id;
        Ok(gated)
    }

    fn capabilities(&self) -> ProviderCapabilities {
        self.model.capabilities()
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
