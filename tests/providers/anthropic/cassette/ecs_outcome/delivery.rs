//! Test-owned backpressure at the first real tool delta. This controls the
//! cancellation boundary, not arbitrary unrestricted transport scheduling.
//! Events are never filtered: releasing the gate resumes the same stream.

use futures::StreamExt;
use rig::{
    completion::{
        CompletionError, CompletionModel, CompletionRequest, CompletionResponse,
        ProviderCapabilities,
    },
    streaming::{Delta, StreamEvent, StreamEvents, StreamingCompletionResponse},
};
use std::sync::Arc;
use tokio::sync::Semaphore;

#[path = "delivery/tests.rs"]
mod tests;

pub(super) struct FirstToolDelta<M>(M);

impl<M> FirstToolDelta<M> {
    pub(super) fn new(model: M) -> Self {
        Self(model)
    }
}

impl<M: CompletionModel> CompletionModel for FirstToolDelta<M> {
    async fn completion(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, CompletionError> {
        self.0.completion(request).await
    }

    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<StreamingCompletionResponse, CompletionError> {
        let stream = self.0.stream(request).await?;
        let provider = stream.provider().to_owned();
        let message_id = stream.message_id.clone();
        let mut gated = StreamingCompletionResponse::from_events(
            provider,
            gate_events(Box::pin(stream), Arc::new(Semaphore::new(0))),
        );
        gated.message_id = message_id;
        Ok(gated)
    }

    fn capabilities(&self) -> ProviderCapabilities {
        self.0.capabilities()
    }
}

fn gate_events(mut events: StreamEvents, release: Arc<Semaphore>) -> StreamEvents {
    Box::pin(async_stream::stream! {
        let mut crossed = false;
        while let Some(item) = events.next().await {
            let boundary = !crossed && item.as_ref().is_ok_and(|event| matches!(event,
                StreamEvent::BlockDelta {
                    delta: Delta::ToolName { .. } | Delta::ToolArguments { .. }, ..
                }
            ));
            yield item;
            if boundary {
                crossed = true;
                // Keep ownership of the provider stream while the ECS consumer
                // observes the published delta and despawns its dispatch.
                release.acquire().await.expect("delivery gate open").forget();
            }
        }
    })
}
