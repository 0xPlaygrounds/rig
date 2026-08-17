//! Copilot reasoning roundtrip tests.

use std::sync::{Arc, Mutex};

use futures::StreamExt;
use rig::completion::{
    CompletionError, CompletionModel, CompletionRequest, CompletionResponse, ProviderCapabilities,
};
use rig::prelude::*;
use rig::providers::copilot::{self, CopilotStreamingResponse};
use rig::streaming::{RawStreamingChoice, StreamingCompletionResponse, normalize_stream};

use crate::copilot::{live_responses_model, with_copilot_cassette};
use crate::reasoning::{self, ReasoningRoundtripAgent};

/// Copilot's own terminal stream record carries reasoning metadata that rig's
/// normalized `StreamFinal` does not model, and the roundtrip cassette records
/// exactly one interaction per turn. This wrapper lets the shared roundtrip
/// drive the same requests through `raw_stream` while the test keeps a copy of
/// Copilot's provider-native terminal records.
#[derive(Clone)]
struct CapturingProviderFinals {
    inner: copilot::CompletionModel,
    finals: Arc<Mutex<Vec<CopilotStreamingResponse>>>,
}

impl CapturingProviderFinals {
    fn new(inner: copilot::CompletionModel) -> Self {
        Self {
            inner,
            finals: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn finals(&self) -> Arc<Mutex<Vec<CopilotStreamingResponse>>> {
        Arc::clone(&self.finals)
    }
}

impl CompletionModel for CapturingProviderFinals {
    async fn completion(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, CompletionError> {
        self.inner.completion(request).await
    }

    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<StreamingCompletionResponse, CompletionError> {
        let raw = self.inner.raw_stream(request).await?;
        let finals = self.finals();
        let captured = raw.map(move |item| {
            if let Ok(RawStreamingChoice::FinalResponse(response)) = &item {
                finals
                    .lock()
                    .expect("captured provider finals should not be poisoned")
                    .push(response.clone());
            }

            item
        });

        Ok(StreamingCompletionResponse::stream(
            copilot::PROVIDER_NAME,
            normalize_stream(Box::pin(captured), |response| {
                Ok((copilot::PROVIDER_NAME, response).into())
            }),
        ))
    }

    fn capabilities(&self) -> ProviderCapabilities {
        self.inner.capabilities()
    }
}

#[tokio::test]
async fn streaming() {
    with_copilot_cassette("reasoning_roundtrip/streaming", |client| async move {
        let expected = serde_json::json!({
            "context": "current_turn",
            "effort": "medium",
            "summary": null
        });
        let model = CapturingProviderFinals::new(client.completion_model(live_responses_model()));
        let finals = model.finals();

        reasoning::run_reasoning_roundtrip_streaming(ReasoningRoundtripAgent::new(
            model,
            Some(serde_json::json!({
                "reasoning": { "effort": "medium" }
            })),
        ))
        .await;

        let finals = finals
            .lock()
            .expect("captured provider finals should not be poisoned");
        let response = finals
            .first()
            .expect("Copilot reasoning stream should yield a provider final response");
        let CopilotStreamingResponse::Responses(response) = response else {
            panic!("Copilot reasoning stream should use the Responses route");
        };
        assert_eq!(response.reasoning_context.as_deref(), Some("current_turn"));
        assert_eq!(response.reasoning_metadata.as_ref(), expected.as_object());
    })
    .await;
}

#[tokio::test]
async fn nonstreaming() {
    with_copilot_cassette("reasoning_roundtrip/nonstreaming", |client| async move {
        reasoning::run_reasoning_roundtrip_nonstreaming(ReasoningRoundtripAgent::new(
            client.completion_model(live_responses_model()),
            Some(serde_json::json!({
                "reasoning": { "effort": "medium" }
            })),
        ))
        .await;
    })
    .await;
}
