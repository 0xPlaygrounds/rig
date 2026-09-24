use google_cloud_aiplatform_v1::model::GenerateContentResponse;
use rig_core::Model;
use rig_core::completion::{CompletionRequestBuilder, CompletionResponse};
use rig_core::driver::{Observation, Opened, Transport};
use rig_core::error::ProviderError;
use rig_core::message::AssistantContent;
use rig_core::wire::Mode;
use rig_vertexai::completion::{GEMINI_2_5_FLASH, GenerateContent, VertexRequest};

/// Answers with one stored reply instead of calling Vertex AI.
#[derive(Clone)]
struct Stored(GenerateContentResponse);

impl Transport<GenerateContent> for Stored {
    fn send(
        &self,
        _payload: VertexRequest,
        _mode: Mode,
        _observation: Option<Observation>,
    ) -> Result<
        impl Future<Output = Opened<VertexRequest, GenerateContentResponse>> + Send + 'static + use<>,
        ProviderError,
    > {
        let reply = self.0.clone();
        Ok(async move { Opened::new(futures::stream::iter([Ok(reply)])) })
    }
}

fn complete(reply: GenerateContentResponse) -> Result<CompletionResponse, ProviderError> {
    futures::executor::block_on(
        Model::new(GenerateContent::new(GEMINI_2_5_FLASH), Stored(reply))
            .call(CompletionRequestBuilder::unbound("hello").build(), None),
    )
}

/// A normalized response's `raw` is Vertex AI's own reply: it stores and
/// reads back as the SDK type.
#[test]
fn typed_raw_response_can_be_stored_and_recovered() -> anyhow::Result<()> {
    let wire: GenerateContentResponse = serde_json::from_value(serde_json::json!({
        "responseId": "offline-vertex-response",
        "candidates": [{
            "content": {"role": "model", "parts": [{"text": "hello"}]},
            "finishReason": 1,
            "avgLogprobs": -0.25
        }]
    }))?;
    anyhow::ensure!(wire.response_id == "offline-vertex-response");
    anyhow::ensure!(
        wire.candidates
            .first()
            .map(|candidate| candidate.avg_logprobs)
            == Some(-0.25)
    );

    let response = complete(wire)?;
    anyhow::ensure!(
        matches!(response.choice.as_slice(), [AssistantContent::Text(text)] if text.text == "hello")
    );
    let restored: GenerateContentResponse = serde_json::from_value(response.raw)?;
    anyhow::ensure!(restored.response_id == "offline-vertex-response");
    anyhow::ensure!(
        restored
            .candidates
            .first()
            .map(|candidate| candidate.avg_logprobs)
            == Some(-0.25)
    );
    Ok(())
}
