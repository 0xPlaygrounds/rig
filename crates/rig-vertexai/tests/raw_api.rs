use rig_core::completion::{CompletionRequest, CompletionResponse};
use rig_core::error::ProviderError;
use rig_core::message::AssistantContent;
use rig_vertexai::completion::{CompletionModel, VertexGenerateContentOutput};

struct SavedResponse(VertexGenerateContentOutput);

async fn capture(
    model: &CompletionModel,
    request: CompletionRequest,
) -> Result<SavedResponse, ProviderError> {
    Ok(SavedResponse(model.raw_completion(request).await?))
}

#[test]
fn typed_raw_response_can_be_stored_and_recovered() -> anyhow::Result<()> {
    // Tie the named type to the real method without constructing a client or making an RPC.
    let _ = capture;
    let raw: VertexGenerateContentOutput = serde_json::from_value(serde_json::json!({
        "responseId": "offline-vertex-response",
        "candidates": [{
            "content": {"role": "model", "parts": [{"text": "hello"}]},
            "finishReason": 1,
            "avgLogprobs": -0.25
        }]
    }))?;
    let saved = SavedResponse(raw);
    let VertexGenerateContentOutput(wire) = saved.0;
    anyhow::ensure!(wire.response_id == "offline-vertex-response");
    anyhow::ensure!(
        wire.candidates
            .first()
            .map(|candidate| candidate.avg_logprobs)
            == Some(-0.25)
    );

    let response: CompletionResponse = VertexGenerateContentOutput(wire).try_into()?;
    anyhow::ensure!(
        matches!(response.choice.as_slice(), [AssistantContent::Text(text)] if text.text == "hello")
    );
    let restored: VertexGenerateContentOutput = serde_json::from_value(response.raw)?;
    anyhow::ensure!(restored.0.response_id == "offline-vertex-response");
    anyhow::ensure!(
        restored
            .0
            .candidates
            .first()
            .map(|candidate| candidate.avg_logprobs)
            == Some(-0.25)
    );
    Ok(())
}
