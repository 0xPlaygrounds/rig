// ================================================================
//! Google Gemini gRPC Streaming Integration
// ================================================================

use crate::completion::{CompletionError, CompletionRequest};
use crate::streaming;
use futures::stream::{Stream, StreamExt};
use std::pin::Pin;

use super::Client;
use super::proto::GenerateContentResponse;

pub type StreamingCompletionResponse = GenerateContentResponse;

pub(crate) async fn stream(
    client: Client,
    model: String,
    completion_request: CompletionRequest,
) -> Result<streaming::StreamingCompletionResponse<StreamingCompletionResponse>, CompletionError> {
    let request = super::completion::create_grpc_request(model, completion_request)?;

    let mut grpc_client = client
        .ext()
        .grpc_client()
        .map_err(|e| CompletionError::ProviderError(e.to_string()))?;

    let response_stream = grpc_client
        .stream_generate_content(request)
        .await
        .map_err(|e| CompletionError::ProviderError(e.to_string()))?
        .into_inner();

    // Convert gRPC stream to Rig's streaming response
    let stream: Pin<
        Box<
            dyn Stream<
                    Item = Result<
                        streaming::RawStreamingChoice<StreamingCompletionResponse>,
                        CompletionError,
                    >,
                > + Send,
        >,
    > = Box::pin(response_stream.map(|result| {
        result
            .map(|r| streaming::RawStreamingChoice::FinalResponse(r))
            .map_err(|e| CompletionError::ProviderError(e.to_string()))
    }));

    Ok(streaming::StreamingCompletionResponse::stream(stream))
}
