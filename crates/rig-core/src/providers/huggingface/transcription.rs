use crate::http_client::HttpClientExt;
use crate::providers::huggingface::Client;
use crate::providers::huggingface::completion::ApiResponse;
use crate::providers::internal::transcription::send_json_transcription;
use crate::transcription;
use crate::transcription::{NormalizeTranscriptionResponse, TranscriptionError};
use crate::wasm_compat::WasmCompatSync;
use base64::Engine;
use base64::prelude::BASE64_STANDARD;
use serde_json::json;

pub const WHISPER_LARGE_V3: &str = "openai/whisper-large-v3";
pub const WHISPER_LARGE_V3_TURBO: &str = "openai/whisper-large-v3-turbo";

pub const WHISPER_SMALL: &str = "openai/whisper-small";

pub use crate::providers::openai::TranscriptionResponse;

pub type TranscriptionModel<T = crate::http_client::DefaultHttp> =
    crate::providers::internal::transcription::GenericTranscriptionModel<
        crate::providers::huggingface::client::HuggingFaceExt,
        T,
    >;

impl<T> TranscriptionModel<T>
where
    T: HttpClientExt + Clone + WasmCompatSync + 'static,
{
    /// Perform the transcription and return the provider's native response
    /// (the OpenAI-shaped `{ text }` payload Hugging Face answers with)
    /// instead of the normalized [`transcription::TranscriptionResponse`].
    /// Same request, transport, parser, and error path as
    /// [`transcription::TranscriptionModel::transcription`].
    pub async fn raw_transcription(
        &self,
        request: transcription::TranscriptionRequest,
    ) -> Result<TranscriptionResponse, TranscriptionError> {
        let data = request.data;
        let data = BASE64_STANDARD.encode(data);

        let request = json!({
            "inputs": data
        });

        let route = self
            .client
            .subprovider()
            .transcription_endpoint(&self.model)?;

        let request = serde_json::to_vec(&request)?;

        // Hugging Face reports no transport request-id header.
        send_json_transcription(
            &self.client,
            self.client
                .post(&route)?
                .header("Content-Type", "application/json"),
            request,
            None,
            |status, body| match serde_json::from_slice::<ApiResponse<TranscriptionResponse>>(body)?
            {
                ApiResponse::Ok(response) => Ok(response),
                ApiResponse::Err(err) => {
                    let message = err
                        .get("error")
                        .and_then(|e| {
                            e.as_str()
                                .or_else(|| e.get("message").and_then(|m| m.as_str()))
                        })
                        .or_else(|| err.get("message").and_then(|m| m.as_str()))
                        .unwrap_or_default();
                    tracing::warn!(message = %message, "provider returned an error response");
                    Err(TranscriptionError::from_http_response(
                        status,
                        String::from_utf8_lossy(body),
                    ))
                }
            },
        )
        .await
        .map(|(response, _)| response)
    }
}

impl<T> transcription::TranscriptionModel for TranscriptionModel<T>
where
    T: HttpClientExt + Clone + WasmCompatSync + 'static,
{
    async fn transcription(
        &self,
        request: transcription::TranscriptionRequest,
    ) -> Result<transcription::TranscriptionResponse, TranscriptionError> {
        crate::telemetry::instrument_modality(
            "huggingface",
            &self.model,
            crate::telemetry::ModalityOperation::Transcription,
            async {
                let response = self.raw_transcription(request).await?;
                let captured = serde_json::to_value(&response)?;
                Ok(response.normalize("huggingface")?.with_raw(captured))
            },
        )
        .await
    }
}

impl<T> crate::client::ConstructTranscriptionModel<Client<T>> for TranscriptionModel<T>
where
    T: HttpClientExt + Clone + WasmCompatSync + 'static,
{
    fn construct(client: &Client<T>, model: String) -> Self {
        TranscriptionModel::new(client.clone(), model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::transcription::TranscriptionClient;
    use crate::test_utils::RecordingHttpClient;
    use crate::transcription::TranscriptionModel as _;

    #[tokio::test]
    async fn transcription_non_success_preserves_status_and_body() {
        let body = r#"{"error":{"message":"boom"}}"#;
        let http_client =
            RecordingHttpClient::with_error_response(http::StatusCode::SERVICE_UNAVAILABLE, body);
        let client = Client::builder()
            .api_key("test-key")
            .http_client(http_client)
            .build()
            .expect("build client");
        let model = client.transcription_model(WHISPER_LARGE_V3);

        let request = model.transcription_request().data(vec![0u8; 16]).build();

        let error = model
            .transcription(request)
            .await
            .expect_err("should fail with non-success status");

        assert!(matches!(error, TranscriptionError::HttpError(_)));
        assert_eq!(
            error.provider_response_status(),
            Some(http::StatusCode::SERVICE_UNAVAILABLE)
        );
        assert_eq!(error.provider_response_body(), Some(body));
    }

    #[tokio::test]
    async fn transcription_2xx_error_envelope_preserves_status_and_body() {
        // A 200 OK body that is not a valid `TranscriptionResponse` (no `text`
        // field) falls through the untagged `ApiResponse` to its `Err(Value)`
        // variant, which the provider routes through `from_http_response`.
        let body = r#"{"error":"Model openai/whisper-large-v3 is currently loading"}"#;
        let http_client = RecordingHttpClient::new(body);
        let client = Client::builder()
            .api_key("test-key")
            .http_client(http_client)
            .build()
            .expect("build client");
        let model = client.transcription_model(WHISPER_LARGE_V3);

        let request = model.transcription_request().data(vec![0u8; 16]).build();

        let error = model
            .transcription(request)
            .await
            .expect_err("should fail with provider error envelope");

        match &error {
            TranscriptionError::ProviderResponse(stored) => {
                assert_eq!(stored.body, body);
                assert_eq!(stored.status, Some(http::StatusCode::OK));
            }
            other => panic!("expected ProviderResponse, got {other:?}"),
        }
    }
}
