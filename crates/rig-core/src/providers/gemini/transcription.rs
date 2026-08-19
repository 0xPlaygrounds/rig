use std::path::Path;

use base64::{Engine, prelude::BASE64_STANDARD};
use mime_guess;
use serde_json::{Map, Value};

use crate::{
    completion::Usage,
    http_client::HttpClientExt,
    providers::gemini::completion::gemini_api_types::{
        Blob, Content, GenerateContentRequest, GenerationConfig, Part, PartKind, Role,
        visible_text_parts,
    },
    providers::internal::transcription::send_json_transcription,
    transcription::{self, NormalizeTranscriptionResponse, TranscriptionError},
    wasm_compat::{WasmCompatSend, WasmCompatSync},
};

use super::completion::gemini_api_types::GenerateContentResponse;

const TRANSCRIPTION_PREAMBLE: &str =
    "Translate the provided audio exactly. Do not add additional information.";

pub type TranscriptionModel<T = reqwest::Client> =
    crate::providers::internal::transcription::GenericTranscriptionModel<
        crate::providers::gemini::client::GeminiExt,
        T,
    >;

impl<T> TranscriptionModel<T>
where
    T: HttpClientExt + WasmCompatSend + WasmCompatSync + Clone + 'static,
{
    /// Perform the transcription and return Gemini's native
    /// [`GenerateContentResponse`] instead of the normalized
    /// [`transcription::TranscriptionResponse`]. Same request, transport,
    /// parser, and error path as
    /// [`transcription::TranscriptionModel::transcription`].
    pub async fn raw_transcription(
        &self,
        request: transcription::TranscriptionRequest,
    ) -> Result<GenerateContentResponse, TranscriptionError> {
        let additional_params = request
            .additional_params
            .unwrap_or_else(|| Value::Object(Map::new()));
        let mut generation_config = serde_json::from_value::<GenerationConfig>(additional_params)?;

        // Set temperature from completion_request or additional_params
        if let Some(temp) = request.temperature {
            generation_config.temperature = Some(temp);
        }

        let system_instruction = Some(Content {
            parts: vec![TRANSCRIPTION_PREAMBLE.into()],
            role: Some(Role::Model),
        });

        let mime_type =
            if let Some(mime) = mime_guess::from_path(Path::new(&request.filename)).first() {
                mime.to_string()
            } else {
                "audio/mpeg".to_string()
            };

        let request = GenerateContentRequest {
            contents: vec![Content {
                parts: vec![Part {
                    thought: Some(false),
                    thought_signature: None,
                    part: PartKind::InlineData(Blob {
                        mime_type,
                        data: BASE64_STANDARD.encode(request.data),
                    }),
                    additional_params: None,
                }],
                role: Some(Role::User),
            }],
            generation_config: Some(generation_config),
            safety_settings: None,
            tools: None,
            tool_config: None,
            system_instruction,
            cached_content: None,
            additional_params: None,
        };

        tracing::trace!(
            target: "rig::transcription",
            "Sending completion request to Gemini API {}",
            serde_json::to_string_pretty(&request)?
        );

        let body = serde_json::to_vec(&request)?;

        // Gemini sends no transport request-id header.
        send_json_transcription(
            &self.client,
            self.client
                .post(format!("/v1beta/models/{}:generateContent", self.model))?,
            body,
            None,
            |_, body| {
                let body: GenerateContentResponse = serde_json::from_slice(body)?;

                match body.usage_metadata {
                    Some(ref usage) => tracing::info!(target: "rig",
                    "Gemini completion token usage: {}",
                    usage
                    ),
                    None => tracing::info!(target: "rig",
                        "Gemini completion token usage: n/a",
                    ),
                }

                tracing::debug!("Received response");

                Ok(body)
            },
        )
        .await
        .map(|(response, _)| response)
    }
}

impl<T> transcription::TranscriptionModel for TranscriptionModel<T>
where
    T: HttpClientExt + WasmCompatSend + WasmCompatSync + Clone + 'static,
{
    async fn transcription(
        &self,
        request: transcription::TranscriptionRequest,
    ) -> Result<transcription::TranscriptionResponse, TranscriptionError> {
        let response = self.raw_transcription(request).await?;
        let captured = serde_json::to_value(&response)?;
        Ok(response
            .normalize(super::completion::PROVIDER_NAME)?
            .with_raw(captured))
    }
}

impl<T> crate::client::ConstructTranscriptionModel<super::Client<T>> for TranscriptionModel<T>
where
    T: HttpClientExt + WasmCompatSend + WasmCompatSync + Clone + 'static,
{
    fn construct(client: &super::Client<T>, model: String) -> Self {
        TranscriptionModel::new(client.clone(), model)
    }
}

impl NormalizeTranscriptionResponse for GenerateContentResponse {
    fn normalize(
        self,
        provider: &str,
    ) -> Result<transcription::TranscriptionResponse, TranscriptionError> {
        let candidate = self.candidates.first().ok_or_else(|| {
            TranscriptionError::ResponseError("No response candidates in response".into())
        })?;

        let mut parts = candidate
            .content
            .as_ref()
            .map(visible_text_parts)
            .into_iter()
            .flatten()
            .peekable();
        if parts.peek().is_none() {
            return Err(TranscriptionError::ResponseError(
                "Response content contains no text".to_string(),
            ));
        }
        let text = parts.collect::<String>();

        let usage = self
            .usage_metadata
            .as_ref()
            .map(Usage::from)
            .unwrap_or_default();

        Ok(transcription::TranscriptionResponse::new(text, provider)
            .with_optional_model(self.model_version)
            .with_response_id(self.response_id)
            .with_usage(usage))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::transcription::TranscriptionClient;
    use crate::providers::gemini::Client;
    use crate::providers::gemini::completion::GEMINI_2_0_FLASH;
    use crate::test_utils::RecordingHttpClient;
    use crate::transcription::TranscriptionModel as _;

    fn transcription_request() -> transcription::TranscriptionRequest {
        transcription::TranscriptionRequest {
            data: b"audio bytes".to_vec(),
            filename: "audio.mp3".to_string(),
            language: None,
            prompt: None,
            temperature: None,
            additional_params: None,
        }
    }

    #[tokio::test]
    async fn transcription_non_success_preserves_status_and_body() {
        let body = r#"{"error":{"code":503,"message":"boom","status":"UNAVAILABLE"}}"#;
        let http_client =
            RecordingHttpClient::with_error_response(http::StatusCode::SERVICE_UNAVAILABLE, body);
        let client = Client::builder()
            .api_key("test-key")
            .http_client(http_client)
            .build()
            .expect("build client");
        let model = client.transcription_model(GEMINI_2_0_FLASH);

        let error = model
            .transcription(transcription_request())
            .await
            .expect_err("should fail with non-success status");

        assert!(matches!(error, TranscriptionError::HttpError(_)));
        assert_eq!(
            error.provider_response_status(),
            Some(http::StatusCode::SERVICE_UNAVAILABLE)
        );
        assert_eq!(error.provider_response_body(), Some(body));
    }
}
