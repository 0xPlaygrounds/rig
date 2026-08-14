use std::path::Path;

use base64::{Engine, prelude::BASE64_STANDARD};
use mime_guess;
use serde_json::{Map, Value};

use crate::{
    http_client::HttpClientExt,
    providers::gemini::completion::gemini_api_types::{
        Blob, Content, GenerateContentRequest, GenerationConfig, Part, PartKind, Role,
        visible_text_parts,
    },
    providers::internal::transcription::send_json_transcription,
    transcription::{self, TranscriptionError},
    wasm_compat::{WasmCompatSend, WasmCompatSync},
};

use super::{Client, completion::gemini_api_types::GenerateContentResponse};

const TRANSCRIPTION_PREAMBLE: &str =
    "Translate the provided audio exactly. Do not add additional information.";

pub type TranscriptionModel<T = reqwest::Client> =
    crate::providers::internal::transcription::GenericTranscriptionModel<
        crate::providers::gemini::client::GeminiExt,
        T,
    >;

impl<T> transcription::TranscriptionModel for TranscriptionModel<T>
where
    T: HttpClientExt + WasmCompatSend + WasmCompatSync + Clone + 'static,
{
    type Response = GenerateContentResponse;
    type Client = Client<T>;

    fn make(client: &Self::Client, model: impl Into<String>) -> Self {
        TranscriptionModel::new(client.clone(), model)
    }

    async fn transcription(
        &self,
        request: transcription::TranscriptionRequest,
    ) -> Result<
        transcription::TranscriptionResponse<Self::Response>,
        transcription::TranscriptionError,
    > {
        // Handle Gemini specific parameters
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
            additional_params: None,
        };

        tracing::trace!(
            target: "rig::transcription",
            "Sending completion request to Gemini API {}",
            serde_json::to_string_pretty(&request)?
        );

        let body = serde_json::to_vec(&request)?;

        send_json_transcription(
            &self.client,
            self.client
                .post(format!("/v1beta/models/{}:generateContent", self.model))?,
            body,
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

                transcription::TranscriptionResponse::try_from(body)
            },
        )
        .await
    }
}

impl TryFrom<GenerateContentResponse>
    for transcription::TranscriptionResponse<GenerateContentResponse>
{
    type Error = TranscriptionError;

    fn try_from(response: GenerateContentResponse) -> Result<Self, Self::Error> {
        let candidate = response.candidates.first().ok_or_else(|| {
            TranscriptionError::ResponseError("No response candidates in response".into())
        })?;

        // The transcript is *every* visible text part, concatenated. Reading
        // only `parts.first()` mistook the two shapes Gemini routinely
        // returns here: a thinking model answers with its chain-of-thought in
        // parts[0] (`thought: true`) and the transcript after it, so the
        // reasoning was returned as the transcript and the transcript was
        // dropped; and a transcript split across several text parts kept only
        // the first. `visible_text_parts` is the shared skip-the-thoughts rule
        // — no separator is invented between parts, because Gemini's split
        // points are not sentence boundaries.
        //
        // "No text" stays a *structural* question — are there visible text
        // parts at all — rather than "is the joined string empty". A turn
        // whose text part is genuinely empty still converted before this
        // change, and still does.
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

        Ok(transcription::TranscriptionResponse { text, response })
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
            .err()
            .expect("should fail with non-success status");

        assert!(matches!(error, TranscriptionError::HttpError(_)));
        assert_eq!(
            error.provider_response_status(),
            Some(http::StatusCode::SERVICE_UNAVAILABLE)
        );
        assert_eq!(error.provider_response_body(), Some(body));
    }
}
