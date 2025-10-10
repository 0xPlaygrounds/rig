use std::path::Path;

use base64::{Engine, prelude::BASE64_STANDARD};
use mime_guess;
use serde_json::{Map, Value};

use crate::{
    http_client::HttpClientExt,
    providers::gemini::completion::gemini_api_types::{
        Blob, Content, GenerateContentRequest, GenerationConfig, Part, PartKind, Role,
    },
    transcription::{self, TranscriptionError},
    wasm_compat::{WasmCompatSend, WasmCompatSync},
};

use super::{Client, completion::gemini_api_types::GenerateContentResponse};

pub use super::completion::{
    GEMINI_1_5_FLASH, GEMINI_1_5_PRO, GEMINI_1_5_PRO_8B, GEMINI_2_0_FLASH,
};

const TRANSCRIPTION_PREAMBLE: &str =
    "Translate the provided audio exactly. Do not add additional information.";

#[derive(Clone)]
pub struct TranscriptionModel<T = reqwest::Client> {
    client: Client<T>,
    /// Name of the model (e.g.: gemini-1.5-flash)
    pub model: String,
}

impl<T> TranscriptionModel<T> {
    pub fn new(client: Client<T>, model: &str) -> Self {
        Self {
            client,
            model: model.to_string(),
        }
    }
}

impl<T> transcription::TranscriptionModel for TranscriptionModel<T>
where
    T: HttpClientExt + WasmCompatSend + WasmCompatSync + Clone,
{
    type Response = GenerateContentResponse;

    #[cfg_attr(feature = "worker", worker::send)]
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

        tracing::debug!(
            "Sending completion request to Gemini API {}",
            serde_json::to_string_pretty(&request)?
        );

        let body = serde_json::to_vec(&request)?;
        let req = self
            .client
            .post(&format!("/v1beta/models/{}:generateContent", self.model))
            .header("Content-Type", "application/json")
            .body(body)
            .map_err(|e| TranscriptionError::HttpError(e.into()))?;

        let response = self.client.send::<_, Vec<u8>>(req).await?;

        if response.status().is_success() {
            let body: GenerateContentResponse =
                serde_json::from_slice(&response.into_body().await?)?;

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

            Ok(transcription::TranscriptionResponse::try_from(body)?)
        } else {
            let text = String::from_utf8_lossy(&response.into_body().await?).into();
            Err(TranscriptionError::ProviderError(text))
        }
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

        let part = candidate.content.parts.first();

        let text = match part {
            Some(Part {
                part: PartKind::Text(text),
                ..
            }) => text,
            None => {
                return Err(TranscriptionError::ResponseError(
                    "Response content contains no text".to_string(),
                ));
            }
            _ => {
                return Err(TranscriptionError::ResponseError(
                    "Response content was not text".to_string(),
                ));
            }
        };

        Ok(transcription::TranscriptionResponse {
            text: text.to_string(),
            response,
        })
    }
}
