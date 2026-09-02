//! Gemini image generation support.

use super::client::{ApiResponse, Client};
use super::completion::gemini_api_types::{
    Content, GenerateContentRequest, GenerateContentResponse, GenerationConfig, ImageConfig, Part,
    PartKind, ResponseModality, Role,
};
use crate::completion::Usage;
use crate::http_client::HttpClientExt;
use crate::image_generation::{
    ImageGenerationError, ImageGenerationRequest, NormalizeImageGenerationResponse,
};
use crate::wasm_compat::WasmCompatSend;
use crate::{http_client, image_generation};
use base64::Engine;
use base64::prelude::BASE64_STANDARD;
use serde_json::Value;

/// `gemini-2.5-flash-image` image generation model, commonly referred to as Nano Banana.
pub const GEMINI_2_5_FLASH_IMAGE: &str = super::completion::GEMINI_2_5_FLASH_IMAGE;

/// Gemini image generation model.
#[derive(Clone)]
pub struct ImageGenerationModel<T = crate::http_client::BoxedHttpClient> {
    client: Client<T>,
    /// Name of the model, for example [`GEMINI_2_5_FLASH_IMAGE`].
    pub model: String,
}

impl<T> ImageGenerationModel<T> {
    pub(crate) fn new(client: Client<T>, model: impl Into<String>) -> Self {
        Self {
            client,
            model: model.into(),
        }
    }
}

impl NormalizeImageGenerationResponse for GenerateContentResponse {
    fn normalize(
        self,
        provider: &str,
    ) -> Result<image_generation::ImageGenerationResponse, ImageGenerationError> {
        let image = first_image_bytes(&self)?;
        let usage = self
            .usage_metadata
            .as_ref()
            .map(Usage::from)
            .unwrap_or_default();

        Ok(
            image_generation::ImageGenerationResponse::new(image, provider)
                .with_optional_model(self.model_version)
                .with_response_id(self.response_id)
                .with_usage(usage),
        )
    }
}

impl<T> ImageGenerationModel<T>
where
    T: HttpClientExt + Clone + WasmCompatSend + 'static,
{
    /// Perform the generation and return Gemini's native
    /// [`GenerateContentResponse`] instead of the normalized
    /// [`image_generation::ImageGenerationResponse`]. Same request, transport,
    /// parser, and error path as
    /// [`image_generation::ImageGenerationModel::image_generation`].
    pub async fn raw_image_generation(
        &self,
        generation_request: ImageGenerationRequest,
    ) -> Result<GenerateContentResponse, ImageGenerationError> {
        let body = serde_json::to_vec(&create_request_body(generation_request)?)?;

        let request = self
            .client
            .post(generate_content_path(&self.model))?
            .body(body)
            .map_err(|e| ImageGenerationError::HttpError(e.into()))?;

        let response = self.client.send(request).await?;

        let status = response.status();
        let text = http_client::text(response).await?;

        if !status.is_success() {
            return Err(ImageGenerationError::from_http_response(status, text));
        }

        match serde_json::from_str::<ApiResponse<GenerateContentResponse>>(&text)? {
            ApiResponse::Ok(response) => Ok(response),
            ApiResponse::Err(err) => {
                tracing::warn!(message = %err.error.message, "provider returned an error response");
                Err(ImageGenerationError::from_http_response(status, text))
            }
        }
    }
}

impl<T> image_generation::ImageGenerationModel for ImageGenerationModel<T>
where
    T: HttpClientExt + Clone + WasmCompatSend + 'static,
{
    async fn image_generation(
        &self,
        generation_request: ImageGenerationRequest,
    ) -> Result<image_generation::ImageGenerationResponse, ImageGenerationError> {
        crate::telemetry::instrument_modality(
            super::completion::PROVIDER_NAME,
            &self.model,
            crate::telemetry::ModalityOperation::ImageGeneration,
            async {
                let response = self.raw_image_generation(generation_request).await?;
                let captured = serde_json::to_value(&response)?;
                Ok(response
                    .normalize(super::completion::PROVIDER_NAME)?
                    .with_raw(captured))
            },
        )
        .await
    }
}

impl<T> crate::client::ConstructImageGenerationModel<Client<T>> for ImageGenerationModel<T>
where
    T: HttpClientExt + Clone + WasmCompatSend + 'static,
{
    fn construct(client: &Client<T>, model: String) -> Self {
        Self::new(client.clone(), model)
    }
}

fn generate_content_path(model: &str) -> String {
    format!("/v1beta/models/{model}:generateContent")
}

fn create_request_body(
    generation_request: ImageGenerationRequest,
) -> Result<Value, ImageGenerationError> {
    let request = GenerateContentRequest {
        contents: vec![Content {
            role: Some(Role::User),
            parts: vec![Part {
                thought: None,
                thought_signature: None,
                part: PartKind::Text(generation_request.prompt),
                additional_params: None,
            }],
        }],
        tools: None,
        tool_config: None,
        generation_config: Some(GenerationConfig {
            response_modalities: Some(vec![ResponseModality::Image]),
            image_config: Some(ImageConfig {
                aspect_ratio: aspect_ratio(generation_request.width, generation_request.height),
                image_size: None,
            }),
            ..Default::default()
        }),
        safety_settings: None,
        system_instruction: None,
        cached_content: None,
        additional_params: None,
    };

    let mut body = serde_json::to_value(request)?;

    if let Some(additional_params) = generation_request.additional_params {
        merge_json_deep(&mut body, additional_params);
    }

    Ok(body)
}

fn merge_json_deep(target: &mut Value, source: Value) {
    match (target, source) {
        (Value::Object(target), Value::Object(source)) => {
            for (key, value) in source {
                if let Some(existing) = target.get_mut(&key) {
                    merge_json_deep(existing, value);
                } else {
                    target.insert(key, value);
                }
            }
        }
        (target, source) => *target = source,
    }
}

fn aspect_ratio(width: u32, height: u32) -> Option<String> {
    match (width, height) {
        (0, _) | (_, 0) => None,
        (w, h) if w == h => Some("1:1".to_string()),
        (w, h) if w.saturating_mul(3) == h.saturating_mul(4) => Some("3:4".to_string()),
        (w, h) if w.saturating_mul(4) == h.saturating_mul(3) => Some("4:3".to_string()),
        (w, h) if w.saturating_mul(9) == h.saturating_mul(16) => Some("9:16".to_string()),
        (w, h) if w.saturating_mul(16) == h.saturating_mul(9) => Some("16:9".to_string()),
        _ => None,
    }
}

fn first_image_bytes(response: &GenerateContentResponse) -> Result<Vec<u8>, ImageGenerationError> {
    for candidate in &response.candidates {
        let Some(content) = &candidate.content else {
            continue;
        };

        for part in &content.parts {
            if part.thought == Some(true) {
                continue;
            }

            if let PartKind::InlineData(inline_data) = &part.part {
                if !inline_data.mime_type.starts_with("image/") {
                    continue;
                }

                return BASE64_STANDARD.decode(&inline_data.data).map_err(|err| {
                    ImageGenerationError::ResponseError(format!(
                        "Gemini image data was not valid base64: {err}"
                    ))
                });
            }
        }
    }

    Err(ImageGenerationError::ResponseError(
        "Gemini image generation response did not include image data".into(),
    ))
}

#[cfg(test)]
mod tests;
