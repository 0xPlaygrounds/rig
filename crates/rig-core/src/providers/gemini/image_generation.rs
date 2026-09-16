//! Gemini image generation support.

use super::completion::gemini_api_types::{
    Content, GenerateContentRequest, GenerateContentResponse, GenerationConfig, ImageConfig, Part,
    PartKind, ResponseModality, Role,
};
use crate::completion::Usage;
use crate::image_generation;
use crate::image_generation::{
    ImageGenerationError, ImageGenerationRequest, NormalizeImageGenerationResponse,
};
use crate::operation::ImageGeneration;
use crate::providers::internal::wire::classify_marker_keyed_frame;
use crate::wire::{
    Body, Decoder, Encoded, Framing, Mode, Output, Sink, Wire, WireEvent, WireFrame,
};
use base64::Engine;
use base64::prelude::BASE64_STANDARD;
use serde_json::Value;

/// `gemini-2.5-flash-image` image generation model, commonly referred to as Nano Banana.
pub const GEMINI_2_5_FLASH_IMAGE: &str = super::completion::GEMINI_2_5_FLASH_IMAGE;

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

/// The image generation wire: `POST /v1beta/models/{model}:generateContent`.
///
/// Gemini generates images through the same `generateContent` endpoint as
/// text, asking for them with `responseModalities: ["IMAGE"]`, and answers
/// with one whole document either way — so both [`Mode`]s send the same
/// request.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Images {
    /// The provider this wire speaks to.
    pub provider: super::Gemini,
    /// Name of the model, for example [`GEMINI_2_5_FLASH_IMAGE`].
    pub model: String,
}

impl Images {
    /// The image generation wire for `model`.
    pub fn new(provider: super::Gemini, model: impl Into<String>) -> Self {
        Self {
            provider,
            model: model.into(),
        }
    }
}

impl Wire for Images {
    type Op = ImageGeneration;
    type Decoder = ImagesDecoder;

    fn name(&self) -> &str {
        super::PROVIDER_NAME
    }

    fn model(&self) -> Option<&str> {
        Some(&self.model)
    }

    fn encode(
        &self,
        request: ImageGenerationRequest,
        _mode: Mode,
    ) -> Result<Encoded, ImageGenerationError> {
        let body = serde_json::to_vec(&create_request_body(request)?)?;
        // The GenerateContent family authenticates through the `key` query
        // parameter, appended last.
        let request = http::Request::post(format!(
            "{}{}?key={}",
            self.provider.base_url,
            generate_content_path(&self.model),
            self.provider.api_key.expose()
        ))
        .header(http::header::CONTENT_TYPE, "application/json")
        .body(Body::Bytes(body))
        .map_err(|error| ImageGenerationError::HttpError(error.into()))?;
        // Gemini reports no transport request-id header.
        Ok(Encoded::new(request, Framing::Whole))
    }

    fn decoder(&self, _mode: Mode) -> Self::Decoder {
        ImagesDecoder
    }
}

/// Decodes one `generateContent` reply into the image it carries.
///
/// `project` stays at its default: this reply carries nothing beyond what the
/// GenerateContent decoder's own projection already observes.
#[derive(Default)]
pub struct ImagesDecoder;

impl Decoder<ImageGeneration> for ImagesDecoder {
    type Event = GenerateContentResponse;

    fn classify(&self, frame: WireFrame) -> WireEvent<Self::Event> {
        classify_marker_keyed_frame(
            &frame.as_str(),
            &["candidates", "promptFeedback", "usageMetadata"],
        )
    }

    /// Picking the image out of the candidates' parts and reporting a
    /// payload that is not base64 is [`NormalizeImageGenerationResponse`]'s,
    /// so this wire holds no second reading of the same reply.
    fn interpret(&mut self, event: Self::Event, out: &mut Output<ImageGeneration>) {
        out.push(event.normalize(super::PROVIDER_NAME));
    }
}

#[cfg(test)]
mod tests;
