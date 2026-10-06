//! Image generation through Gemini's `generateContent` endpoint.
//!
//! ```no_run
//! use rig_core::providers::gemini::{Gemini, image_generation::GEMINI_2_5_FLASH_IMAGE};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let model = Gemini::from_env()?.image_generation(GEMINI_2_5_FLASH_IMAGE);
//! # Ok(())
//! # }
//! ```

use super::completion::usage_of;
use crate::error::{EncodeError, ProviderError};
use crate::image_generation;
use crate::image_generation::ImageGenerationRequest;
use crate::json_utils::Lenient;
use crate::operation::ImageGeneration;
use crate::providers::internal::wire::classify_marker_keyed_frame;
use crate::wire::{
    Body, Decoder, Descriptor, Encoded, Flow, Framing, Mode, Out, Wire, WireEvent, WireFrame,
};
use base64::Engine;
use base64::prelude::BASE64_STANDARD;
use serde_json::{Value, json};

/// `gemini-2.5-flash-image` image generation model, commonly referred to as Nano Banana.
pub const GEMINI_2_5_FLASH_IMAGE: &str = super::completion::GEMINI_2_5_FLASH_IMAGE;

/// The first non-thought image in `reply`, with its usage and identity.
///
/// # Errors
///
/// When `reply` holds no image data, or data that is not base64.
pub fn image_of(reply: &Value) -> Result<image_generation::ImageGenerationResponse, ProviderError> {
    let data = reply
        .arr("candidates")
        .iter()
        .flat_map(|candidate| {
            candidate
                .get("content")
                .map(|content| content.arr("parts"))
                .unwrap_or_default()
        })
        .filter(|part| part.bool("thought") != Some(true))
        .filter_map(|part| part.get("inlineData"))
        .find(|blob| {
            blob.str("mimeType")
                .is_some_and(|mime| mime.starts_with("image/"))
        })
        .and_then(|blob| blob.str("data"))
        .ok_or_else(|| {
            ProviderError::Response(
                "Gemini image generation response did not include image data".into(),
            )
        })?;
    let image = BASE64_STANDARD.decode(data).map_err(|err| {
        ProviderError::Response(format!("Gemini image data was not valid base64: {err}"))
    })?;
    Ok(image_generation::ImageGenerationResponse {
        model: reply.str("modelVersion").map(str::to_owned),
        response_id: Some(reply.str("responseId").unwrap_or_default().to_owned()),
        usage: reply.get("usageMetadata").map(usage_of).unwrap_or_default(),
        ..image_generation::ImageGenerationResponse::new(image)
    })
}

fn create_request_body(generation_request: ImageGenerationRequest) -> Value {
    let mut image_config = serde_json::Map::new();
    if let Some(ratio) = aspect_ratio(generation_request.width, generation_request.height) {
        image_config.insert("aspectRatio".to_owned(), json!(ratio));
    }
    let mut body = json!({
        "contents": [{ "role": "user", "parts": [{ "text": generation_request.prompt }] }],
        "toolConfig": null,
        "generationConfig": { "responseModalities": ["IMAGE"], "imageConfig": image_config },
        "safetySettings": null,
        "systemInstruction": null,
    });
    if let Some(additional_params) = generation_request.additional_params {
        merge_json_deep(&mut body, additional_params);
    }
    body
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

/// The image generation wire: `POST /v1beta/models/{model}:generateContent`.
///
/// Both [`Mode`]s request image output and decode a whole response document.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Images {
    /// The provider this wire speaks to.
    pub provider: super::GeminiConfig,
    /// Name of the model, for example [`GEMINI_2_5_FLASH_IMAGE`].
    pub model: String,
}

impl Images {
    /// The image generation wire for `model`.
    pub fn new(provider: super::GeminiConfig, model: impl Into<String>) -> Self {
        Self {
            provider,
            model: model.into(),
        }
    }
}

impl Wire for Images {
    type Op = ImageGeneration;
    type Payload = crate::wire::Encoded;
    type Frame = crate::wire::WireFrame;
    type Decoder<'id> = ImagesDecoder;
    type Reassembler = crate::wire::document::Unreassembled;

    fn describe(&self) -> Descriptor<'_> {
        Descriptor::new(super::PROVIDER_NAME).model(self.model.as_str())
    }

    fn encode(&self, request: ImageGenerationRequest, _mode: Mode) -> Result<Encoded, EncodeError> {
        let body = serde_json::to_vec(&create_request_body(request))?;
        let request = http::Request::post(format!(
            "{}/v1beta/models/{}:generateContent?key={}",
            self.provider.base_url,
            self.model,
            self.provider.api_key.expose()
        ))
        .header(http::header::CONTENT_TYPE, "application/json")
        .body(Body::Bytes(body))?;
        // Gemini reports no transport request-id header.
        Ok(Encoded::new(request, Framing::Whole))
    }

    fn decoder<'id>(&self) -> Self::Decoder<'id> {
        ImagesDecoder
    }
}

/// Decode the first non-thought image in a `generateContent` reply.
/// Missing image data and invalid base64 produce response errors.
#[derive(Default)]
pub struct ImagesDecoder;

impl<'id> Decoder<'id, ImageGeneration> for ImagesDecoder {
    type Event = Value;

    fn classify(&self, frame: WireFrame) -> WireEvent<Self::Event> {
        classify_marker_keyed_frame(
            &frame.as_str(),
            &["candidates", "promptFeedback", "usageMetadata"],
        )
    }

    fn decode(
        &mut self,
        event: Self::Event,
        out: Out<'id, ImageGeneration>,
    ) -> Result<Flow, ProviderError> {
        Ok(out.end(image_of(&event)?))
    }
}

impl super::GeminiConfig {
    /// The image generation wire.
    pub(crate) fn image_generation(&self, model: impl Into<String>) -> Images {
        Images::new(self.clone(), model)
    }
}

#[cfg(test)]
mod tests;
