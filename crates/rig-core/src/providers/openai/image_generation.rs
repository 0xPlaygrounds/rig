use super::{OpenAICompletions, OpenAIResponses};
use crate::image_generation;
use crate::image_generation::{
    ImageGenerationError, ImageGenerationRequest, NormalizeImageGenerationResponse,
};
use crate::json_utils::merge_inplace;
use crate::providers::internal::image_generation::{
    GenericImageGenerationModel, JsonImageGenerationProvider, decode_base64_image,
};
use serde::{Deserialize, Serialize};
use serde_json::json;

// ================================================================
// OpenAI Image Generation API
// ================================================================
pub const DALL_E_2: &str = "dall-e-2";
pub const DALL_E_3: &str = "dall-e-3";
pub const GPT_IMAGE_1: &str = "gpt-image-1";
pub const GPT_IMAGE_1_5: &str = "gpt-image-1.5";
pub const GPT_IMAGE_2: &str = "gpt-image-2";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageGenerationData {
    pub b64_json: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageGenerationResponse {
    pub created: i32,
    pub data: Vec<ImageGenerationData>,
}

impl NormalizeImageGenerationResponse for ImageGenerationResponse {
    fn normalize(
        self,
        provider: &str,
    ) -> Result<image_generation::ImageGenerationResponse, ImageGenerationError> {
        let image = decode_base64_image(
            &self,
            |response| response.data.first().map(|image| image.b64_json.as_str()),
            "missing image data",
            None,
        )?;
        Ok(image_generation::ImageGenerationResponse::new(
            image, provider,
        ))
    }
}

/// OpenAI image generation model.
pub type ImageGenerationModel<T = crate::http_client::BoxedHttpClient> =
    GenericImageGenerationModel<OpenAIResponses, T>;

/// OpenAI image generation model for a client using Chat Completions.
pub type CompletionsImageGenerationModel<T = crate::http_client::BoxedHttpClient> =
    GenericImageGenerationModel<OpenAICompletions, T>;

/// Build the `/v1/images/generations` body.
///
/// `response_format` is deliberately absent: it is no longer part of this
/// endpoint's request schema, which rejects it before it even looks at the
/// model — a request naming a model that does not exist still fails on
/// `400 Unknown parameter: 'response_format'` first. Rig used to add it for
/// every model outside a hardcoded `gpt-image-1`/`1.5`/`2` allowlist, so every
/// other image model — `gpt-image-1-mini`, `chatgpt-image-latest`, and any
/// dated snapshot of an allowlisted model such as `gpt-image-2-2026-04-21` —
/// could not generate an image at all. The models this endpoint currently
/// serves answer with `data[].b64_json`, which is what
/// [`decode_base64_image`] reads.
///
/// This is a statement about *this* endpoint. An OpenAI-**compatible** images
/// endpoint reached through the same client may still take the field, and may
/// need it to answer with base64 rather than a URL; such a caller passes it
/// explicitly through `additional_params`, which the merge below now honors.
fn build_request(model: &str, generation_request: ImageGenerationRequest) -> serde_json::Value {
    let mut request = json!({
        "model": model,
        "prompt": generation_request.prompt,
        "size": format!("{}x{}", generation_request.width, generation_request.height),
    });

    // Last, so a caller can reach the endpoint's other parameters (`quality`,
    // `background`, `output_format`, `user`, …) and override what is derived
    // above. xAI's and Gemini's image bodies already honor this field;
    // dropping it here made `ImageGenerationRequestBuilder::additional_params`
    // silently inert for OpenAI.
    //
    // Azure OpenAI's image body (`providers::azure`) has both defects and in a
    // worse combination: it hardcodes `response_format` *and* drops
    // `additional_params`, so an Azure caller cannot even work around the
    // former. Left alone here because a fix that cannot be recorded against
    // Azure would be a guess, which is what this change set is trying not to
    // ship.
    if let Some(additional_params) = generation_request.additional_params {
        merge_inplace(&mut request, additional_params);
    }

    request
}

impl JsonImageGenerationProvider for OpenAIResponses {
    const IMAGE_GENERATION_PATH: &'static str = "/images/generations";
    const PROVIDER_NAME: &'static str = "openai";
    const REQUEST_ID_HEADER: Option<&'static str> = Some("x-request-id");
    type Response = ImageGenerationResponse;

    fn image_generation_request_body(
        model: &str,
        request: ImageGenerationRequest,
    ) -> Result<serde_json::Value, ImageGenerationError> {
        Ok(build_request(model, request))
    }
}

impl JsonImageGenerationProvider for OpenAICompletions {
    const IMAGE_GENERATION_PATH: &'static str = "/images/generations";
    const PROVIDER_NAME: &'static str = "openai";
    const REQUEST_ID_HEADER: Option<&'static str> = Some("x-request-id");
    type Response = ImageGenerationResponse;

    fn image_generation_request_body(
        model: &str,
        request: ImageGenerationRequest,
    ) -> Result<serde_json::Value, ImageGenerationError> {
        Ok(build_request(model, request))
    }
}

#[cfg(test)]
mod tests;
